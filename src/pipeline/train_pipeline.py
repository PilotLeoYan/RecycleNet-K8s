"""End-to-end distributed training pipeline orchestrator for RecycleNet."""

import gc
import os
import tempfile
from datetime import datetime
from typing import Any

import numpy as np
import ray.train
import ray.train.torch
import torch
from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation
from src.components.evals import TestDistributed
from src.components.log_model import LogModel
from src.components.loss_functions import get_criterion
from src.components.model import build_mobilenet_v3
from src.components.optimizers import get_optimizer, get_scheduler
from src.components.train_step import TrainStep
from src.config.schema import AppConfig
from src.pipeline.ray_resources import assign_ray_resources
from src.pipeline.reproducibility import make_reproducibility
from src.utils import RecycleNetException, get_logger

logger = get_logger(__name__)


def train_loop_per_worker(
    config: dict[str, Any],
) -> None:
    """Execute the distributed training and validation loop for a single Ray worker.

    Args:
        config: Dictionary containing dataset paths, hyperparameters, and settings.
    """
    try:
        make_reproducibility(config["reproducibility"])
        worker_device = ray.train.torch.get_device()

        transformation = DataTransformation(
            config["transformation"],
            seed=config["reproducibility"].torch_seed,
        )
        train_loader, valid_loader, test_loader = transformation.get_dataloaders(
            config["raw_path"]
        )
        train_loader = ray.train.torch.prepare_data_loader(train_loader)

        model = build_mobilenet_v3(len(transformation.classes))
        model = ray.train.torch.prepare_model(model)

        criterion = get_criterion()
        optimizer = get_optimizer(
            filter(lambda p: p.requires_grad, model.parameters()),
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=config.get("scheduler", "cosine"),
            epochs=config["epochs"],
            min_lr=config.get("min_lr", 1e-5),
            warmup_epochs=config.get("warmup_epochs", 0),
            patience=max(1, config.get("patience", 5) // 2),
        )

        trainer = TrainStep(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=worker_device,
        )

        best_vloss = float("inf")
        best_model_weights = None
        history: list[dict[str, float]] = []
        patience_counter = 0
        patience = config.get("patience", 5)

        try:
            world_rank = ray.train.get_context().get_world_rank()
        except RuntimeError:
            world_rank = 0

        for epoch in range(config["epochs"]):
            sampler = getattr(train_loader, "sampler", None)
            if sampler and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            metrics = trainer.train_valid_step()
            current_lr = float(optimizer.param_groups[0]["lr"])
            metrics["learning_rate"] = current_lr

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics["valid_loss"])
                else:
                    scheduler.step()

            epoch_record: dict[str, float] = {
                "train_loss": float(metrics["train_loss"]),
                "valid_loss": float(metrics["valid_loss"]),
                "valid_accuracy": float(metrics.get("valid_accuracy", 0.0)),
                "valid_precision": float(metrics.get("valid_precision", 0.0)),
                "valid_recall": float(metrics.get("valid_recall", 0.0)),
                "valid_f1_score": float(metrics.get("valid_f1_score", 0.0)),
                "learning_rate": current_lr,
            }
            history.append(epoch_record)

            logger.info(
                "Epoch [%d/%d] - train_loss: %.4f - valid_loss: %.4f - "
                "valid_accuracy: %.4f - valid_f1: %.4f",
                epoch + 1,
                config["epochs"],
                epoch_record["train_loss"],
                epoch_record["valid_loss"],
                epoch_record["valid_accuracy"],
                epoch_record["valid_f1_score"],
            )

            # --- Real-time MLflow logging (rank 0 only to avoid concurrent writes) ---
            mlflow_run_id = config.get("mlflow_run_id")
            tracking_uri = config.get("tracking_uri")
            if world_rank == 0 and mlflow_run_id and tracking_uri:
                try:
                    client = MlflowClient(tracking_uri=tracking_uri)
                    for key, value in epoch_record.items():
                        client.log_metric(
                            run_id=mlflow_run_id,
                            key=key,
                            value=value,
                            step=epoch,
                        )
                except Exception as exc:
                    logger.debug(
                        "Real-time MLflow worker metric logging skipped: %s", exc
                    )

            # --- Early stopping check ---
            if metrics["valid_loss"] < best_vloss:
                best_vloss = metrics["valid_loss"]
                patience_counter = 0
                raw_model = (
                    model.module
                    if isinstance(model, DistributedDataParallel)
                    else model
                )
                best_model_weights = {
                    k: v.cpu().clone() for k, v in raw_model.state_dict().items()
                }
            else:
                patience_counter += 1

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                raw_model = (
                    model.module
                    if isinstance(model, DistributedDataParallel)
                    else model
                )
                state_dict = raw_model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": state_dict,
                        "optimizer_state": optimizer.state_dict(),
                    },
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )

                ray.train.report(
                    metrics=metrics,
                    checkpoint=Checkpoint.from_directory(temp_checkpoint_dir),
                )

            if patience_counter >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch + 1,
                    patience,
                )
                break

        if best_model_weights is not None:
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model.load_state_dict(best_model_weights)

        test_loader = ray.train.torch.prepare_data_loader(test_loader)
        test_distributed = TestDistributed(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            num_classes=len(transformation.classes),
            classes_name=transformation.classes,
            device=worker_device,
        )
        eval_metrics = test_distributed.compute_distributed_test_metrics()
        eval_metrics["history"] = history
        ray.train.report(eval_metrics)

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class TrainPipeline:
    """Orchestrate Ray Train execution, MLflow tracking, and model registration.

    Attributes:
        config: Application configuration settings.
        logmodel: MLflow tracking and model registry helper.
        ingestion: Dataset ingestion helper.
        transformation: Data transformation and pipeline preparation component.
    """

    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        """Initialize TrainPipeline with configuration and reproducibility seeds.

        Args:
            config: Root application configuration object.
        """
        self.config = config
        make_reproducibility(config.reproducibility)
        self.logmodel = LogModel(config.tracking)
        self.ingestion = DataIngestion(config.ingestion)
        self.transformation = DataTransformation(
            config.transformation,
            seed=config.reproducibility.torch_seed,
        )

    def _log_test_evaluation(self, test_record: dict[str, Any]) -> None:
        """Log test evaluation metrics and confusion matrix to MLflow.

        Args:
            test_record: Dictionary containing test metrics and optional
                confusion matrix.
        """
        fig_cm = test_record.get("confusion_matrix")
        if fig_cm is not None and not isinstance(fig_cm, Figure):
            fig_cm = None

        test_metrics = {
            "test_loss": float(test_record["test_loss"]),
            "accuracy": float(
                test_record.get("accuracy", test_record.get("test_accuracy", 0.0))
            ),
            "precision": float(
                test_record.get("precision", test_record.get("test_precision", 0.0))
            ),
            "recall": float(
                test_record.get("recall", test_record.get("test_recall", 0.0))
            ),
            "f1_score": float(
                test_record.get("f1_score", test_record.get("test_f1_score", 0.0))
            ),
        }
        self.logmodel.log_test(
            roc=float(test_record.get("auc", 0.0)),
            metrics=test_metrics,
            fig_cm=fig_cm,
            per_class_metrics=test_record.get("per_class_metrics"),
        )
        logger.info(
            "MLflow Test Evaluation Logged - test_loss: %.4f, "
            "test_acc: %.4f, test_f1: %.4f, test_roc: %.4f",
            test_metrics["test_loss"],
            test_metrics["accuracy"],
            test_metrics["f1_score"],
            float(test_record.get("auc", 0.0)),
        )

    def run(self) -> None:
        """Execute the distributed training, logging, and evaluation workflow.

        Raises:
            RecycleNetException: If an error occurs during distributed training
                or registration.
        """
        logger.info("Starting distributed training pipeline...")

        try:
            effective_device, resources = assign_ray_resources(
                self.config.training.device,
                self.config.training.cpu_resources_per_worker,
                self.config.training.gpu_resources_per_worker,
            )
            use_gpu = effective_device == "cuda"

            scaling_config = ScalingConfig(
                num_workers=self.config.training.num_workers,
                use_gpu=use_gpu,
                resources_per_worker=resources,
            )

            checkpoint_config = CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="valid_loss",
                checkpoint_score_order="min",
            )

            failure_config = FailureConfig(
                max_failures=0,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.config.tracking.experiment_name}_{timestamp}"

            run_config = RunConfig(
                name=run_name,
                checkpoint_config=checkpoint_config,
                failure_config=failure_config,
            )

            dataset_dir = self.ingestion.extract_dataset().resolve()
            logger.info("Dataset extracted at: %s", dataset_dir)

            self.transformation.discover_classes(dataset_dir)
            num_classes = len(self.transformation.classes)
            logger.info(
                "Detected [%d] classes: %s", num_classes, self.transformation.classes
            )

            train_loop_config: dict[str, Any] = {
                "raw_path": dataset_dir,
                "transformation": self.config.transformation,
                "reproducibility": self.config.reproducibility,
                "learning_rate": self.config.training.learning_rate,
                "weight_decay": self.config.training.weight_decay,
                "epochs": self.config.training.epochs,
                "patience": self.config.training.patience,
                "scheduler": self.config.training.scheduler,
                "min_lr": self.config.training.min_lr,
                "warmup_epochs": self.config.training.warmup_epochs,
                "tracking_uri": self.config.tracking.tracking_uri,
            }

            split_dist: dict[str, Any] | None = None
            try:
                split_dist = self.transformation.get_split_distribution(dataset_dir)
            except Exception as split_exc:
                logger.debug("Split distribution computation skipped: %s", split_exc)

            self.logmodel.set_experiment_tracking()
            pipeline_tags = self.config.tracking.tags.copy()
            pipeline_tags["hardware"] = effective_device

            with self.logmodel.start_run() as active_run:
                logger.info("Active MLflow Run ID: %s", active_run.info.run_id)

                # Inject the active run_id so workers can log in real-time to this run
                train_loop_config["mlflow_run_id"] = active_run.info.run_id

                trainer = TorchTrainer(
                    train_loop_per_worker=train_loop_per_worker,
                    train_loop_config=train_loop_config,
                    scaling_config=scaling_config,
                    run_config=run_config,
                )

                idx_to_class = {
                    str(idx): name
                    for idx, name in enumerate(self.transformation.classes)
                }

                self.logmodel.log_pipeline_metadata(
                    idx_to_class=idx_to_class,
                    tags=pipeline_tags,
                    params={
                        "epochs": self.config.training.epochs,
                        "patience": self.config.training.patience,
                        "batch_size": self.config.transformation.batch_size,
                        "learning_rate": self.config.training.learning_rate,
                        "weight_decay": self.config.training.weight_decay,
                        "scheduler": self.config.training.scheduler,
                        "min_lr": self.config.training.min_lr,
                        "warmup_epochs": self.config.training.warmup_epochs,
                        "num_workers": self.config.training.num_workers,
                    },
                    config_dict=self.config.model_dump(mode="json"),
                    split_distribution=split_dist,
                )

                logger.info("Running Ray distributed training...")
                results = trainer.fit()

                best_checkpoint = results.checkpoint
                if best_checkpoint is None:
                    raise RecycleNetException(
                        "Ray training completed without generating a valid checkpoint."
                    )

                test_logged = False

                if (
                    isinstance(results.metrics, dict)
                    and isinstance(results.metrics.get("history"), list)
                    and len(results.metrics["history"]) > 0
                ):
                    for epoch_step, record in enumerate(results.metrics["history"]):
                        train_loss = float(record["train_loss"])
                        valid_loss = float(record["valid_loss"])
                        valid_metrics = {
                            "accuracy": float(record.get("valid_accuracy", 0.0)),
                            "precision": float(record.get("valid_precision", 0.0)),
                            "recall": float(record.get("valid_recall", 0.0)),
                            "f1_score": float(record.get("valid_f1_score", 0.0)),
                        }
                        lr_val = (
                            float(record["learning_rate"])
                            if "learning_rate" in record
                            and record["learning_rate"] is not None
                            else None
                        )
                        self.logmodel.log_epoch(
                            train_loss=train_loss,
                            valid_loss=valid_loss,
                            valid_metrics=valid_metrics,
                            step=epoch_step,
                            lr=lr_val,
                        )
                        logger.info(
                            "MLflow Epoch [%d/%d] Logged - "
                            "train_loss: %.4f, valid_loss: %.4f, "
                            "valid_accuracy: %.4f, valid_f1: %.4f",
                            epoch_step + 1,
                            self.config.training.epochs,
                            train_loss,
                            valid_loss,
                            valid_metrics["accuracy"],
                            valid_metrics["f1_score"],
                        )
                elif (
                    results.metrics_dataframe is not None
                    and not results.metrics_dataframe.empty
                ):
                    records = results.metrics_dataframe.to_dict(orient="records")
                    epoch_step = 0
                    for record in records:
                        if (
                            "train_loss" in record
                            and record["train_loss"] is not None
                            and not np.isnan(record["train_loss"])
                        ):
                            train_loss = float(record["train_loss"])
                            valid_loss = float(record["valid_loss"])
                            valid_metrics = {
                                "accuracy": float(record.get("valid_accuracy", 0.0)),
                                "precision": float(record.get("valid_precision", 0.0)),
                                "recall": float(record.get("valid_recall", 0.0)),
                                "f1_score": float(record.get("valid_f1_score", 0.0)),
                            }
                            lr_val = (
                                float(record["learning_rate"])
                                if "learning_rate" in record
                                and record["learning_rate"] is not None
                                and not (
                                    isinstance(record["learning_rate"], float)
                                    and np.isnan(record["learning_rate"])
                                )
                                else None
                            )
                            self.logmodel.log_epoch(
                                train_loss=train_loss,
                                valid_loss=valid_loss,
                                valid_metrics=valid_metrics,
                                step=epoch_step,
                                lr=lr_val,
                            )
                            logger.info(
                                "MLflow Epoch [%d/%d] Logged - "
                                "train_loss: %.4f, valid_loss: %.4f, "
                                "valid_accuracy: %.4f, valid_f1: %.4f",
                                epoch_step + 1,
                                self.config.training.epochs,
                                train_loss,
                                valid_loss,
                                valid_metrics["accuracy"],
                                valid_metrics["f1_score"],
                            )
                            epoch_step += 1

                        if (
                            not test_logged
                            and "test_loss" in record
                            and record["test_loss"] is not None
                            and not np.isnan(record["test_loss"])
                        ):
                            self._log_test_evaluation(record)
                            test_logged = True
                elif results.metrics:
                    last_metrics = results.metrics
                    if "train_loss" in last_metrics and "valid_loss" in last_metrics:
                        lr_val = (
                            float(last_metrics["learning_rate"])
                            if "learning_rate" in last_metrics
                            and last_metrics["learning_rate"] is not None
                            else None
                        )
                        self.logmodel.log_epoch(
                            train_loss=float(last_metrics["train_loss"]),
                            valid_loss=float(last_metrics["valid_loss"]),
                            valid_metrics={
                                "accuracy": float(
                                    last_metrics.get("valid_accuracy", 0.0)
                                ),
                                "precision": float(
                                    last_metrics.get("valid_precision", 0.0)
                                ),
                                "recall": float(last_metrics.get("valid_recall", 0.0)),
                                "f1_score": float(
                                    last_metrics.get("valid_f1_score", 0.0)
                                ),
                            },
                            step=0,
                            lr=lr_val,
                        )
                    if (
                        not test_logged
                        and "test_loss" in last_metrics
                        and last_metrics["test_loss"] is not None
                    ):
                        self._log_test_evaluation(last_metrics)
                        test_logged = True

                # Ensure test evaluation metrics reported by rank 0 are always logged
                if (
                    not test_logged
                    and results.metrics
                    and "test_loss" in results.metrics
                    and results.metrics["test_loss"] is not None
                ):
                    self._log_test_evaluation(results.metrics)
                    test_logged = True

                logger.info("Restoring best model checkpoint for model registry...")
                best_model = build_mobilenet_v3(num_classes)
                with best_checkpoint.as_directory() as checkpoint_dir:
                    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
                    checkpoint_data = torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    best_model.load_state_dict(checkpoint_data["model_state"])

                logger.info("Registering model in MLflow Model Registry...")
                sample_shape = (
                    1,
                    3,
                    *self.config.transformation.image_size,
                )
                self.logmodel.register_checkpoint(
                    model=best_model,
                    input_shape=sample_shape,
                )
                logger.info("Pipeline successfully completed.")

        except Exception as e:
            raise RecycleNetException(
                "Failure during the distributed training or assessment cycle", e
            ) from e
