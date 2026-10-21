"""End-to-end distributed training pipeline orchestrator for RecycleNet."""

import gc
import os
import tempfile
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
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
from src.utils import RecycleNetException, get_logger, resolve_tracking_uri

logger = get_logger(__name__)


def _log_test_metrics_to_mlflow(
    client: MlflowClient,
    run_id: str,
    eval_metrics: dict[str, Any],
) -> None:
    """Log final test metrics, per-class metrics and confusion matrix.

    Runs on the rank-0 worker, since Ray Train v2 does not expose metrics of
    reports that carry no checkpoint to the driver.

    Args:
        client: MLflow client bound to the tracking URI.
        run_id: Active MLflow run identifier created by the driver.
        eval_metrics: Output of ``compute_distributed_test_metrics``.
    """
    metrics: dict[str, float] = {
        "test_roc": float(eval_metrics.get("auc", 0.0)),
        "test_accuracy": float(eval_metrics.get("test_accuracy", 0.0)),
        "test_precision": float(eval_metrics.get("test_precision", 0.0)),
        "test_recall": float(eval_metrics.get("test_recall", 0.0)),
        "test_f1_score": float(eval_metrics.get("test_f1_score", 0.0)),
    }
    if eval_metrics.get("test_loss") is not None:
        metrics["test_loss"] = float(eval_metrics["test_loss"])

    per_class = eval_metrics.get("per_class_metrics")
    if per_class:
        client.log_dict(run_id, per_class, "per_class_metrics.json")
        for cls_name, vals in per_class.items():
            metrics[f"test_f1_{cls_name}"] = float(vals["f1_score"])
            metrics[f"test_recall_{cls_name}"] = float(vals["recall"])

    for key, value in metrics.items():
        client.log_metric(run_id=run_id, key=key, value=value)

    fig_cm = eval_metrics.get("confusion_matrix")
    if isinstance(fig_cm, Figure):
        client.log_figure(run_id, fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)


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
                    logger.warning(
                        "Real-time MLflow worker metric logging failed: %s", exc
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

        if (
            world_rank == 0
            and config.get("mlflow_run_id")
            and config.get("tracking_uri")
        ):
            try:
                _log_test_metrics_to_mlflow(
                    MlflowClient(tracking_uri=config["tracking_uri"]),
                    config["mlflow_run_id"],
                    eval_metrics,
                )
            except Exception as exc:
                logger.warning("Worker MLflow test metric logging failed: %s", exc)

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
                "tracking_uri": resolve_tracking_uri(self.config.tracking.tracking_uri),
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

                # Epoch and test metrics are logged by the rank-0 worker, because
                # Ray Train v2 only keeps metrics of checkpointed reports.

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
