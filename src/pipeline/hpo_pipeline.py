"""Hyperparameter Optimization (HPO) pipeline orchestrator using Ray Tune and Optuna."""

from pathlib import Path
from typing import Any

import torch
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import FailureConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation
from src.components.loss_functions import get_criterion
from src.components.model import build_mobilenet_v3
from src.components.optimizers import get_optimizer
from src.components.train_step import TrainStep
from src.config import AppConfig, TransformationConfig
from src.pipeline.ray_resources import assign_ray_resources
from src.pipeline.reproducibility import make_reproducibility
from src.utils import RecycleNetException, get_logger

logger = get_logger(__name__)


def train_eval_trial(
    config: dict[str, Any],
    data_dir: Path,
    app_config: AppConfig,
) -> None:
    """Execute a single Ray Tune trial training and validation loop.

    Args:
        config: Hyperparameter and resource configuration dictionary for the trial.
        data_dir: Path to the extracted dataset directory.
        app_config: Full application configuration containing static defaults.
    """
    try:
        make_reproducibility(app_config.reproducibility)

        trans_config = TransformationConfig(
            image_size=app_config.transformation.image_size,
            image_mean=app_config.transformation.image_mean,
            image_std=app_config.transformation.image_std,
            random_h_flip=app_config.transformation.random_h_flip,
            random_rotation=app_config.transformation.random_rotation,
            train_split=app_config.transformation.train_split,
            eval_split=app_config.transformation.eval_split,
            test_split=app_config.transformation.test_split,
            batch_size=config["batch_size"],
            num_workers=config["dataloader_n_workers"],
            pin_memory=app_config.transformation.pin_memory,
            seed=app_config.reproducibility.torch_seed,
        )
        transformation = DataTransformation(
            trans_config,
            seed=app_config.reproducibility.torch_seed,
        )
        train_loader, valid_loader, _ = transformation.get_dataloaders(data_dir)

        model = build_mobilenet_v3(len(transformation.classes))
        model = model.to(config["device"])
        criterion = get_criterion()
        optimizer = get_optimizer(
            filter(lambda p: p.requires_grad, model.parameters()),
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        trainer = TrainStep(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config["device"],
        )

        for _ in range(config["max_epochs"]):
            metrics = trainer.train_valid_step()
            tune.report(metrics)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()


class HPOPipeline:
    """Manage distributed Hyperparameter Optimization experiments using Ray Tune.

    Attributes:
        config: Application configuration with HPO, tracking, and data parameters.
        ingestion: Data ingestion helper for extracting raw datasets.
    """

    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        """Initialize HPO pipeline with configuration and reproducibility seeds.

        Args:
            config: Root application configuration object.
        """
        self.config = config
        make_reproducibility(config.reproducibility)
        self.ingestion = DataIngestion(config.ingestion)

    def run(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Execute the Ray Tune HPO search across configured worker resources.

        Returns:
            tuple[dict[str, Any] | None, dict[str, Any] | None]: Best trial
                hyperparameter configuration dictionary and evaluation metrics.

        Raises:
            RecycleNetException: If Ray Tune encounters an unrecoverable failure.
        """
        logger.info("Initializing Ray Tune HPO pipeline...")

        try:
            device, resources = assign_ray_resources(
                self.config.hpo.device,
                self.config.hpo.cpu_resources_per_trial,
                self.config.hpo.gpu_resources_per_trial,
            )

            logger.info(
                "HPO hardware resolved: effective_device=%s, "
                "cpu_per_trial=%s, gpu_per_trial=%s",
                device,
                resources.get("CPU", 0.0),
                resources.get("GPU", 0.0),
            )

            param_space = {
                "learning_rate": tune.loguniform(
                    self.config.hpo.learning_rate_range[0],
                    self.config.hpo.learning_rate_range[1],
                ),
                "weight_decay": tune.loguniform(
                    self.config.hpo.weight_decay_range[0],
                    self.config.hpo.weight_decay_range[1],
                ),
                "batch_size": tune.choice(self.config.hpo.batch_size),
                "max_epochs": self.config.hpo.max_epochs,
                "grace_period": self.config.hpo.grace_period,
                "dataloader_n_workers": self.config.hpo.dataloader_n_workers,
                "device": device,
            }

            dataset_dir = self.ingestion.extract_dataset().resolve()
            logger.info("Dataset extracted at: %s", dataset_dir)

            tuner_parameters = tune.with_parameters(
                train_eval_trial,
                data_dir=dataset_dir,
                app_config=self.config,
            )

            tuner_resources = tune.with_resources(
                tuner_parameters,
                resources=resources,
            )

            scheduler = ASHAScheduler(
                max_t=self.config.hpo.max_epochs,
                grace_period=self.config.hpo.grace_period,
                reduction_factor=self.config.hpo.reduction_factor,
                metric="valid_loss",
                mode="min",
            )

            search_alg = OptunaSearch(
                metric="valid_loss",
                mode="min",
            )

            tune_config = tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=self.config.hpo.num_samples,
                max_concurrent_trials=self.config.hpo.max_concurrent_trials,
            )

            run_config = tune.RunConfig(
                name=self.config.hpo.experiment_name,
                failure_config=FailureConfig(
                    max_failures=0,
                    fail_fast=True,
                ),
                callbacks=[
                    MLflowLoggerCallback(
                        tracking_uri=self.config.tracking.tracking_uri,
                        experiment_name=self.config.hpo.experiment_name,
                        save_artifact=True,
                    )
                ],
            )

            tuner = tune.Tuner(
                trainable=tuner_resources,
                param_space=param_space,
                tune_config=tune_config,
                run_config=run_config,
            )

            logger.info(
                "Starting Ray Tune execution (samples=%d, max_concurrent=%d)...",
                self.config.hpo.num_samples,
                self.config.hpo.max_concurrent_trials,
            )

            results = tuner.fit()

            if results.errors:
                logger.error(
                    "Ray Tune HPO aborted: detected %d trial(s) with errors.",
                    results.num_errors,
                )

                for error in results.errors:
                    root_cause = getattr(error, "cause", error)
                    err_type = type(root_cause).__name__
                    err_msg = str(root_cause)

                    tb_lines = str(error).strip().splitlines()
                    tail_tb = "\n    ".join(tb_lines[-8:])

                    logger.error(
                        "Trial failed with %s: %s\n"
                        "  Traceback snippet (last lines):\n    %s",
                        err_type,
                        err_msg,
                        tail_tb,
                    )

                raise RecycleNetException(
                    f"Ray Tune HPO aborted due to failures in "
                    f"{len(results.errors)} trial(s)."
                )

            best_result = results.get_best_result(metric="valid_loss", mode="min")
            logger.info("Ray Tune HPO search completed successfully.")
            logger.info("Best trial config: %s", best_result.config)
            logger.info("Best trial metrics: %s", best_result.metrics)

            return best_result.config, best_result.metrics

        except Exception as e:
            logger.exception("Critical error during HPO pipeline execution: %s", str(e))
            raise RecycleNetException(
                "Error encountered during distributed Ray Tune HPO execution", e
            ) from e
