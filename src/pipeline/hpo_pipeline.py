from pathlib import Path
from typing import Any

from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from src.components import DataIngestion, DataTransformation, ModelTrainer
from src.components.loss_functions import get_criterion
from src.components.model import build_mobilenet_v3
from src.components.optimizers import get_optimizer
from src.config import AppConfig, TransformationConfig


def train_eval_trial(
    config: dict[str, Any],
    data_dir: Path,
    app_config: AppConfig,
) -> None:
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
    )
    transformation = DataTransformation(trans_config)
    train_loader, valid_loader, _ = transformation.get_dataloaders(data_dir)

    model = build_mobilenet_v3(len(transformation.classes))
    criterion = get_criterion()
    optimizer = get_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config["device"],
        tracking_config=None,
    )

    for epoch in range(config["max_epochs"]):
        loss = trainer._train_step()
        vloss, metrics = trainer._valid_step()
        tune.report(
            {
                "train_loss": loss,
                "val_loss": vloss,
                "val_accuracy": metrics["accuracy"],
            }
        )


class HPOPipeline:
    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        self.config = config

    def run(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
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
            # fixed hyperparameters
            "max_epochs": self.config.hpo.max_epochs,
            "grace_period": self.config.hpo.grace_period,
            "dataloader_n_workers": self.config.hpo.dataloader_n_workers,
            "device": self.config.hpo.device,
        }

        ingestion = DataIngestion(self.config.ingestion)
        # Plasma Store
        tuner_parameters = tune.with_parameters(
            train_eval_trial,
            data_dir=ingestion.extract_dataset().resolve(),
            app_config=self.config,
        )

        tuner_resourcers = tune.with_resources(
            tuner_parameters,
            resources={
                "cpu": self.config.hpo.cpu_resources_per_trial,
                "gpu": self.config.hpo.gpu_resources_per_trial,
            },
        )

        scheduler = ASHAScheduler(
            max_t=self.config.hpo.max_epochs,
            grace_period=self.config.hpo.grace_period,
            reduction_factor=self.config.hpo.reduction_factor,
            metric="val_loss",
            mode="min",
        )

        search_alg = OptunaSearch(
            metric="val_loss",
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
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=self.config.tracking.tracking_uri,
                    experiment_name=self.config.tracking.experiment_name,
                    save_artifact=True,
                )
            ],
        )

        tuner = tune.Tuner(
            trainable=tuner_resourcers,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )

        results = tuner.fit()
        best_result = results.get_best_result(metric="val_loss", mode="min")

        return best_result.config, best_result.metrics


if __name__ == "__main__":
    from pathlib import Path

    config_path = Path("configs/config.yaml")
    app_config = AppConfig.from_yaml(config_path)

    pipeline = HPOPipeline(app_config)
    best_config, best_metrics = pipeline.run()

    print(f"Best config: {best_config}")
    print(f"Best metrics: {best_metrics}")
