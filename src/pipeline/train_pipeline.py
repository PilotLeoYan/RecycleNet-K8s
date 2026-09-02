"""End-to-end training pipeline orchestrator for RecycleNet."""

import tempfile
from datetime import datetime

import mlflow

from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation
from src.components.evaluator import Evaluator
from src.components.loss_functions import get_criterion
from src.components.model import build_mobilenet_v3
from src.components.optimizers import get_optimizer
from src.components.trainer import ModelTrainer
from src.config.schema import AppConfig
from src.exception import RecycleNetException
from src.pipeline.reproducibility import make_reproducibility
from src.utils import get_logger

logger = get_logger(__name__)


class TrainPipeline:
    """Orchestrates end-to-end ingestion, training, evaluation, and tracking.

    Attributes:
        config: High-level pipeline execution configuration.
        seed_config: Seeds configuration for deterministic execution.
        ingestion_config: Dataset extraction settings.
        ingestion: DataIngestion component instance.
        transformation_config: Image transformation settings.
        transformation: DataTransformation component instance.
    """

    def __init__(self, config: AppConfig):
        """Initializes the training pipeline with configuration and subcomponents.

        Args:
            config: Training pipeline configuration settings.
        """
        self.config = config
        self.ingestion = DataIngestion(config.ingestion)
        self.transformation = DataTransformation(config.transformation)

        make_reproducibility(config.reproducibility)

    def run(self) -> None:
        """Executes the full end-to-end training and evaluation workflow.

        Steps:
            1. Ingests and unpacks the raw dataset archive.
            2. Builds torchvision transformations and train/val/test DataLoaders.
            3. Instantiates pre-trained MobileNetV3 with custom classification head.
            4. Configures loss function and optimizer.
            5. Initializes MLflow tracking run and logs parameters and tags.
            6. Executes training loop with early stopping.
            7. Evaluates best checkpoint on test set and logs metrics/confusion matrix.

        Raises:
            RecycleNetException: If any pipeline stage encounters a fatal error.
        """
        logger.info("Starting training pipeline...")

        try:
            logger.info("Extracting dataset...")
            raw_path = self.ingestion.extract_dataset()
        except Exception as e:
            raise RecycleNetException(
                "Error during the data ingestion stage (zip extraction)", e
            ) from e

        try:
            logger.info("Creating dataloaders...")
            train_loader, valid_loader, test_loader = (
                self.transformation.get_dataloaders(raw_path)
            )
        except Exception as e:
            raise RecycleNetException(
                "Error creating DataLoaders or splitting data", e
            ) from e

        try:
            logger.info("Building the MobileNetV3 model...")
            model = build_mobilenet_v3(self.config.training.num_classes)
        except Exception as e:
            raise RecycleNetException("Error initialising the model", e) from e

        try:
            criterion = get_criterion()
        except Exception as e:
            raise RecycleNetException("Error initialising the criterio", e) from e

        try:
            optimizer = get_optimizer(
                filter(lambda p: p.requires_grad, model.parameters()),
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        except Exception as e:
            raise RecycleNetException("Error initialising the optimizer", e) from e

        try:
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.config.training.device,
            )
        except Exception as e:
            raise RecycleNetException("Error initialising the trainer", e) from e

        logger.info("Training...")

        mlflow.set_tracking_uri(self.config.tracking.tracking_uri)
        mlflow.set_experiment(self.config.tracking.experiment_name)

        run_name = f"mobilenetv3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as active_run:
            logger.info("Active MLflow Run ID: %s", active_run.info.run_id)

            idx_to_class = {
                str(idx): name for idx, name in enumerate(self.transformation.classes)
            }
            mlflow.log_dict(idx_to_class, "classes_mapping.json")

            try:
                mlflow.set_tags(
                    {
                        "framework": "pytorch",
                        "model_architecture": "mobilenet_v3_small",
                        "dataset": "trashnet",
                        "task": "image_classification",
                        "hardware": self.config.training.device,
                    }
                )

                mlflow.log_params(
                    {
                        # Pipeline & Hardware
                        "device": self.config.training.device,
                        "epochs": self.config.training.epochs,
                        "patience": self.config.training.patience,
                        "batch_size": self.config.transformation.batch_size,
                        # Seeds
                        "seed_torch": self.config.reproducibility.torch_seed,
                        # Model
                        "model_architecture": "mobilenet_v3_small",
                        "num_classes": self.config.training.num_classes,
                        "freeze_base": True,
                        # Optimizer & Loss
                        "optimizer": optimizer.__class__.__name__,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "weight_decay": optimizer.param_groups[0].get(
                            "weight_decay", 0.0
                        ),
                        "criterion": criterion.__class__.__name__,
                        # Preprocessing & Augmentation
                        "image_size": f"{self.config.transformation.image_size[0]}x"
                        f"{self.config.transformation.image_size[1]}",
                        "random_h_flip_prob": self.config.transformation.random_h_flip,
                        "random_rotation_deg": str(
                            self.config.transformation.random_rotation
                        ),
                        "split_ratios": f"{self.config.transformation.train_split}/"
                        f"{self.config.transformation.eval_split}/"
                        f"{self.config.transformation.test_split}",
                    }
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    best_path = trainer.fit(
                        epochs=self.config.training.epochs,
                        weights_path=temp_dir,
                        patience=self.config.training.patience,
                    )

                    logger.info("Running test evaluation...")
                    evaluator = Evaluator(
                        model=model,
                        weights_path=best_path,
                        test_loader=test_loader,
                        device=self.config.training.device,
                    )
                    evaluator.evaluate()

            except Exception as e:
                raise RecycleNetException(
                    "Failure during the training or assessment cycle", e
                ) from e
