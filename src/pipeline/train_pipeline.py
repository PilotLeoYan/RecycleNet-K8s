"""End-to-end training pipeline orchestrator for RecycleNet."""

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import torch

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transform import DataTransformation, DataTransformationConfig
from src.components.evaluator import Evaluator
from src.components.loss_functions import get_criterion
from src.components.model import build_mobilenet_v3
from src.components.optimizers import get_optimizer
from src.components.trainer import ModelTrainer
from src.exception import RecycleNetException
from src.pipeline.reproducibility import ReproducibilityConfig, make_reproducibility
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingPipelineConfig:
    """Configuration options for the training pipeline run.

    Attributes:
        num_classes: Number of distinct classification categories.
        epochs: Number of complete passes over the training dataset.
        patience: Epoch patience threshold for early stopping.
        device: Device identifier string ('cuda' or 'cpu').
    """

    num_classes: int = 6
    epochs: int = 1
    patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

    def __init__(self, config: TrainingPipelineConfig):
        """Initializes the training pipeline with configuration and subcomponents.

        Args:
            config: Training pipeline configuration settings.
        """
        self.config = config
        self.seed_config = ReproducibilityConfig()

        self.ingestion_config = DataIngestionConfig()
        self.ingestion = DataIngestion(self.ingestion_config)

        self.transformation_config = DataTransformationConfig()
        self.transformation = DataTransformation(self.transformation_config)

        make_reproducibility(self.seed_config)

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
            model = build_mobilenet_v3(self.config.num_classes)
        except Exception as e:
            raise RecycleNetException("Error initialising the model", e) from e

        try:
            criterion = get_criterion()
        except Exception as e:
            raise RecycleNetException("Error initialising the criterio", e) from e

        try:
            optimizer = get_optimizer(model.parameters())
        except Exception as e:
            raise RecycleNetException("Error initialising the optimizer", e) from e

        try:
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.config.device,
            )
        except Exception as e:
            raise RecycleNetException("Error initialising the trainer", e) from e

        logger.info("Training...")

        db_path = Path(__file__).resolve().parents[2] / "mlflow.db"
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", f"sqlite:////{db_path}")
        )
        mlflow.set_experiment("RecycleNet_Training")

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
                        "hardware": "cuda" if torch.cuda.is_available() else "cpu",
                    }
                )

                mlflow.log_params(
                    {
                        # Pipeline & Hardware
                        "device": self.config.device,
                        "epochs": self.config.epochs,
                        "patience": self.config.patience,
                        "batch_size": self.transformation_config.batch_size,
                        # Seeds
                        "seed_torch": self.seed_config.torch_seed,
                        # Model
                        "model_architecture": "mobilenet_v3_small",
                        "num_classes": self.config.num_classes,
                        "freeze_base": True,
                        # Optimizer & Loss
                        "optimizer": optimizer.__class__.__name__,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "weight_decay": optimizer.param_groups[0].get(
                            "weight_decay", 0.0
                        ),
                        "criterion": criterion.__class__.__name__,
                        # Preprocessing & Augmentation
                        "image_size": f"{self.transformation_config.image_size[0]}x"
                        f"{self.transformation_config.image_size[1]}",
                        "random_h_flip_prob": self.transformation_config.random_h_flip,
                        "random_rotation_deg": str(
                            self.transformation_config.random_rotation
                        ),
                        "split_ratios": f"{self.transformation_config.train_split}/"
                        f"{self.transformation_config.eval_split}/"
                        f"{self.transformation_config.test_split}",
                    }
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    best_path = trainer.fit(
                        epochs=self.config.epochs,
                        weights_path=temp_dir,
                        patience=self.config.patience,
                    )

                    logger.info("Running test evaluation...")
                    evaluator = Evaluator(
                        model=model,
                        weights_path=best_path,
                        test_loader=test_loader,
                        device=self.config.device,
                    )
                    evaluator.evaluate()

            except Exception as e:
                raise RecycleNetException(
                    "Failure during the training or assessment cycle", e
                ) from e
