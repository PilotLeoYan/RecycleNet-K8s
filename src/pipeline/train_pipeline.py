"""End-to-end training pipeline orchestrator for RecycleNet."""

from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation
from src.components.evaluator import Evaluator
from src.components.log_model import LogModel
from src.components.loss_functions import get_criterion
from src.components.model import build_mobilenet_v3
from src.components.optimizers import get_optimizer
from src.components.trainer import ModelTrainer
from src.config.schema import AppConfig
from src.pipeline import make_reproducibility
from src.utils import RecycleNetException, get_logger

logger = get_logger(__name__)


class TrainPipeline:
    """Orchestrate end-to-end ingestion, training, evaluation, and tracking.

    Attributes:
        config: High-level pipeline execution configuration.
        seed_config: Seeds configuration for deterministic execution.
        ingestion_config: Dataset extraction settings.
        ingestion: DataIngestion component instance.
        transformation_config: Image transformation settings.
        transformation: DataTransformation component instance.
    """

    def __init__(self, config: AppConfig):
        """Initialize the training pipeline with configuration and subcomponents.

        Args:
            config: Training pipeline configuration settings.
        """
        self.config = config
        make_reproducibility(config.reproducibility)
        self.logmodel = LogModel(config.tracking)
        self.ingestion = DataIngestion(config.ingestion)
        self.transformation = DataTransformation(
            config.transformation, seed=config.reproducibility.torch_seed
        )

    def run(self) -> None:
        """Execute the full end-to-end training and evaluation workflow.

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
            logger.info(
                "Detected [%i] classes: %s",
                len(self.transformation.classes),
                self.transformation.classes,
            )
        except Exception as e:
            raise RecycleNetException(
                "Error creating DataLoaders or splitting data", e
            ) from e

        try:
            logger.info("Building the MobileNetV3 model...")
            model = build_mobilenet_v3(len(self.transformation.classes))
        except Exception as e:
            raise RecycleNetException("Error initialising the model", e) from e

        try:
            criterion = get_criterion()
        except Exception as e:
            raise RecycleNetException("Error initialising the criterion", e) from e

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
                train_loader=train_loader,
                val_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.config.training.device,
                logmodel=self.logmodel,
            )
        except Exception as e:
            raise RecycleNetException("Error initialising the trainer", e) from e

        try:
            evaluator = Evaluator(
                test_loader=test_loader,
                device=self.config.training.device,
                logmodel=self.logmodel,
            )
        except Exception as e:
            raise RecycleNetException("Error initialising the evaluator", e) from e

        logger.info("Starting train pipeline...")

        self.logmodel.set_experiment_tracking()
        pipeline_tags = self.config.tracking.tags.copy()
        pipeline_tags["hardware"] = self.config.training.device

        with self.logmodel.start_run() as active_run:
            logger.info("Active MLflow Run ID: %s", active_run.info.run_id)

            idx_to_class = {
                str(idx): name for idx, name in enumerate(self.transformation.classes)
            }

            try:
                self.logmodel.log_pipeline_metadata(
                    idx_to_class=idx_to_class,
                    tags=pipeline_tags,
                    params={
                        "epochs": self.config.training.epochs,
                        "patience": self.config.training.patience,
                        "batch_size": self.config.transformation.batch_size,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "weight_decay": optimizer.param_groups[0].get(
                            "weight_decay", 0.0
                        ),
                    },
                    config_dict=self.config.model_dump(mode="json"),
                )

                logger.info("Running training...")
                model = trainer.fit(
                    model=model,
                    epochs=self.config.training.epochs,
                    patience=self.config.training.patience,
                )

                logger.info("Running test evaluation...")
                evaluator.evaluate(model)

                self.logmodel.register_checkpoint(
                    model=model,
                    input_shape=(1, *valid_loader.dataset[0][0].shape),
                )

            except Exception as e:
                raise RecycleNetException(
                    "Failure during the training or assessment cycle", e
                ) from e

        logger.info("Pipeline successfully completed")
