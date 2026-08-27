from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseModel):
    """Configuration paths for dataset ingestion.

    Attributes:
        zip_source: Path to the compressed raw dataset zip archive.
        raw_output_dir: Destination directory where files will be extracted.
    """

    zip_source: Path
    raw_output_dir: Path = Path("data/raw")  # default value


class TransformationConfig(BaseModel):
    """Configuration parameters for dataset transformations and DataLoader setup.

    Attributes:
        image_size: Target (height, width) dimensions for model input tensors.
        image_mean: Channel-wise RGB normalization means (ImageNet defaults).
        image_std: Channel-wise RGB normalization standard deviations.
        random_h_flip: Probability of horizontal flip augmentation.
        random_rotation: Rotation angle range (min, max) or single max angle in degrees.
        train_split: Proportion of data allocated for training.
        eval_split: Proportion of data allocated for validation.
        test_split: Proportion of data allocated for final testing.
        batch_size: Number of image samples per mini-batch.
        num_workers: Number of subprocesses for multi-threaded data loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory before returning.
    """

    # MobileNetV3 Standard Input Format
    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Data Augmentation
    random_h_flip: float = Field(default=0.5, ge=0.0, le=1.0)
    random_rotation: tuple[int, int] | int = (-90, 90)
    # Data split
    train_split: float = 0.7
    eval_split: float = 0.15
    test_split: float = 0.15
    # Data Loader
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = 4
    pin_memory: bool = True

    @model_validator(mode="after")
    def validate_splits(self) -> TransformationConfig:
        """Ensures train, validation, and test splits sum up to 1.0."""
        total = self.train_split + self.eval_split + self.test_split
        if not abs(total - 1.0) < 1e-5:
            raise ValueError(f"Splits must sum to 1.0, got: {total}")
        return self


class ReproducibilityConfig(BaseModel):
    """Random seed configuration settings for environment reproducibility.

    Attributes:
        random_seed: Seed for standard Python random and hashseed generation.
        numpy_seed: Seed for NumPy pseudorandom number generators.
        torch_seed: Seed for PyTorch CPU and CUDA random number generators.
    """

    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42


class TrainingConfig(BaseModel):
    """Configuration options for the training pipeline run.

    Attributes:
        num_classes: Number of distinct classification categories.
        epochs: Number of complete passes over the training dataset.
        patience: Epoch patience threshold for early stopping.
        learning_rate: Learning Rate (LR), Alpha, or Learning Step.
        weight_decay: Weight Decay, Lambda, or Penalty.
        device: Device identifier string ('cuda' or 'cpu').
    """

    num_classes: int = 6
    epochs: int = Field(default=10, ge=1)
    patience: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, gt=0)
    device: str = "cuda"


class TrackingConfig(BaseModel):
    """Configuration for MLflow tracking"""

    experiment_name: str = "RecycleNet_Training"
    registered_model_name: str = "RecycleNet"
    tracking_uri: str = "sqlite:///mlflow.db"


class AppConfig(BaseSettings):
    """Root application configuration"""

    ingestion: IngestionConfig
    transformation: TransformationConfig
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig
    )
    training: TrainingConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    model_config = SettingsConfigDict(
        env_prefix="RECYCLENET_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> AppConfig:
        """Loads configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        return cls(**data)
