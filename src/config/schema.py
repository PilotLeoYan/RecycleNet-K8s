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

    def __str__(self) -> str:
        return f"""IngestionConfig:
  zip_source: {self.zip_source}
  raw_output_dir: {self.raw_output_dir}"""


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
    train_split: float = Field(default=0.7, gt=0.0, lt=1.0)
    eval_split: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.15, gt=0.0, lt=1.0)
    # Data Loader
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True

    @model_validator(mode="after")
    def validate_splits(self) -> TransformationConfig:
        """Ensures train, validation, and test splits sum up to 1.0."""
        total = self.train_split + self.eval_split + self.test_split
        if not abs(total - 1.0) < 1e-5:
            raise ValueError(f"Splits must sum to 1.0, got: {total}")
        return self

    def __str__(self) -> str:
        return f"""TransformationConfig:
  image_size: {self.image_size}
  image_mean: {self.image_mean}
  image_std: {self.image_std}
  random_h_flip: {self.random_h_flip}
  random_rotation: {self.random_rotation}
  train_split: {self.train_split}
  eval_split: {self.eval_split}
  test_split: {self.test_split}
  batch_size: {self.batch_size}
  num_workers: {self.num_workers}
  pin_memory: {self.pin_memory}"""


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

    def __str__(self) -> str:
        return f"""ReproducibilityConfig:
  random_seed: {self.random_seed}
  numpy_seed: {self.numpy_seed}
  torch_seed: {self.torch_seed}"""


class TrainingConfig(BaseModel):
    """Configuration options for the training pipeline run.

    Attributes:
        epochs: Number of complete passes over the training dataset.
        patience: Epoch patience threshold for early stopping.
        learning_rate: Learning Rate (LR), Alpha, or Learning Step.
        weight_decay: Weight Decay, Lambda, or Penalty.
        device: Device identifier string ('cuda' or 'cpu').
    """

    epochs: int = Field(default=10, ge=1)
    patience: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    device: str = "cuda"

    def __str__(self) -> str:
        return f"""TrainingConfig:
  epochs: {self.epochs}
  patience: {self.patience}
  learning_rate: {self.learning_rate}
  weight_decay: {self.weight_decay}
  device: {self.device}"""


class TrackingConfig(BaseModel):
    """Configuration for MLflow tracking"""

    experiment_name: str = "RecycleNet_Training"
    registered_model_name: str = "RecycleNet"
    tracking_uri: str = "sqlite:///mlflow.db"

    def __str__(self) -> str:
        return f"""TrackingConfig
  experiment_name: {self.experiment_name}
  registered_model_name: {self.registered_model_name}
  tracking_uri: {self.tracking_uri}"""


class HPOConfig(BaseModel):
    """Configuration for Hyperparameters Optimization"""

    # loguniform(1e-4, 1e-1)
    weight_decay_range: tuple[float, float] = Field(default=(1e-4, 1e-1))
    # loguniform(1e-4, 1e-1)
    learning_rate_range: tuple[float, float] = Field(default=(1e-4, 1e-1))
    batch_size: list[int] = Field(default=[8, 16, 32])
    num_samples: int = Field(default=2, gt=0)
    max_epochs: int = Field(default=4, gt=0)
    grace_period: int = Field(default=2, gt=0)
    reduction_factor: int = Field(default=2, ge=2)
    dataloader_n_workers: int = Field(default=4, ge=1)
    max_concurrent_trials: int = Field(default=2, ge=1)
    cpu_resources_per_trial: float = Field(default=2.0, ge=1.0)
    gpu_resources_per_trial: float = Field(default=0.5, ge=0.0)
    device: str = Field(default="cpu")
    experiment_name: str = Field(default="hpo_mobilenetv3_experiment")


class AppConfig(BaseSettings):
    """Root application configuration"""

    ingestion: IngestionConfig
    transformation: TransformationConfig
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig
    )
    training: TrainingConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)

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
            data: Any = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration YAML at {path} must define a mapping, "
                "got {type(data).__name__}"
            )

        return cls(**data)

    def __str__(self) -> str:
        return f"""AppConfig:
{self.ingestion.__str__()}
{self.transformation.__str__()}
{self.reproducibility.__str__()}
{self.training.__str__()}
{self.tracking.__str__()}"""
