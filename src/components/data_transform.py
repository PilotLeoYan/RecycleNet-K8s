"""Data transformation component for image preprocessing and DataLoaders."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset, random_split


@dataclass
class DataTransformationConfig:
    """Configuration parameters for dataset transformations and DataLoader setup.

    Attributes:
        torch_manual_seed: Seed for reproducible dataset splitting.
        tensor_float: Precision dtype for tensor outputs.
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

    torch_manual_seed: int = 42
    tensor_float: torch.dtype = torch.float32
    # MobileNetV3 Standard Input Format
    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Data Augmentation
    random_h_flip: float = 0.5
    random_rotation: tuple[int, int] | int = (-90, 90)
    # Data split
    train_split: float = 0.7
    eval_split: float = 0.15
    test_split: float = 0.15
    # Data Loader
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


class TransformSubset(Dataset):
    """PyTorch Dataset wrapper applying specific transforms to a Dataset subset.

    Attributes:
        subset: The underlying dataset partition (train, val, or test).
        transform: Torchvision v2 transform pipeline to apply per sample.
    """

    def __init__(self, subset: Subset, transform: v2.Compose) -> None:
        """Initializes TransformSubset with a data partition and transform pipeline.

        Args:
            subset: Data subset partition.
            transform: Transformation composition to apply on fetched samples.
        """
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the subset.

        Returns:
            int: Sample count.
        """
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retrieves and transforms the image tensor and label at the specified index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int]: Transformed image tensor and integer class index.
        """
        image, label = self.subset[idx]
        return self.transform(image), label


class DataTransformation:
    """Manages torchvision preprocessing pipelines, dataset splits, and DataLoaders.

    Attributes:
        config: Transformation configuration parameters.
        classes: List of class directory names detected in the dataset.
        class_to_idx: Mapping from class names to integer target indices.
    """

    def __init__(self, config: DataTransformationConfig):
        """Initializes DataTransformation with configuration.

        Args:
            config: Data transformation configuration.
        """
        self.config = config
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

    def _get_transforms(self) -> tuple[v2.Compose, v2.Compose]:
        """Constructs torchvision v2 transform pipelines for training and evaluation.

        Returns:
            tuple[v2.Compose, v2.Compose]: Training transform (with data augmentation)
                and evaluation transform (base resizing and normalization only).
        """
        base_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=self.config.image_size),
            ]
        )

        norm_transform = v2.Compose(
            [
                v2.Normalize(mean=self.config.image_mean, std=self.config.image_std),
            ]
        )

        # data augmentation + normalization
        train_compose = v2.Compose(
            list(base_transform.transforms)
            + [
                v2.RandomHorizontalFlip(p=self.config.random_h_flip),
                v2.RandomRotation(degrees=self.config.random_rotation),
            ]
            + list(norm_transform.transforms)
        )

        # normalization
        val_transform = v2.Compose(
            list(base_transform.transforms) + list(norm_transform.transforms)
        )

        return train_compose, val_transform

    def _get_dataset(self, raw_data_dir: Path) -> datasets.ImageFolder:
        """Loads raw images from structured class subfolders via ImageFolder.

        Args:
            raw_data_dir: Path to directory containing class subfolders of images.

        Returns:
            datasets.ImageFolder: Loaded PyTorch dataset.
        """
        return datasets.ImageFolder(root=raw_data_dir)

    def _get_subsets(self, dataset: Dataset) -> list[Subset]:
        """Splits the complete dataset into train, validation, and test subsets.

        Args:
            dataset: The full PyTorch dataset.

        Returns:
            list[Subset]: Subsets for train, validation, and test splits.
        """
        torch.manual_seed(self.config.torch_manual_seed)
        return random_split(
            dataset,
            (
                self.config.train_split,
                self.config.eval_split,
                self.config.test_split,
            ),
        )

    def _get_dataloader(self, dataset: Dataset, is_train: bool) -> DataLoader:
        """Wraps a dataset into a configured PyTorch DataLoader.

        Args:
            dataset: Dataset or TransformSubset to wrap.
            is_train: Whether this DataLoader is for training (enables shuffling
                and drop_last).

        Returns:
            DataLoader: Configured PyTorch DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=is_train,
        )

    def get_dataloaders(
        self, raw_data_dir: Path
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Builds and returns DataLoaders for train, validation, and test partitions.

        Args:
            raw_data_dir: Path to extracted dataset root directory.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test
                DataLoaders.
        """
        full_dataset = self._get_dataset(raw_data_dir)

        self.classes = full_dataset.classes
        self.class_to_idx = full_dataset.class_to_idx

        train_transform, val_transform = self._get_transforms()

        train_sub, val_sub, test_sub = self._get_subsets(full_dataset)

        train_ds = TransformSubset(train_sub, train_transform)
        val_ds = TransformSubset(val_sub, val_transform)
        test_ds = TransformSubset(test_sub, val_transform)

        train_loader = self._get_dataloader(train_ds, True)
        val_loader = self._get_dataloader(val_ds, False)
        test_loader = self._get_dataloader(test_ds, False)
        return train_loader, val_loader, test_loader
