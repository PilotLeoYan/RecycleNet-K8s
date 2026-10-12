import random
from collections.abc import Sized
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

from src.config import TransformationConfig

SampleType = tuple[torch.Tensor, int]


def seed_worker(_worker_id: int) -> None:
    """Set random seeds for NumPy and Python random in DataLoader worker processes.

    Ensures deterministic augmentations and operations across multi-process data
    loading workers by reseeding external libraries using PyTorch's generated
    worker seed.

    Args:
        _worker_id: The integer ID of the worker subprocess.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RecycleDataset(Dataset[SampleType]):
    """Wrap a PyTorch dataset partition to apply torchvision v2 transforms per sample.

    Attributes:
        data_source: Underlying dataset partition or subset containing raw samples.
        transform: Optional torchvision v2 transformation pipeline to apply to images.
    """

    def __init__(
        self,
        data_source: Dataset[SampleType],
        transform: v2.Compose | None = None,
    ) -> None:
        """Initialize RecycleDataset with a data source and optional transform.

        Args:
            data_source: Underlying dataset or subset partition.
            transform: Optional torchvision v2 transformation pipeline.
        """
        self.data_source = data_source
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples in the dataset partition.

        Returns:
            int: Number of samples in the underlying data source.

        Raises:
            TypeError: If the underlying data source does not implement Sized.
        """
        if isinstance(self.data_source, Sized):
            return len(self.data_source)
        raise TypeError(
            f"Underlying data_source of type {type(self.data_source)} is not Sized."
        )

    def __getitem__(self, idx: int) -> SampleType:
        """Retrieve and transform the image tensor and class label at the index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            SampleType: Tuple of (transformed_image_tensor, integer_class_index).
        """
        image, label = self.data_source[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DataTransformation:
    """Manage torchvision preprocessing pipelines, dataset splits, and DataLoaders.

    Attributes:
        config: Transformation and DataLoader configuration parameters.
        seed: Optional random seed for reproducible DataLoader sampling and splitting.
        classes: List of class directory names detected in the dataset.
        class_to_idx: Mapping from class names to integer target indices.
    """

    def __init__(
        self,
        config: TransformationConfig,
        seed: int | None = None,
    ) -> None:
        """Initialize DataTransformation with configuration and optional seed.

        Args:
            config: Data transformation and loading configuration.
            seed: Optional random seed for DataLoaders and dataset splitting.
        """
        self.config = config
        self.seed = seed if seed is not None else getattr(config, "seed", None)
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

    def _get_generator(self, seed: int | None = None) -> torch.Generator:
        """Create a seeded PyTorch Generator for reproducible operations.

        Resolves the effective seed using a cascade: explicit seed argument,
        then self.seed, falling back to torch.initial_seed().

        Args:
            seed: Optional explicit random seed.

        Returns:
            torch.Generator: Initialized PyTorch generator instance.
        """
        generator = torch.Generator()
        effective_seed = seed if seed is not None else self.seed

        if effective_seed is not None:
            generator.manual_seed(effective_seed)
        else:
            generator.manual_seed(torch.initial_seed())
        return generator

    def _get_transformations(self) -> tuple[v2.Compose, v2.Compose]:
        """Construct torchvision v2 transform pipelines for training and evaluation.

        Returns:
            tuple[v2.Compose, v2.Compose]: Training transform (with data augmentation)
                and evaluation transform (base resizing and normalization only).
        """
        base_transformation = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=self.config.image_size),
            ]
        )
        norm_transformation = v2.Compose(
            [
                v2.Normalize(
                    mean=self.config.image_mean,
                    std=self.config.image_std,
                )
            ]
        )
        # Data Augmentation + Normalization
        train_compose = v2.Compose(
            list(base_transformation.transforms)
            + [
                v2.RandomHorizontalFlip(p=self.config.random_h_flip),
                v2.RandomRotation(degrees=self.config.random_rotation),
            ]
            + list(norm_transformation.transforms)
        )

        # Normalization
        valid_compose = v2.Compose(
            list(base_transformation.transforms) + list(norm_transformation.transforms)
        )

        return train_compose, valid_compose

    def _build_dataloader(
        self,
        dataset: Dataset[SampleType],
        is_train: bool,
    ) -> DataLoader[SampleType]:
        """Wrap a dataset partition into a configured PyTorch DataLoader.

        Args:
            dataset: Dataset or RecycleDataset instance to load.
            is_train: Whether this DataLoader is for training (enables shuffling
                and drop_last).

        Returns:
            DataLoader[SampleType]: Deterministic PyTorch DataLoader instance.
        """
        generator = self._get_generator()
        return DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory if torch.cuda.is_available() else False,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def get_dataloaders(
        self,
        raw_data_dir: Path,
    ) -> tuple[DataLoader[SampleType], DataLoader[SampleType], DataLoader[SampleType]]:
        """Build and return DataLoaders for train, validation, and test partitions.

        Discovers dataset classes, generates reproducible splits, binds transformations,
        and constructs corresponding DataLoaders.

        Args:
            raw_data_dir: Path to directory containing class subfolders of images.

        Returns:
            tuple[
                DataLoader[SampleType],
                DataLoader[SampleType],
                DataLoader[SampleType]
            ]:
                Train, validation, and test DataLoaders.
        """

        full_dataset = datasets.ImageFolder(root=raw_data_dir)
        self.classes = full_dataset.classes
        self.class_to_idx = full_dataset.class_to_idx

        train_transform, valid_transform = self._get_transformations()

        split_generator = self._get_generator()
        train_sub, valid_sub, test_sub = random_split(
            full_dataset,
            (
                self.config.train_split,
                self.config.eval_split,
                self.config.test_split,
            ),
            generator=split_generator,
        )

        train_ds = RecycleDataset(train_sub, train_transform)
        valid_ds = RecycleDataset(valid_sub, valid_transform)
        test_ds = RecycleDataset(test_sub, valid_transform)

        train_loader = self._build_dataloader(train_ds, True)
        valid_loader = self._build_dataloader(valid_ds, False)
        test_loader = self._build_dataloader(test_ds, False)

        return train_loader, valid_loader, test_loader

    def discover_classes(
        self,
        raw_data_dir: Path,
    ) -> list[str]:
        """Discover class labels from directory structure and populate class metadata.

        Args:
            raw_data_dir: Path to directory containing class subfolders of images.

        Returns:
            list[str]: List of detected class names sorted alphabetically.
        """
        dataset = datasets.ImageFolder(root=raw_data_dir)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        return self.classes
