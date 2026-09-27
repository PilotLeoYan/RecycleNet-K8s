import random
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.config.schema import TransformationConfig


def seed_worker(worker_id: int) -> None:
    """Sets random seeds for NumPy and Python random in DataLoader worker processes.

    Ensures deterministic augmentations and operations across multi-process data
    loading workers by reseeding external libraries using PyTorch's generated
    worker seed.

    Args:
        worker_id: The integer ID of the worker subprocess.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        seed: Optional random seed for reproducible DataLoader sampling and splitting.
        classes: List of class directory names detected in the dataset.
        class_to_idx: Mapping from class names to integer target indices.
    """

    def __init__(self, config: TransformationConfig, seed: int | None = None):
        """Initializes DataTransformation with configuration and optional seed.

        Args:
            config: Data transformation configuration.
            seed: Optional random seed for DataLoaders and dataset splitting.
        """
        self.config = config
        self.seed = seed if seed is not None else getattr(config, "seed", None)
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

    def _get_subsets(
        self, dataset: Dataset, generator: torch.Generator | None = None
    ) -> list[Subset]:
        """Splits the complete dataset into train, validation, and test subsets.

        Args:
            dataset: The full PyTorch dataset.
            generator: Optional PyTorch Generator for reproducible subset splitting.

        Returns:
            list[Subset]: Subsets for train, validation, and test splits.
        """
        if generator is None:
            generator = torch.Generator()
            if self.seed is not None:
                generator.manual_seed(self.seed)
            else:
                generator.manual_seed(torch.initial_seed())

        return random_split(
            dataset,
            (
                self.config.train_split,
                self.config.eval_split,
                self.config.test_split,
            ),
            generator=generator,
        )

    def _get_dataloader(
        self, dataset: Dataset, is_train: bool, generator: torch.Generator | None = None
    ) -> DataLoader:
        """Wraps a dataset into a configured PyTorch DataLoader.

        Args:
            dataset: Dataset or TransformSubset to wrap.
            is_train: Whether this DataLoader is for training (enables shuffling
                and drop_last).
            generator: Optional PyTorch Generator for DataLoader sampling.

        Returns:
            DataLoader: Configured PyTorch DataLoader instance.
        """
        if generator is None:
            generator = torch.Generator()
            if self.seed is not None:
                generator.manual_seed(self.seed)
            else:
                generator.manual_seed(torch.initial_seed())

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=is_train,
            worker_init_fn=seed_worker,
            generator=generator,
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
