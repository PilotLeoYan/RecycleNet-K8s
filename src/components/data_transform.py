from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset, random_split


@dataclass
class DataTransformationConfig:
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
    def __init__(self, subset: Subset, transform: v2.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.subset[idx]
        return self.transform(image), label


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

    def _get_transforms(self) -> tuple[v2.Compose, v2.Compose]:
        """"""
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
        """"""
        return datasets.ImageFolder(root=raw_data_dir)

    def _get_subsets(self, dataset: Dataset) -> list[Subset]:
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
