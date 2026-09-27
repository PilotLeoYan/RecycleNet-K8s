import random
from pathlib import Path

import numpy as np
import torch

from src.components.data_transform import DataTransformation, seed_worker
from src.config.schema import TransformationConfig


def test_dataloaders_return_correct_batch_shape(fake_dataset: Path) -> None:
    config = TransformationConfig(batch_size=4, num_workers=0, pin_memory=False)
    data_loader = DataTransformation(config)
    loaders = data_loader.get_dataloaders(raw_data_dir=fake_dataset)

    assert len(loaders) == 3
    for loader in loaders:
        input_, target = next(iter(loader))
        assert input_.size() == (config.batch_size, 3, *config.image_size)
        assert target.size() == (config.batch_size,)


def test_reproducibility_with_same_seed(fake_dataset: Path) -> None:
    config = TransformationConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    torch.manual_seed(42)
    data_loaders1 = DataTransformation(config).get_dataloaders(fake_dataset)

    torch.manual_seed(42)
    data_loaders2 = DataTransformation(config).get_dataloaders(fake_dataset)

    for loader1, loader2 in zip(data_loaders1, data_loaders2):
        torch.manual_seed(43)
        input1, target1 = next(iter(loader1))

        torch.manual_seed(43)
        input2, target2 = next(iter(loader2))

        assert torch.equal(input1, input2)
        assert torch.equal(target1, target2)


def test_seed_worker_sets_numpy_and_random_seeds() -> None:
    torch.manual_seed(99999)
    seed_worker(worker_id=0)

    expected_seed = torch.initial_seed() % 2**32
    np.random.seed(expected_seed)
    expected_np = np.random.randint(0, 1000000)
    random.seed(expected_seed)
    expected_py = random.randint(0, 1000000)

    # Re-run seed_worker and verify random streams match expected
    torch.manual_seed(99999)
    seed_worker(worker_id=0)
    assert np.random.randint(0, 1000000) == expected_np
    assert random.randint(0, 1000000) == expected_py


def test_dataloader_configures_worker_init_fn_and_generator(
    fake_dataset: Path,
) -> None:
    config = TransformationConfig(
        batch_size=4, num_workers=2, pin_memory=False, seed=42
    )
    data_trans = DataTransformation(config)
    train_loader, val_loader, test_loader = data_trans.get_dataloaders(fake_dataset)

    for loader in (train_loader, val_loader, test_loader):
        assert loader.worker_init_fn is seed_worker
        assert isinstance(loader.generator, torch.Generator)


def test_reproducibility_with_explicit_seed(fake_dataset: Path) -> None:
    config = TransformationConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        seed=123,
    )

    data_loaders1 = DataTransformation(config, seed=123).get_dataloaders(fake_dataset)
    data_loaders2 = DataTransformation(config, seed=123).get_dataloaders(fake_dataset)

    for loader1, loader2 in zip(data_loaders1, data_loaders2):
        torch.manual_seed(43)
        input1, target1 = next(iter(loader1))

        torch.manual_seed(43)
        input2, target2 = next(iter(loader2))

        assert torch.equal(input1, input2)
        assert torch.equal(target1, target2)
