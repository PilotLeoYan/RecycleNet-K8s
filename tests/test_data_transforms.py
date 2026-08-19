from pathlib import Path

import torch

from src.components.data_transform import DataTransformation, DataTransformationConfig


def test_dataloaders_return_correct_batch_shape(fake_dataset: Path) -> None:
    config = DataTransformationConfig(batch_size=4, num_workers=0, pin_memory=False)
    data_loader = DataTransformation(config)
    loaders = data_loader.get_dataloaders(raw_data_dir=fake_dataset)

    assert len(loaders) == 3
    for loader in loaders:
        input_, target = next(iter(loader))
        assert input_.size() == (config.batch_size, 3, *config.image_size)
        assert target.size() == (config.batch_size,)


def test_reproducibility_with_same_seed(fake_dataset: Path) -> None:
    config = DataTransformationConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    data_loaders1 = DataTransformation(config).get_dataloaders(fake_dataset)
    data_loaders2 = DataTransformation(config).get_dataloaders(fake_dataset)

    for loader1, loader2 in zip(data_loaders1, data_loaders2):
        torch.manual_seed(42)
        input1, target1 = next(iter(loader1))

        torch.manual_seed(42)
        input2, target2 = next(iter(loader2))

        assert torch.equal(input1, input2)
        assert torch.equal(target1, target2)
