import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    fake_dataset = tmp_path / "dataset-original"
    classes = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

    images_sizes = ((224, 224), (280, 224), (224, 280), (280, 280), (330, 380))
    image_format = "jpg"

    for class_ in classes:
        path = fake_dataset / class_
        path.mkdir(parents=True, exist_ok=True)

        for idx, (height, width) in enumerate(images_sizes):
            rnd_array = np.random.randint(
                0, 256, size=(height, width, 3), dtype=np.uint8
            )
            rnd_image = Image.fromarray(rnd_array, "RGB")
            rnd_image.save(path / f"{idx}.{image_format}")
    return fake_dataset


@pytest.fixture
def fake_dataset_zip(tmp_path: Path, fake_dataset: Path) -> Path:
    zip_path = tmp_path / "fake-compressed"

    shutil.make_archive(base_name=str(zip_path), format="zip", root_dir=fake_dataset)

    return tmp_path / "fake-compressed.zip"
