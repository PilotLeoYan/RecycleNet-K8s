from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.config.schema import IngestionConfig


def test_extract_dataset_creates_output_dir(
    tmp_path: Path, fake_dataset_zip: Path
) -> None:
    output_dir = tmp_path / "extracted"
    config = IngestionConfig(zip_source=fake_dataset_zip, raw_output_dir=output_dir)

    ingestion = DataIngestion(config)
    path = ingestion.extract_dataset()

    assert path.exists()
    assert path.is_dir()
    assert (path / "cardboard").exists()


def test_extract_dataset_flat_zip(tmp_path: Path, fake_dataset: Path) -> None:
    import shutil

    # Create a zip file where class folders are at the root
    zip_path = tmp_path / "flat_dataset"
    shutil.make_archive(
        base_name=str(zip_path),
        format="zip",
        root_dir=str(fake_dataset),
    )
    flat_zip = tmp_path / "flat_dataset.zip"

    output_dir = tmp_path / "extracted_flat"
    config = IngestionConfig(zip_source=flat_zip, raw_output_dir=output_dir)

    ingestion = DataIngestion(config)
    path = ingestion.extract_dataset()

    assert path.exists()
    assert path.is_dir()
    assert (path / "cardboard").exists()
