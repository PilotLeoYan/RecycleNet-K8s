from pathlib import Path

from src.components.data_ingestion import DataIngestion, DataIngestionConfig


def test_extract_dataset_creates_output_dir(
    tmp_path: Path, fake_dataset_zip: Path
) -> None:
    config = DataIngestionConfig(fake_dataset_zip, tmp_path)

    ingestion = DataIngestion(config)
    path = ingestion.extract_dataset()

    assert path.exists()
    assert path.is_dir()
