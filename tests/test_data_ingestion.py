from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.config.schema import IngestionConfig


def test_extract_dataset_creates_output_dir(
    tmp_path: Path, fake_dataset_zip: Path
) -> None:
    config = IngestionConfig(zip_source=fake_dataset_zip, raw_output_dir=tmp_path)

    ingestion = DataIngestion(config)
    path = ingestion.extract_dataset()

    assert path.exists()
    assert path.is_dir()
