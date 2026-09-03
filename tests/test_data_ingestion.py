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


def test_extract_dataset_ignores_macosx_metadata(
    tmp_path: Path, fake_dataset: Path
) -> None:
    import zipfile

    zip_path = tmp_path / "mac_dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Archivo real
        zf.writestr("dataset-original/cardboard/0.jpg", b"fake image bytes")
        # Basura de macOS
        zf.writestr("__MACOSX/dataset-original/cardboard/._0.jpg", b"mac metadata")
        zf.writestr("dataset-original/.DS_Store", b"ds store")

    output_dir = tmp_path / "extracted_mac"
    config = IngestionConfig(zip_source=zip_path, raw_output_dir=output_dir)

    ingestion = DataIngestion(config)
    path = ingestion.extract_dataset()

    assert path.exists()
    assert path.name == "dataset-original"
    assert (path / "cardboard").exists()
    assert not (output_dir / "__MACOSX").exists()
