"""Data ingestion component for extracting and preparing raw image datasets."""

import zipfile
from pathlib import Path

from src.config.schema import IngestionConfig


class DataIngestion:
    """Manages extraction and verification of raw dataset archives.

    Attributes:
        config: Ingestion configuration containing source and destination paths.
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Initializes DataIngestion with configuration settings.

        Args:
            config: Data ingestion configuration parameters.
        """
        self.config = config

    def extract_dataset(self) -> Path:
        """Extracts the zipped dataset archive into the raw output directory.

        Returns:
            Path: Path to the extracted dataset directory containing class subfolders.

        Raises:
            FileNotFoundError: If the zip archive does not exist at `zip_source`.
            zipfile.BadZipFile: If the file is not a valid zip archive.
        """
        output_dir = Path(self.config.raw_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.config.zip_source, "r") as zip_file:
            zip_file.extractall(output_dir)

        # Dynamically detect if dataset was extracted into a single wrapper directory
        subdirs = [
            p for p in output_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
        if len(subdirs) == 1 and any(p.is_dir() for p in subdirs[0].iterdir()):
            return subdirs[0]

        return output_dir
