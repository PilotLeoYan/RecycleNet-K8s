import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    zip_source: Path = PROJECT_ROOT / "trashnet" / "dataset-original.zip"
    raw_output_dir: Path = PROJECT_ROOT / "data" / "raw"


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def extract_dataset(self) -> Path:
        """"""
        with zipfile.ZipFile(self.config.zip_source, "r") as zip_file:
            zip_file.extractall(self.config.raw_output_dir)
        return self.config.raw_output_dir / "dataset-original"
