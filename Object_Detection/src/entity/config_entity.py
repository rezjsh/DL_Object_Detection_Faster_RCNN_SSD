from ast import Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the Data Ingestion Stage.
    """
    download_location: Path
    workspace: str
    project_name: str
    version: int
    format: str

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for the Data Validation Stage.
    """
    data_dir: Path 
    required_files: list 
    status_file: Path

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for the Dataset.
    """
    data_dir: Path
    train_dir: Path
    valid_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the Data Transformation Stage.
    """
    resize: bool
    image_size: Tuple[int, int]
    flip_prob: float

@dataclass(frozen=True)
class DataLoaderConfig:
    """
    Configuration for the Data Loader Stage.
    """
    train_batch_size: int
    valid_batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool