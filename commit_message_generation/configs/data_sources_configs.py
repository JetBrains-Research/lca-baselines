from dataclasses import dataclass
from typing import List, Optional

from omegaconf import MISSING

from .utils import BASELINES_CLASSES_ROOT_PKG


@dataclass
class DataSourceConfig:
    _target_: str = MISSING


@dataclass
class LocalFileDataSourceConfig(DataSourceConfig):
    """Configuration for an iterator over a local file."""

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.data_sources.LocalFileDataSource"
    path: str = MISSING


@dataclass
class HFDataSourceConfig(DataSourceConfig):
    """Configuration for an iterator over a HuggingFace Hub dataset."""

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.data_sources.HFDataSource"
    cache_dir: Optional[str] = None
    hub_name: str = MISSING
    configs: List[str] | str = MISSING
    split: str = "test"
