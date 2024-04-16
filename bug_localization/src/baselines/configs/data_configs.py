from dataclasses import dataclass
from typing import Optional, List

from omegaconf import MISSING


@dataclass
class DataSourceConfig:
    _target_: str = MISSING


@dataclass
class HFDataSourceConfig(DataSourceConfig):
    _target_: str = f"src.baselines.data_sources.hf.HFDataSource"
    cache_dir: Optional[str] = None
    hub_name: str = MISSING
    configs: List[str] = MISSING
    split: str = "test"
