from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    name: str = MISSING


@dataclass
class DataSourceConfig:
    _target_: str = MISSING


@dataclass
class BaselineConfig:
    backbone: BackboneConfig = MISSING
    data_source: DataSourceConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="baseline_config", node=BaselineConfig)
