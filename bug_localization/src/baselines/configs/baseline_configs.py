from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.baselines.configs.backbone_configs import BackboneConfig, OpenAIGenBackboneConfig, TfIdfEmbBackboneConfig
from src.baselines.configs.data_configs import DataSourceConfig, HFDataSourceConfig
from src.baselines.configs.prompt_configs import FileListPromptConfig
from src.baselines.configs.ranker_config import CosineDistanceRankerConfig
from src.baselines.configs.tokenizer_config import NltkTokenizerConfig


@dataclass
class BaselineConfig:
    backbone: BackboneConfig = MISSING
    data_src: DataSourceConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="baseline_config", node=BaselineConfig)
# all available options for the backbone
cs.store(name="openai", group="backbone", node=OpenAIGenBackboneConfig)
cs.store(name="tfidf", group="backbone", node=TfIdfEmbBackboneConfig)
# all available options for the tokenizer
cs.store(name="tfidf", group="backbone/tokenizer", node=NltkTokenizerConfig)
# all available options for the ranker
cs.store(name="tfidf", group="backbone/ranker", node=CosineDistanceRankerConfig)
# all available options for the prompt
cs.store(name="filelist", group="backbone/prompt", node=FileListPromptConfig)
# all available options for the input
cs.store(name="hf", group="data_src", node=HFDataSourceConfig)
