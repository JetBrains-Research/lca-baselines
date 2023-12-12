from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .backbones_configs import BackboneConfig, HFBackboneConfig, OpenAIBackboneConfig
from .data_sources_configs import DataSourceConfig, HFDataSourceConfig, LocalFileDataSourceConfig
from .preprocessor_configs import BasePreprocessorConfig, SimpleCMGDiffPreprocessor, TruncationCMGDiffPreprocessor
from .prompts_configs import DetailedPrompt, SimplePrompt


@dataclass
class WandbConfig:
    """
    Configuration for logging via Weights & Biases.

    Attributes:
        use_wandb: Whether W&B will be used for logging or not.
        name: Run name.
        project: Name of a project this run will appear in.
        local_artifact: True to store the reference to a local file as an artifact, False to upload predictions to W&B.

    """

    use_wandb: bool = True
    name: Optional[str] = None
    project: str = "lca_cmg"
    local_artifact: bool = False


@dataclass
class BaselineConfig:
    """Main configuration class for a baseline for CMG task.

    Attributes:
        backbone: Configuration for a model.
        preprocessor: Configuration for a data preprocessor.
        logger: Configuration for logging via Weights & Biases.
        data_src: Configuration for input.
    """

    backbone: BackboneConfig = MISSING
    preprocessor: BasePreprocessorConfig = MISSING
    logger: WandbConfig = field(default_factory=WandbConfig)
    data_src: DataSourceConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="baseline_config", node=BaselineConfig)
# all available options for the backbone
cs.store(name="openai", group="backbone", node=OpenAIBackboneConfig)
cs.store(name="hf", group="backbone", node=HFBackboneConfig)
# all available options for the prompt
cs.store(name="simple", group="backbone/prompt", node=SimplePrompt)
cs.store(name="detailed", group="backbone/prompt", node=DetailedPrompt)
# all available options for the preprocessor
cs.store(name="simple", group="preprocessor", node=SimpleCMGDiffPreprocessor)
cs.store(name="truncation", group="preprocessor", node=TruncationCMGDiffPreprocessor)
# all available options for the input
cs.store(name="local", group="data_src", node=LocalFileDataSourceConfig)
cs.store(name="hf", group="data_src", node=HFDataSourceConfig)
