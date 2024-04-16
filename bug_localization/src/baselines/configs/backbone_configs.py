from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from omegaconf import MISSING

from src.baselines.configs.prompt_configs import PromptConfig
from src.baselines.configs.ranker_config import RankerConfig
from src.baselines.configs.tokenizer_config import TokenizerConfig


@dataclass
class BackboneConfig:
    _target_: str = MISSING


@dataclass
class OpenAIGenBackboneConfig(BackboneConfig):
    _target_: str = f"src.baselines.backbones.gen.openai_gen_backbone.OpenAIGenBackbone"
    prompt: Optional[PromptConfig] = None
    model_name: str = MISSING
    api_key: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class TfIdfEmbBackboneConfig(BackboneConfig):
    _target_: str = f"src.baselines.backbones.emb.tfidf_emb_backbone.TfIdfEmbBackbone"
    tokenizer: TokenizerConfig = MISSING
    ranker: RankerConfig = MISSING
    pretrained_path: Optional[str] = None
