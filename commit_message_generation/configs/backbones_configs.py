from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING

from .prompts_configs import PromptConfig
from .utils import BASELINES_CLASSES_ROOT_PKG


@dataclass
class BackboneConfig:
    """Base config for instantiating a backbone. Should be extended for each case."""

    _target_: str = MISSING
    prompt: Optional[PromptConfig] = None
    model_name: str = MISSING


@dataclass
class OpenAIBackboneConfig(BackboneConfig):
    """Config for instantiating an OpenAI backbone.

    Attributes:
        model_name: Name for LLM profile to use in OpenAI API.
        prompt: Name for one of the supported prompt configurations.
        token_path: Path to file with an OpenAI API key.
        parameters: Arbitrary keyword parameters that can be passed to corresponding Completion or ChatCompletion endpoint.
    """

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.backbones.OpenAIBackbone"
    token_path: str = "data/token.txt"
    parameters: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class HFModelConfig:
    """Config for initializing a HuggingFace model. Includes some options; the rest can be added via Hydra's override
    (e.g., ++backbone.model_kwargs.cache_dir=some_dir).

    All kwargs will be passed to transformers.PreTrainedModel.from_pretrained. See docs here:
    https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    """

    torch_dtype: str = "auto"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class HFGenerationConfig:
    """Config for generation via HuggingFace models. Includes some options; the rest can be added via Hydra's override
    (e.g., ++generation.forced_bos_token_id=0).

    All kwargs will be passed to transformers.GenerationConfig. See docs here:
    https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig"""

    do_sample: bool = True
    temperature: float = 0.8
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None


@dataclass
class HFBackboneConfig(BackboneConfig):
    """Config for instantiating a HuggingFace backbone.

    Attributes:
        model_name: Name of the model on HF Hub or local path to checkpoint.
        prompt: Name for one of the supported prompt configurations (optional, if not given, raw diff will be passed).
        is_encoder_decoder: True for seq2seq models, False for decoder-only models.
        model_kwargs: Config for model initialization.
        generation: Config for generation.
        device: Device to put model & data on
          (docs here: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        seed: Seed for reproducibility; if None, will be chosen randomly.
        use_bettertransformer: Set to True to enable BetterTransformer
          (details here: https://huggingface.co/docs/transformers/perf_infer_gpu_one#bettertransformer)
    """

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.backbones.HuggingFaceBackbone"
    is_encoder_decoder: bool = MISSING
    model_kwargs: HFModelConfig = field(default_factory=HFModelConfig)
    generation: HFGenerationConfig = field(default_factory=HFGenerationConfig)
    device: str = "cpu"
    seed: Optional[int] = None
    use_bettertransformer: bool = False
