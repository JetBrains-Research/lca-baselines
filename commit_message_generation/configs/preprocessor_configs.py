from dataclasses import dataclass

from omegaconf import MISSING

from .utils import BASELINES_CLASSES_ROOT_PKG


@dataclass
class BasePreprocessorConfig:
    """Base config for instantiating a preprocessor. Should be extended for each case."""

    _target_: str = MISSING


@dataclass
class SimpleCMGDiffPreprocessor(BasePreprocessorConfig):
    """Config for instantiating a simple CMG preprocessor (simply concatenates all file mods into a single diff).
    Takes no additional arguments."""

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.preprocessors.SimpleCMGPreprocessor"
    include_path: bool = True


@dataclass
class TruncationCMGDiffPreprocessor(BasePreprocessorConfig):
    """Config for instantiating a simple CMG preprocessor with truncation:
    simply concatenates all file mods into a single diff, tokenizes it, truncates to first `max_num_tokens` tokens,
    decodes back to string."""

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.preprocessors.TruncationCMGPreprocessor"
    max_num_tokens: int = MISSING
    include_path: bool = True
