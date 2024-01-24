from dataclasses import dataclass

from omegaconf import MISSING

from .utils import BASELINES_CLASSES_ROOT_PKG


@dataclass
class PromptConfig:
    """Base config for instantiating a prompt. Should be extended for each case."""

    _target_: str = MISSING


@dataclass
class SimplePrompt(PromptConfig):
    """Config for instantiating a simple CMG prompt. Takes no additional arguments."""

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.prompts.SimpleCMGPrompt"


@dataclass
class DetailedPrompt(PromptConfig):
    """Config for instantiating a CMG prompt with a detailed instruction. Takes no additional arguments."""

    _target_: str = f"{BASELINES_CLASSES_ROOT_PKG}.prompts.DetailedCMGPrompt"
