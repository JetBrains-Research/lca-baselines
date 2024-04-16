from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class PromptConfig:
    _target_: str = MISSING


@dataclass
class FileListPromptConfig(PromptConfig):
    _target_: str = f"src.baselines.backbones.gen.prompts.file_list_prompt.FileListPrompt"
