from abc import ABC, abstractmethod
from typing import List

from src.utils import CommitDiff


class CMGPreprocessor(ABC):
    @abstractmethod
    def __init__(self, model_name: str, model_provider: str, *args, **kwars):
        pass

    @abstractmethod
    def __call__(self, commit_mods: List[CommitDiff], **kwargs) -> str:
        pass
