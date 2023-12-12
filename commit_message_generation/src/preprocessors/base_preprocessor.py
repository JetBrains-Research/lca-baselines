from abc import ABC, abstractmethod
from typing import List

from src.utils import CommitDiff, CommitMods


class CMGDiffPreprocessor(ABC):
    @abstractmethod
    def __init__(self, model_name: str, model_provider: str, *args, **kwars):
        pass

    @abstractmethod
    def __call__(self, commit_diff: List[CommitDiff], **kwargs) -> str:
        pass


class CMGPreprocessor:
    def __init__(
        self,
        preprocessor: CMGDiffPreprocessor,
    ):
        self._preprocessor = preprocessor

    def __call__(self, commit_mods: CommitMods, **kwargs):
        if commit_mods["diff"]:
            return self._preprocessor(commit_mods["diff"], **kwargs)

        raise ValueError("Provided input is not supported.")
