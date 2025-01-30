from abc import ABC, abstractmethod
from typing import Any, Dict

from src.utils.typing_utils import UnifiedCommitExample


class CMGPreprocessor(ABC):
    @abstractmethod
    def __init__(self, model_name: str, model_provider: str, *args, **kwars):
        pass

    @abstractmethod
    def __call__(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, Any]:
        pass
