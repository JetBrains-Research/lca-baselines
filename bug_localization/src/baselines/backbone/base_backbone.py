from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseBackbone(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def localize_bugs(self, dp: dict, **kwargs) -> Dict[str, Any]:
        pass
