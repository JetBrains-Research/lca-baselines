from abc import ABC, abstractmethod
from typing import Dict, Optional


class CMGBackbone(ABC):
    name: str = "base"

    @abstractmethod
    def generate_msg(self, preprocessed_commit_mods: str, **kwargs) -> Dict[str, Optional[str]]:
        pass
