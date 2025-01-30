from abc import ABC, abstractmethod
from typing import Dict, Optional


class CMGBackbone(ABC):
    name: str = "base"

    @abstractmethod
    def generate_msg(self, preprocessed_commit: Dict[str, str], **kwargs) -> Dict[str, Optional[str]]:
        pass

    async def agenerate_msg(self, preprocessed_commit: Dict[str, str], **kwargs) -> Dict[str, Optional[str]]:
        return self.generate_msg(preprocessed_commit, **kwargs)
