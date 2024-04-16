from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseBackbone(ABC):
    name: str = "base"

    @abstractmethod
    def localize_bugs(self, issue_description: str, repo_content: Dict[str, str], **kwargs) -> Dict[str, Any]:
        pass
