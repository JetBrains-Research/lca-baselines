from abc import ABC, abstractmethod
from typing import Dict, Any


class Baseline(ABC):
    name: str = "base"

    @abstractmethod
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str], **kwargs) -> Dict[str, Any]:
        pass
