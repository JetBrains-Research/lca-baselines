from abc import ABC, abstractmethod
from typing import List

from src.baselines.utils.type_utils import ChatMessage


class BasePrompt(ABC):

    @abstractmethod
    def base_prompt(self, issue_description: str, project_content: dict[str, str]) -> str:
        pass

    def complete(self, issue_description: str, project_content: dict[str, str]) -> str:
        return self.base_prompt(issue_description, project_content)

    def chat(self, issue_description: str, project_content: dict[str, str]) -> List[ChatMessage]:
        return [
            {
                "role": "system",
                "content": "You are python java and kotlin developer or "
                           "QA or support engineer who is in a duty and looking through bugs reports in GitHub repo."
            },
            {
                "role": "user",
                "content": self.base_prompt(issue_description, project_content)
            },
        ]
