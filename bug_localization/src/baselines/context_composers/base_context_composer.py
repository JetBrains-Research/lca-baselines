from abc import ABC, abstractmethod
from pathlib import Path

from src import PROJECT_DIR
from src.baselines.utils.type_utils import ChatMessage


class BaseContextComposer(ABC):

    @staticmethod
    def _read_prompt(prompt_template_path: str) -> str:
        absolute_path = PROJECT_DIR / Path(prompt_template_path)
        with absolute_path.open("r") as f:
            return f.read()

    @abstractmethod
    def compose_chat(self, dp: dict, model_name: str) -> list[ChatMessage]:
        pass
