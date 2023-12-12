import os
from typing import Any, Dict, Optional

import openai

from src.prompts import CMGPrompt

from .base_backbone import CMGBackbone


class OpenAIBackbone(CMGBackbone):
    name: str = "openai"

    def __init__(
        self,
        model_name: str,
        prompt: CMGPrompt,
        parameters: Dict[str, Any],
        api_key: Optional[str] = None,
    ):
        self._client = openai.OpenAI(api_key=api_key if api_key else os.environ.get("OPENAI_API_KEY"))
        self._model_name = model_name
        self._prompt = prompt
        self._parameters = parameters

    def generate_msg(self, preprocessed_commit_mods: str, **kwargs) -> str:
        prompt = self._prompt.chat(preprocessed_commit_mods)
        return self._client.chat.completions.create(messages=prompt, model=self._model_name, **self._parameters)["choices"][0][  # type: ignore[index, arg-type]
            "message"
        ][
            "content"
        ]
