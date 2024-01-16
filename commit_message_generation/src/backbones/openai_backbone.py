import os
from typing import Any, Dict, List, Optional

import backoff
import openai

from src.prompts import CMGPrompt
from src.utils import ChatMessage

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

    @backoff.on_exception(backoff.expo, openai.APIError)
    def _get_chat_completion(self, messages: List[ChatMessage]) -> openai.types.chat.ChatCompletion:
        return self._client.chat.completions.create(messages=messages, model=self._model_name, **self._parameters)  # type: ignore[arg-type]

    def generate_msg(self, preprocessed_commit_mods: str, **kwargs) -> Dict[str, Optional[str]]:
        prompt = self._prompt.chat(preprocessed_commit_mods)
        response = self._get_chat_completion(messages=prompt)
        assert response.choices[0].message.content, "Empty content in OpenAI API response."
        return {
            "prediction": response.choices[0].message.content,
            "created": str(response.created),
            "model": response.model,
            "system_fingerprint": response.system_fingerprint,
        }
