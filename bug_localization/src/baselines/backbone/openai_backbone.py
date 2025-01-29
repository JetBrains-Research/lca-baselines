import os
from typing import Dict, Any, List

import backoff
import openai
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.baselines.backbone.base_backbone import BaseBackbone
from src.baselines.context_composers.base_context_composer import BaseContextComposer
from src.baselines.utils.type_utils import ChatMessage


class OpenAIBackbone(BaseBackbone):

    def __init__(
            self,
            name: str,
            model_name: str,
            parameters: Dict[str, Any],
            context_composer: BaseContextComposer,
    ):
        super().__init__(name)
        self._client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self._model_name = model_name
        self._parameters = parameters
        self._context_composer = context_composer

    @backoff.on_exception(backoff.expo, openai.APIError)
    def _get_chat_completion(self, messages: List[ChatMessage]) -> ChatCompletion:
        return self._client.chat.completions.create(
            messages=messages,
            model=self._model_name,
            **self._parameters
        )

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, dp: dict) -> Dict[str, Any]:
        messages = self._context_composer.compose_chat(dp, self._model_name)
        completion = self._get_chat_completion(messages)
        raw_completion_content = completion.choices[0].message.content

        return {
            "messages": messages,
            "completion": raw_completion_content,
        }
