import json
import os
from typing import Dict, Any, List

import backoff
import openai
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src import logger
from src.baselines.backbone.base_backbone import BaseBackbone
from src.baselines.backbone.utils import extract_json_from_output
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

        json_completion_content = None
        try:
            json_completion_content = extract_json_from_output(raw_completion_content)
        except Exception:
            logger.info(f"Failed to parse json from output: {raw_completion_content}")

        return {
            "messages": json.dumps(messages),
            "raw_completion": raw_completion_content,
            "json_completion": json.dumps(json_completion_content) if json_completion_content else None,
        }
