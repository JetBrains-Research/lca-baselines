from typing import Dict, Any, Optional, List

import backoff
import openai
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.backbones.gen.prompts.base_prompt import BasePrompt
from src.baselines.utils.prompt_utils import batch_project_context, parse_list_files_completion
from src.baselines.utils.type_utils import ChatMessage


class OpenAIGenBackbone(BaseBackbone):

    def __init__(
            self,
            model_name: str,
            prompt: BasePrompt,
            parameters: Dict[str, Any],
            api_key: Optional[str] = None,
    ):
        self._client = openai.OpenAI(api_key=api_key)
        self._model_name = model_name
        self._prompt = prompt
        self._parameters = parameters

    name: str = "openai"

    @backoff.on_exception(backoff.expo, openai.APIError)
    def _get_chat_completion(self, messages: List[ChatMessage]) -> ChatCompletion:
        return self._client.chat.completions.create(messages=messages, model=self._model_name,
                                                    **self._parameters)  # type: ignore[arg-type]

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        batched_project_contents = batch_project_context(
            self._model_name, self._prompt, issue_description, repo_content, True
        )

        expected_files = set()
        raw_completions = []
        for batched_project_content in batched_project_contents:
            messages = self._prompt.chat(issue_description, batched_project_content)

            completion = self._get_chat_completion(messages)
            raw_completion_content = completion.choices[0].message.content
            raw_completions.append(raw_completion_content)

            expected_files += parse_list_files_completion(raw_completion_content, repo_content)

        return {
            "expected_files": list(expected_files),
            "raw_completions": raw_completions
        }
