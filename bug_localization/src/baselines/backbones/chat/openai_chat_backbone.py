from typing import Dict, Any, Optional, List

import backoff
import openai
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.baselines.utils.prompt_utils import batch_project_context, parse_list_files_completion
from src.baselines.utils.type_utils import ChatMessage


class OpenAIChatBackbone(BaseBackbone):

    def __init__(
            self,
            name: str,
            model_name: str,
            prompt: ChatBasePrompt,
            parameters: Dict[str, Any],
            api_key: Optional[str] = None,
    ):
        super().__init__(name)
        self._client = openai.OpenAI(api_key=api_key)
        self._model_name = model_name
        self._prompt = prompt
        self._parameters = parameters

    @backoff.on_exception(backoff.expo, openai.APIError)
    def _get_chat_completion(self, messages: List[ChatMessage]) -> ChatCompletion:
        return self._client.chat.completions.create(
            messages=messages,
            model=self._model_name,
            **self._parameters
        )

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        batched_project_contents = batch_project_context(
            self._model_name, self._prompt, issue_description, repo_content, True
        )

        files = set()
        final_files = set()
        raw_completions = []
        for batched_project_content in batched_project_contents:
            messages = self._prompt.chat(issue_description, batched_project_content)

            completion = self._get_chat_completion(messages)
            raw_completion_content = completion.choices[0].message.content
            raw_completions.append(raw_completion_content)
            files.update(parse_list_files_completion(raw_completion_content))

        if len(batched_project_contents) > 1:
            messages = self._prompt.chat(issue_description, {f: repo_content[f] for f in files if f in repo_content})
            completion = self._get_chat_completion(messages)
            raw_completion_content = completion.choices[0].message.content
            raw_completions.append(raw_completion_content)
            final_files.update(parse_list_files_completion(raw_completion_content))
        else:
            final_files = [f for f in files if f in repo_content]

        return {
            "all_generated_files": list(files),
            "final_files": list(final_files),
            "raw_completions": raw_completions,
            "batches_count": len(batched_project_contents)
        }
