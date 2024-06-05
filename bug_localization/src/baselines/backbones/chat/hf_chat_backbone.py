from typing import Dict, Any

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.baselines.utils.prompt_utils import batch_project_context


class HfChatBackbone(BaseBackbone):

    def __init__(self, name: str, model_name: str, prompt: ChatBasePrompt) -> None:
        super().__init__(name)
        self._model_name = model_name
        self._prompt = prompt

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        batched_project_contents = batch_project_context(self._model_name, self._prompt, issue_description,
                                                         repo_content, True)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self._model_name, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()

        expected_files = set()
        for batched_project_content in batched_project_contents:
            messages = self._prompt.chat(issue_description, batched_project_content)

            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
                model.device)

            # TODO: move to parameters
            outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95,
                                     num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print(output)

            for file in output.split('\n'):
                if file in repo_content:
                    expected_files.add(file)

        return {
            "expected_files": list(expected_files)
        }
