import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from jinja2.exceptions import TemplateError
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast  # type: ignore[import-untyped]

from src.utils import ChatMessage

DEFAULT_HF_CHAT_TEMPLATE = PreTrainedTokenizerBase().default_chat_template


class CMGPrompt(ABC):
    @abstractmethod
    def _base_prompt(self, diff: str) -> str:
        pass

    def complete(self, diff: str) -> str:
        return self._base_prompt(diff)

    def chat(self, diff: str) -> List[ChatMessage]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that generates commit message subjects for given diffs.",
            },
            {"role": "user", "content": self._base_prompt(diff)},
        ]

    def hf(self, diff: str, prompt_format: Optional[str], tokenizer: Optional[PreTrainedTokenizerFast] = None) -> str:
        if (
            not prompt_format
            and tokenizer
            and (tokenizer.chat_template is not None or tokenizer.default_chat_template != DEFAULT_HF_CHAT_TEMPLATE)
        ):
            logging.info("Using chat template from HF tokenizer.")
            chat_messages = self.chat(diff)
            assert (
                len(chat_messages) == 2 and chat_messages[0]["role"] == "system" and chat_messages[1]["role"] == "user"
            ), "By default, a single system message followed by a single user message are expected."
            try:
                return tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            except TemplateError:
                logging.error("Couldn't apply chat template. Concatenating system message with user message.")
                chat_messages = [
                    {"role": "user", "content": chat_messages[0]["content"] + "\n" + chat_messages[1]["content"]}
                ]
                return tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        if not prompt_format:
            logging.warning("Prompt format not passed, using raw diff as model input.")
            return self.complete(diff)

        if prompt_format == "octocoder":  # https://huggingface.co/bigcode/octocoder#intended-use
            completion_prompt = self.complete(diff)
            if completion_prompt.endswith("Commit message:\n"):
                completion_prompt = completion_prompt[: len("Commit message:\n")]
            return f"Question: {completion_prompt}\n\nAnswer:"

        if prompt_format == "starchat":  # https://huggingface.co/HuggingFaceH4/starchat-beta#intended-uses--limitations
            chat_messages = self.chat(diff)
            assert (
                len(chat_messages) == 2 and chat_messages[0]["role"] == "system" and chat_messages[1]["role"] == "user"
            ), "By default, a single system message followed by a single user message are expected."
            return "".join(
                [
                    f"<|system|>{chat_messages[0]['content']}<|end|>\n",
                    f"<|user|>\n{chat_messages[1]['content']}<|end|>\n",
                    "<|assistant|>",
                ]
            )

        if prompt_format == "llama":  # https://huggingface.co/blog/codellama#conversational-instructions
            chat_messages = self.chat(diff)
            assert (
                len(chat_messages) == 2 and chat_messages[0]["role"] == "system" and chat_messages[1]["role"] == "user"
            ), "By default, a single system message followed by a single user message are expected."
            return f"<s>[INST] <<SYS>>\n{chat_messages[0]['content']}\n<</SYS>>\n\n{chat_messages[1]['content']}[/INST]"

        if prompt_format == "alpaca":  # https://github.com/sahil280114/codealpaca#data-release
            completion_prompt = self.complete(diff)
            if completion_prompt.endswith("Commit message:\n"):
                completion_prompt = completion_prompt[: len("Commit message:\n")]
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{completion_prompt}\n\n### Response:\n"

        raise NotImplementedError("Unknown prompt format.")
