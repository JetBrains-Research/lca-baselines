from typing import List

import anthropic
import tiktoken
from transformers import AutoTokenizer


class TokenizationUtils:
    """A wrapper for two tokenization-related operations:
    - estimating the number of tokens for a prompt
    - truncating a prompt to first X tokens.
    """

    PROFILE_NAME_TO_PROVIDER_AND_MODEL = {
        "deepseek-ai/deepseek-coder-1.3b-instruct": {"model_provider": "huggingface",
                                                     "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
                                                     "context_size": 16384},
        "chat-llama-v2-7b": {"model_provider": "huggingface", "model_name": "codellama/CodeLlama-7b-Instruct-hf",
                             "context_size": 16000},
        "anthropic-claude": {"model_provider": "anthropic", "model_name": "claude", "context_size": 16000},

        "gpt-3.5-turbo-0613": {"model_provider": "openai", "model_name": "gpt-3.5-turbo", "context_size": 4096},
        "gpt-3.5-turbo-1106": {"model_provider": "openai", "model_name": "gpt-3.5-turbo", "context_size": 16385},
        "gpt-4-0613": {"model_provider": "openai", "model_name": "gpt-3.5-turbo", "context_size": 8192},
        "gpt-4-1106-preview": {"model_provider": "openai", "model_name": "gpt-4", "context_size": 128000},
    }

    def __init__(self, profile_name: str):
        model_info = self.PROFILE_NAME_TO_PROVIDER_AND_MODEL.get(profile_name, None)
        if not model_info:
            raise ValueError(f"Unknown profile {profile_name}.")

        self._model_provider = model_info["model_provider"]
        self._model_name = model_info["model_name"]
        self._context_size = model_info["context_size"]

        if self._model_provider == "openai":
            self._tokenizer = tiktoken.encoding_for_model(self._model_name)
        elif self._model_provider == "anthropic":
            self._tokenizer = anthropic.Anthropic().get_tokenizer()
        elif self._model_provider == "huggingface":
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def _encode(self, text: str) -> List[str]:
        """Estimates the number of tokens for a given string."""
        if self._model_provider == "openai":
            return self._tokenizer.encode(text)
        if self._model_provider == "anthropic":
            return self._tokenizer.encode(text)
        if self._model_provider == "huggingface":
            return self._tokenizer(text).input_ids

        raise ValueError(f"{self._model_provider} is currently not supported for token estimation.")

    def count_text_tokens(self, text: str) -> int:
        """Estimates the number of tokens for a given string."""
        return len(self._encode(text))

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimates the number of tokens for a given list of messages.

        Note: Currently, for some agents (e.g., OpenAI) the returned number might be slightly lower than the actual number of tokens, because the
        special tokens are not considered.
        """
        return sum([self.count_text_tokens(value) for message in messages for key, value in message.items()])

    def truncate(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Truncates a given list of messages to first `max_num_tokens` tokens.

        Note: A current version only truncates a last message, which might not be suitable for all use-cases.
        """
        num_tokens_except_last = self.count_messages_tokens(messages[:-1])
        messages[-1]["content"] = self._truncate(
            messages[-1]["content"], max_num_tokens=self._context_size - num_tokens_except_last
        )
        return messages

    def messages_match_context_size(self, messages: list[dict[str, str]]) -> bool:
        return self.count_messages_tokens(messages) <= self._context_size

    def text_match_context_size(self, text: str) -> bool:
        return self.text_match_context_size(text) <= self._context_size

    def _truncate(self, text: str, max_num_tokens: int) -> str:
        """Truncates a given string to first `max_num_tokens` tokens.

        1. Encodes string to a list of tokens via corresponding tokenizer.
        2. Truncates the list of tokens to first `max_num_tokens` tokens.
        3. Decodes list of tokens back to a string.
        """
        if self._model_provider == "openai":
            encoding = self._tokenizer.encode(text)[:max_num_tokens]
            return self._tokenizer.decode(encoding)
        if self._model_provider == "anthropic":
            encoding = self._tokenizer.encode(text)[:max_num_tokens]
            return self._tokenizer.decode(encoding)
        if self._model_provider == "huggingface":
            encoding = self._tokenizer(text).input_ids[:max_num_tokens]
            return self._tokenizer.decode(encoding)

        raise ValueError(f"{self._model_provider} is currently not supported for prompt truncation.")


if __name__ == '__main__':
    print(TokenizationUtils("gpt-4-0613").count_text_tokens("sfef efwe ef"))