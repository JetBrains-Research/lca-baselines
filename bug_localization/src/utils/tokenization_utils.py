from typing import List

import tiktoken
from transformers import AutoTokenizer


class TokenizationUtils:
    """A wrapper for two tokenization-related operations:
    - estimating the number of tokens for a prompt
    - truncating a prompt to first X tokens.
    """

    PROFILE_NAME_TO_PROVIDER_AND_MODEL = {
        "gpt-3.5-turbo-1106": {"model_provider": "openai", "model_name": "gpt-3.5-turbo", "context_size": 16000},
        "gpt-4o": {"model_provider": "openai", "model_name": "gpt-4", "context_size": 32000},
        "gpt-4o-mini": {"model_provider": "openai", "model_name": "gpt-4", "context_size": 8000},
        "claude-3.5-sonnet": {"model_provider": "anthropic", "model_name": "claude-3.5-sonnet", "context_size": 10000},
        "claude-3-opus": {"model_provider": "anthropic", "model_name": "claude-3-opus", "context_size": 10000},
        "claude-3-haiku": {"model_provider": "anthropic", "model_name": "claude-3-haiku", "context_size": 10000},
        "gemini-1.5-pro": {"model_provider": "google", "model_name": "gemini-1.5-pro", "context_size": 10000},
        "llama-3.1-405B": {"model_provider": "huggingface", "model_name": "llama-3.1-405B", "context_size": 4000},
        "llama-3.1-70B": {"model_provider": "huggingface", "model_name": "llama-3.1-70B", "context_size": 2000},
        "llama-3.1-8B": {"model_provider": "huggingface", "model_name": "llama-3.1-8B", "context_size": 1000},
        "llama-3.2-3B": {"model_provider": "huggingface", "model_name": "llama-3.2-3B", "context_size": 2000},
        "llama-3.2-1B-mini": {"model_provider": "huggingface", "model_name": "llama-3.2-1B-mini", "context_size": 1000},
    }

    def __init__(self, profile_name: str):
        model_info = self.PROFILE_NAME_TO_PROVIDER_AND_MODEL.get(profile_name, None)
        if not model_info:
            raise ValueError(f"Unknown profile {profile_name}.")

        self._model_provider = model_info["model_provider"]
        self._model_name = model_info["model_name"]
        self._context_size = model_info["context_size"]

        if self._model_provider == "openai":
            # OpenAI models (e.g., GPT-4, GPT-3.5)
            self._tokenizer = tiktoken.encoding_for_model(self._model_name)
        elif self._model_provider == "anthropic":
            # Anthropic models (e.g., Claude 3.5, Claude 3)
            # TODO: implement, use gpt-4 for now instead
            self._tokenizer = tiktoken.encoding_for_model('gpt-4')
        elif self._model_provider == "huggingface":
            # Hugging Face models (e.g., Llama 3.1, Llama 3.2)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        elif self._model_provider == "google":
            # Gemini models by Google (e.g., Gemini 1.5 Pro)
            # TODO: implement, use gpt-4 for now instead
            self._tokenizer = tiktoken.encoding_for_model('gpt-4')
        else:
            raise ValueError(f"Unsupported model provider {self._model_provider}.")

    def _encode(self, text: str) -> List[str]:
        """Estimates the number of tokens for a given string."""
        if self._model_provider == "openai":
            return self._tokenizer.encode(text)
        elif self._model_provider == "anthropic":
            return self._tokenizer.encode(text)
        elif self._model_provider == "huggingface":
            return self._tokenizer(text).input_ids
        elif self._model_provider == "google":
            return self._tokenizer.encode(text)

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
    print(TokenizationUtils("gemini-1.5-pro").count_text_tokens("sfef efwe ef"))
    print(TokenizationUtils("gpt-4o-mini").count_text_tokens("sfef efwe ef"))
    print(TokenizationUtils("gemini-1.5-pro").count_text_tokens("sfef efwe ef"))
    print(TokenizationUtils("claude-3-haiku").count_text_tokens("sfef efwe ef"))
