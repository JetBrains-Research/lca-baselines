import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore[import-untyped]


class TokenizationUtils:
    """A wrapper for two tokenization-related operations for OpenAI & HuggingFace models:
    - calculating the number of tokens in a prompt
    - truncating a prompt to first X tokens.
    """

    def __init__(self, model_name: str, model_provider: str):
        self._model_provider = model_provider
        self._model_name = model_name

        self._tokenizer: PreTrainedTokenizerFast | tiktoken.Encoding
        if self._model_provider == "openai":
            self._tokenizer = tiktoken.encoding_for_model(self._model_name)
        elif self._model_provider == "huggingface":
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        else:
            raise ValueError(f"{self._model_provider} is currently not supported.")

    def count_tokens(self, text: str) -> int:
        """Estimates the number of tokens for a given string."""
        if self._model_provider == "openai":
            assert isinstance(self._tokenizer, tiktoken.Encoding)
            return len(self._tokenizer.encode(text))

        if self._model_provider == "huggingface":
            assert isinstance(self._tokenizer, PreTrainedTokenizerFast)
            return len(self._tokenizer(text).input_ids)

        raise ValueError(f"{self._model_provider} is currently not supported.")

    def truncate(self, text: str, max_num_tokens: int) -> str:
        """Truncates a given string to first `max_num_tokens` tokens.

        1. Encodes string to a list of tokens via corresponding tokenizer.
        2. Truncates the list of tokens to first `max_num_tokens` tokens.
        3. Decodes list of tokens back to a string.
        """
        if self._model_provider == "openai":
            assert isinstance(self._tokenizer, tiktoken.Encoding)
            encoding = self._tokenizer.encode(text)[:max_num_tokens]
            return self._tokenizer.decode(encoding)

        if self._model_provider == "huggingface":
            assert isinstance(self._tokenizer, PreTrainedTokenizerFast)
            encoding = self._tokenizer(text).input_ids[:max_num_tokens]
            return self._tokenizer.decode(encoding, skip_special_tokens=True)

        raise ValueError(f"{self._model_provider} is currently not supported.")
