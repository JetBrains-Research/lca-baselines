from typing import List

from src.utils import CommitDiff, TokenizationUtils

from .simple_diff_preprocessor import SimpleCMGPreprocessor


class TruncationCMGPreprocessor(SimpleCMGPreprocessor):
    """Concatenates all file diffs into a single diff and then truncates the diff to first X tokens."""

    def __init__(self, max_num_tokens: int, model_name: str, model_provider: str, include_path: bool = True):
        self._max_num_tokens = max_num_tokens
        self._tokenization_utils = TokenizationUtils(model_name=model_name, model_provider=model_provider)
        super().__init__(model_name=model_name, model_provider=model_provider, include_path=include_path)

    def __call__(self, commit_mods: List[CommitDiff], **kwargs) -> str:
        processed_diff = super().__call__(commit_mods)
        return self._tokenization_utils.truncate(processed_diff, max_num_tokens=self._max_num_tokens)
