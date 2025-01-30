from typing import Any, Dict

from src.utils import TokenizationUtils

from ..utils.typing_utils import UnifiedCommitExample
from .simple_diff_preprocessor import SimpleCMGPreprocessor


class TruncationCMGPreprocessor(SimpleCMGPreprocessor):
    """Concatenates all file diffs into a single diff and then truncates the diff to first X tokens."""

    def __init__(self, max_num_tokens: int, model_name: str, model_provider: str, include_path: bool = True):
        self._max_num_tokens = max_num_tokens
        self._tokenization_utils = TokenizationUtils(model_name=model_name, model_provider=model_provider)
        super().__init__(model_name=model_name, model_provider=model_provider, include_path=include_path)

    def __call__(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, Any]:
        processed_commit = super().__call__(commit)
        return {
            k: self._tokenization_utils.truncate(v, max_num_tokens=self._max_num_tokens)
            for k, v in processed_commit.items()
        }
