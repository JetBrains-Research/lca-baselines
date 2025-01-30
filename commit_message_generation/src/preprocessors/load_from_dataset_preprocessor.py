from typing import Dict

from datasets import load_dataset  # type: ignore[import-untyped]

from ..utils.typing_utils import UnifiedCommitExample
from .base_preprocessor import CMGPreprocessor


class LoadFromDatasetPreprocessor(CMGPreprocessor):
    def __init__(
        self,
        model_name: str,
        model_provider: str,
        hf_repo_id: str,
        hf_repo_config: str,
        hf_repo_split: str,
        *args,
        **kwargs,
    ):
        dataset = load_dataset(hf_repo_id, hf_repo_config, split=hf_repo_split).to_pandas()
        self.data_map = {
            (row["repo"], row["hash"]): {k: v for k, v in row.items() if k not in ["repo", "hash"]}
            for _, row in dataset.iterrows()
        }

    def __call__(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, str]:
        return {**self.data_map[(commit["repo"], commit["hash"])]}
