import os
from typing import Optional

import numpy as np

from baselines.baseline_tokenizers import BaseTokenizer


class EmbedBaseline:

    def __init__(self,
                 pretrained_path: str,
                 tokenizer: Optional[BaseTokenizer]):
        self.pretrained_path = pretrained_path
        self.tokenizer = tokenizer

    @staticmethod
    def name() -> str:
        pass

    def embed(self, file_contents: np.ndarray[str]) -> np.ndarray[float]:
        pass

    def get_embeddings_path(self) -> str:
        return os.path.join(self.pretrained_path, self.name(), 'embeddings.npy')

    def dump_embeddings(self, embeddings: np.ndarray[float]):
        np.save(self.get_embeddings_path(), embeddings)

    def load_embeddings(self) -> Optional[np.ndarray[float]]:
        embeddings_path = self.get_embeddings_path()
        if os.path.exists(embeddings_path):
            return np.load(embeddings_path)

        return None


class ScoreBaseline:

    @staticmethod
    def name() -> str:
        pass

    def score(self, issue_text: str, file_paths: np.ndarray[str], file_contents: dict[str, str]) -> np.ndarray[int]:
        pass
