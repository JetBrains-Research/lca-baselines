import numpy as np
from datasets import Dataset

from src.baselines.model.baseline_models import Baseline


class ScoreBaseline(Baseline):

    def init(self, data_path: str):
        super().__init__(data_path)

    @staticmethod
    def name() -> str:
        pass

    def score(self, issue_text: str, file_paths: np.ndarray[str], file_contents: dict[str, str]) -> np.ndarray[int]:
        pass

    def run(self, dataset: Dataset, category: str, split: str):
        pass
