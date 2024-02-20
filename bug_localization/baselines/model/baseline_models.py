import os

import numpy as np
from datasets import Dataset

from baselines.metrics.metrics import Metrics
from utils.git_utils import get_repo_content_on_commit, get_changed_files_between_commits


class Baseline:

    def __init__(self, repos_path: str):
        self.repos_path = repos_path

    def run(self, dataset: Dataset, category: str, split: str) -> list[Metrics]:
        pass

    def get_repo_content(self, datapoint: dict, category: str) -> dict[str, str]:
        extensions = category if category != "mixed" else None
        repo_path = os.path.join(self.repos_path, f"{datapoint['repo_owner']}__{datapoint['repo_name']}")
        repo_content = get_repo_content_on_commit(repo_path, datapoint['base_sha'], extensions)

        return repo_content

    def get_changed_files(self, datapoint: dict, category: str) -> np.ndarray[str]:
        extensions = category if category != "mixed" else None
        repo_path = os.path.join(self.repos_path, f"{datapoint['repo_owner']}__{datapoint['repo_name']}")
        changed_files = get_changed_files_between_commits(repo_path, datapoint['base_sha'], datapoint['head_sha'],
                                                          extensions)

        return np.asarray(changed_files, dtype=str)
