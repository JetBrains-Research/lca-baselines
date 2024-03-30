import os

import numpy as np
from datasets import Dataset

from src.baselines.metrics.metrics import Metrics
from src.utils.git_utils import get_repo_content_on_commit, get_changed_files_between_commits


class Baseline:

    def __init__(self, repos_path: str):
        self.repos_path = repos_path

    def run(self, dataset: Dataset, category: str, split: str) -> list[Metrics]:
        pass


