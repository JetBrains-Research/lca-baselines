import os

import numpy as np

from src.utils.git_utils import get_repo_content_on_commit, get_changed_files_between_commits


def get_repo_content(datapoint: dict, category: str, repos_path: str) -> dict[str, str]:
    extensions = category if category != "mixed" else None
    repo_path = os.path.join(repos_path, f"{datapoint['repo_owner']}__{datapoint['repo_name']}")
    repo_content = get_repo_content_on_commit(repo_path, datapoint['base_sha'], extensions)

    return repo_content


def get_changed_files(datapoint: dict, category: str, repos_path: str) -> np.ndarray[str]:
    extensions = category if category != "mixed" else None
    repo_path = os.path.join(repos_path, f"{datapoint['repo_owner']}__{datapoint['repo_name']}")
    changed_files = get_changed_files_between_commits(repo_path, datapoint['base_sha'], datapoint['head_sha'],
                                                      extensions)

    return np.asarray(changed_files, dtype=str)
