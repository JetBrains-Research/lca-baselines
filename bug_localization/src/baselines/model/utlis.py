import os

import numpy as np

from src.utils.git_utils import get_repo_content_on_commit, get_changed_files_between_commits


def get_repo_content(dp: dict, category: str, repos_path: str) -> dict[str, str]:
    extensions = category if category != "mixed" else None
    repo_path = os.path.join(repos_path, f"{dp['repo_owner']}__{dp['repo_name']}")
    repo_content = get_repo_content_on_commit(dp[repo_path], dp['base_sha'], extensions)

    return repo_content


def get_changed_files(dp: dict, category: str, repos_path: str) -> np.ndarray[str]:
    extensions = category if category != "mixed" else None
    repo_path = os.path.join(repos_path, f"{dp['repo_owner']}__{dp['repo_name']}")
    changed_files = get_changed_files_between_commits(repo_path, dp['base_sha'], dp['head_sha'],
                                                      extensions)

    return np.asarray(changed_files, dtype=str)