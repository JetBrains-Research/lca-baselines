import os

import hydra
import numpy as np
from omegaconf import DictConfig

from data.hf_utils import update_hf_data
from utils.git_utils import get_changed_files_between_commits, get_repo_content_on_commit


def add_object_stats(dp, objects, object_name, prefix):
    dp[f'{prefix}_files_max_{object_name}_count'] = np.max(np.array(objects))
    dp[f'{prefix}_files_min_{object_name}_count'] = np.min(np.array(objects))
    dp[f'{prefix}_files_avg_{object_name}_count'] = np.mean(np.array(objects))
    dp[f'{prefix}_files_sum_{object_name}_count'] = np.sum(np.array(objects))


def add_lines_stats(dp, content, prefix):
    lines_count = [len(content.split('\n')) for content in content.values()]
    add_object_stats(dp, lines_count, 'lines', prefix)


def add_symbols_stats(dp, content, prefix):
    symbols_count = [len(content) for content in content.values()]
    add_object_stats(dp, symbols_count, 'symbols', prefix)


def add_statistics_to_dp(dp, data_path: str, category: str):
    repo_path = os.path.join(data_path, "repos", f"{dp['repo_owner']}__{dp['repo_name']}")
    repo_content = get_repo_content_on_commit(repo_path, dp["base_sha"])

    if category != 'mixed':
        repo_content = {file: content for file, content in repo_content.items()
                        if file.endswith(category)}

    dp['base_files_count'] = len(repo_content)
    add_lines_stats(dp, repo_content, 'base')
    add_symbols_stats(dp, repo_content, 'base')

    changed_files = get_changed_files_between_commits(repo_path, dp["base_sha"], dp["head_sha"])
    assert dp['changed_files_count'] == len(changed_files)

    for file in changed_files:
        if file not in repo_content:
            print(dp['pull_url'], dp['issue_url'], dp['diff_url'])

    changed_files_content = {file: repo_content[file] for file in changed_files}
    add_lines_stats(dp, changed_files_content, 'changed')
    add_symbols_stats(dp, changed_files_content, 'changed')


@hydra.main(config_path="./../configs", config_name="data", version_base=None)
def add_statistics_to_data(config: DictConfig):
    update_hf_data(
        lambda df, category, split:
        df.map(
            lambda dp: add_statistics_to_dp(dp, category, config.data_path),
        )
    )


if __name__ == '__main__':
    add_statistics_to_data()
