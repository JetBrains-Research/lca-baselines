import os

import hydra
from omegaconf import DictConfig

from hf_data.hf_utils import update_hf_data
from file_utlis.git_utils import get_changed_files_between_commits, get_repo_content_on_commit


def filter_can_extract_change(dp, data_path: str):
    print(f"Processing dp {dp['id']}")
    repo_path = os.path.join(data_path, "repos", f"{dp['repo_owner']}__{dp['repo_name']}")

    try:
        repo_content = get_repo_content_on_commit(repo_path, dp["base_sha"])
        changed_files = get_changed_files_between_commits(repo_path, dp["base_sha"], dp["head_sha"])
    except Exception as e:
        print(e)
        return False

    if dp['changed_files_count'] != len(changed_files):
        print("Wrong number of changed files")
        return False

    for file in changed_files:
        if file not in repo_content:
            print(f"No file {file} in diff", dp['pull_url'], dp['issue_url'], dp['diff_url'])
            return False

    return True


@hydra.main(config_path="./../configs", config_name="data", version_base=None)
def filter_data(config: DictConfig):
    update_hf_data(
        lambda df, category, split: df.filter(lambda dp: filter_can_extract_change(dp, config.data_path)),
    )


if __name__ == '__main__':
    filter_data()
