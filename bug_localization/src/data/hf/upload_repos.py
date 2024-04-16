import json
import multiprocessing
import os
import shutil

import datasets
import huggingface_hub
import hydra
from huggingface_hub import HfApi
from omegaconf import DictConfig

from src.utils.hf_utils import HUGGINGFACE_REPO, CATEGORIES


def archive_repo(repo_owner: str, repo_name: str, repos_path: str, archives_path: str):
    os.chdir(archives_path)
    shutil.make_archive(
        f"{repo_owner}__{repo_name}",
        'gztar',
        root_dir=repos_path,
        base_dir=f"{repo_owner}__{repo_name}"
    )


def archive_repos(repos_list: list[tuple[str, str]], repos_path: str, archives_path: str):
    params = [(repo_owner, repo_name, repos_path, archives_path) for repo_owner, repo_name in repos_list]

    cpus = multiprocessing.cpu_count()
    assert cpus > 0

    with multiprocessing.Pool(processes=cpus) as pool:
        pool.starmap(archive_repo, params)


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def upload_bug_localization_repos(config: DictConfig):
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])
    api = HfApi()

    repos = {}
    for category in CATEGORIES:
        for split in ['test']:
            df = datasets.load_dataset(
                HUGGINGFACE_REPO, category,
                split=split,
                ignore_verifications=True,
            )

            repos[category] = list(set(zip(df['repo_owner'], df['repo_name'])))
            print(f"Find {len(repos[category])} repos in category {category}")

            archive_path = str(os.path.join(config.repos_archive_path, category))
            os.makedirs(archive_path, exist_ok=True)
            archive_repos(repos[category], config.repos_path, archive_path)

            api.upload_folder(
                folder_path=archive_path,
                repo_id=HUGGINGFACE_REPO,
                path_in_repo=f'./repos/{category}',
                repo_type="dataset"
            )

            shutil.rmtree(archive_path, ignore_errors=True)

    repos_paths = {}
    path_json_path = os.path.join(config.repos_archive_path, 'repos_paths.json')
    for category in CATEGORIES:
        repos_paths[category] = [f'./repos/{category}/{repo_owner}__{repo_name}.tar.gz'
                                 for repo_owner, repo_name in repos[category]]

    with open(path_json_path, 'w') as f:
        json.dump(repos_paths, f)

    api.upload_file(
        path_or_fileobj=path_json_path,
        repo_id=HUGGINGFACE_REPO,
        repo_type="dataset",
        path_in_repo="repos_paths.json"
    )


if __name__ == '__main__':
    upload_bug_localization_repos()
