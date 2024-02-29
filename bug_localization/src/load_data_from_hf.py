import os
import shutil

import datasets
import huggingface_hub
import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from src.utils.hf_utils import HUGGINGFACE_REPO, FEATURES, CATEGORIES


def load_repos(repos_path: str):
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])

    # Load json file with repos paths
    paths_json = datasets.load_dataset(
        HUGGINGFACE_REPO,
        data_files=f"repos_paths.json",
        ignore_verifications=True,
        split="train",
        features=FEATURES['repos_paths']
    )

    local_repo_tars_path = os.path.join(repos_path, "local_repos_tars")
    # Load each repo in .tar.gz format, unzip, delete archive
    for category in CATEGORIES:
        repos = paths_json[category][0]
        for i, repo_tar_path in enumerate(repos):
            print(f"Loading {i}/{len(repos)} {repo_tar_path}")

            repo_name = os.path.basename(repo_tar_path)
            if os.path.exists(os.path.join(repos_path, repo_name)):
                print(f"Repo {repo_tar_path} is already loaded...")
                continue

            local_repo_tar_path = hf_hub_download(
                HUGGINGFACE_REPO,
                filename=repo_tar_path,
                repo_type='dataset',
                local_dir=local_repo_tars_path,
            )
            shutil.unpack_archive(local_repo_tar_path, extract_dir=repos_path, format='gztar')
            os.remove(local_repo_tar_path)
    shutil.rmtree(local_repo_tars_path)


@hydra.main(config_path="../configs/data", config_name="local", version_base=None)
def load_dataset(config: DictConfig) -> None:
    load_repos(config.repos_path)


if __name__ == '__main__':
    load_dataset()
