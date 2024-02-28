import os
import subprocess

import datasets
import huggingface_hub
import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from src.utils.hf_utils import HUGGINGFACE_REPO, FEATURES, CATEGORIES


def load_repos(data_path: str):
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])

    # Load json file with repos paths
    paths_json = datasets.load_dataset(
        HUGGINGFACE_REPO,
        data_files=f"paths.json",
        ignore_verifications=True,
        features=FEATURES['repos_paths']
    )

    # Load each repo in .tar.gz format, unzip, delete archive
    for category in CATEGORIES:
        repos = paths_json['category']
        for i, repo_tar_path in enumerate(repos):
            print(f"Loading {i}/{len(repos)} {repo_tar_path}")

            if os.path.exists(os.path.join(data_path, repo_tar_path[:-7])):
                print(f"Repo {repo_tar_path} is already loaded...")
                continue

            local_repo_tars = hf_hub_download(
                HUGGINGFACE_REPO,
                filename=repo_tar_path,
                repo_type='dataset',
                local_dir=data_path,
            )
            # TODO: rewrite with tarfile
            result = subprocess.run(
                ["tar", "-xzf", local_repo_tars, "-C", os.path.join(data_path, 'repos')])
            os.remove(local_repo_tars)


@hydra.main(config_path="../configs", config_name="local", version_base=None)
def load_dataset(config: DictConfig) -> None:
    load_repos(config.data_path)


if __name__ == '__main__':
    load_dataset()
