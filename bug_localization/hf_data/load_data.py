import os
import subprocess
from argparse import ArgumentParser

import datasets
import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from hf_data.hf_utils import HUGGINGFACE_REPO, FEATURES, CATEGORIES, SPLITS


@hydra.main(config_path="./../configs", config_name="data_config", version_base=None)
def load_repos(config: DictConfig):
    huggingface_token = os.environ['HUGGINGFACE_TOKEN']

    # Load json file with repos paths
    paths_json = datasets.load_dataset(
        HUGGINGFACE_REPO,
        data_files=f"paths.json",
        token=huggingface_token,
        split="train",
        ignore_verifications=True,
        features=FEATURES['repos_paths']
    )

    # Load each repo in .tar.gz format, unzip, delete archive
    repos = paths_json['repos'][0]

    for i, repo_tar_path in enumerate(repos):
        print(f"Loading {i}/{len(repos)} {repo_tar_path}")

        if os.path.exists(os.path.join(config.data_path, repo_tar_path[:-7])):
            print(f"Repo {repo_tar_path} is already loaded...")
            continue

        local_repo_tars = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=repo_tar_path,
            token=huggingface_token,
            repo_type='dataset',
            local_dir=config.data_path,
        )
        # TODO: rewrite with tarfile
        result = subprocess.run(
            ["tar", "-xzf", local_repo_tars, "-C", os.path.join(config.data_path, 'repos')])
        os.remove(local_repo_tars)


@hydra.main(config_path="./../configs", config_name="data_config", version_base=None)
def load_bug_localization_data(config: DictConfig):
    huggingface_token = os.environ['HUGGINGFACE_TOKEN']

    # Load jsonl file with bug localization dataset data
    for config in CATEGORIES:
        for split in SPLITS:
            df = datasets.load_dataset(
                HUGGINGFACE_REPO, config,
                token=huggingface_token,
                split=split,
                ignore_verifications=True,
            )
            csv_path = os.path.join(config.data_path, 'data', config, split)
            os.makedirs(csv_path, exist_ok=True)
            df.to_csv(os.path.join(csv_path, "data.csv"))


if __name__ == '__main__':
    argparser = ArgumentParser()

    load_repos()
    load_bug_localization_data()
