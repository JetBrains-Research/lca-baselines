import os
import subprocess
from argparse import ArgumentParser

import datasets
from huggingface_hub import hf_hub_download

from load_data.hf_utils import HUGGINGFACE_REPO, FEATURES


def load_repos(data_path: str, cache_path: str):
    huggingface_token = os.environ['HUGGINGFACE_TOKEN']

    # Load json file with repos paths
    paths_json = datasets.load_dataset(
        HUGGINGFACE_REPO,
        data_files=f"paths.json",
        token=huggingface_token,
        split="train",
        cache_dir=cache_path,
        ignore_verifications=True,
        features=FEATURES['repos_paths']
    )

    # Load each repo in .tar.gz format, unzip, delete archive
    repos = paths_json['repos'][0]

    for i, repo_tar_path in enumerate(repos):
        print(f"Loading {i}/{len(repos)} {repo_tar_path}")

        if os.path.exists(os.path.join(data_path, repo_tar_path[:-7])):
            print(f"Repo {repo_tar_path} is already loaded...")
            continue

        local_repo_tars = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=repo_tar_path,
            token=huggingface_token,
            repo_type='dataset',
            local_dir=data_path,
            cache_dir=cache_path,
        )
        # TODO: rewrite with tarfile
        result = subprocess.run(
            ["tar", "-xzf", local_repo_tars, "-C", os.path.join(data_path, 'repos')])
        os.remove(local_repo_tars)


def load_bug_localization_data(data_path: str, cache_path: str):
    huggingface_token = os.environ['HUGGINGFACE_TOKEN']

    # Load jsonl file with bug localization dataset data
    bug_localization_data = datasets.load_dataset(
        HUGGINGFACE_REPO,
        token=huggingface_token,
        split="train",
        cache_dir=cache_path,
        ignore_verifications=True,
    )
    bug_localization_data.to_json(os.path.join(data_path, 'bug_localization_data.jsonl'))


if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument(
        "--data-path",
        type=str,
        help="Path to directory where to save data loaded from hugging face.",
        default="./../../data/lca-bug-localization"
    )

    argparser.add_argument(
        "--hf-cache-path",
        type=str,
        help="Path to directory where to cache data loaded from hugging face.",
        default="./../../datasets/lca-bug-localization"
    )

    args = argparser.parse_args()

    # load_repos(args.data_path, args.hf_cache_path)
    load_bug_localization_data(args.data_path, args.hf_cache_path)
