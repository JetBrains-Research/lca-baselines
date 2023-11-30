import os
import subprocess

import datasets
from huggingface_hub import hf_hub_download

HUGGINGFACE_REPO = 'JetBrains-Research/lca-bug-localization'


def load_repos():
    huggingface_token = os.environ['HUGGINGFACE_TOKEN']
    paths_json = datasets.load_dataset(
        HUGGINGFACE_REPO,
        data_files=f"paths.json",
        token=huggingface_token,
        split="train",
        cache_dir=f"datasets/lca-bug-localization",
        ignore_verifications=True,
        features=datasets.Features({"repos": [datasets.Value("string")]})
    )

    for repo_tar_path in paths_json['repos'][0][:2]:
        local_repo_tars = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=repo_tar_path,
            token=huggingface_token,
            repo_type='dataset',
            local_dir="./data/",
            cache_dir=f"datasets/lca-bug-localization",
        )
        # TODO: rewrite with tarfile
        result = subprocess.run(["tar", "-xzvf", local_repo_tars, "-C", './data/repos'])
        os.remove(local_repo_tars)


def main():
    load_repos()


if __name__ == '__main__':
    main()
