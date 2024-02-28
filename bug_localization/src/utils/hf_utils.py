import os
from typing import Callable

import datasets

HUGGINGFACE_REPO = 'JetBrains-Research/lca-bug-localization'
CATEGORIES = ['py', 'java', 'kt', 'mixed']
SPLITS = ['dev', 'test', 'train']

FEATURES = {
    'repos_paths': datasets.Features(
        {
            category: [datasets.Value("string")] for category in CATEGORIES
        }
    ),
    'bug_localization_data': datasets.Features(
        {
            "id": datasets.Value("int64"),
            "repo_owner": datasets.Value("string"),
            "repo_name": datasets.Value("string"),
            "issue_url": datasets.Value("string"),
            "pull_url": datasets.Value("string"),
            "comment_url": datasets.Value("string"),
            "links_count": datasets.Value("int64"),
            "issue_title": datasets.Value("string"),
            "issue_body": datasets.Value("string"),
            "base_sha": datasets.Value("string"),
            "head_sha": datasets.Value("string"),
            "diff_url": datasets.Value("string"),
            "diff": datasets.Value("string"),
            "changed_files": datasets.Value("string"),
            "changed_files_exts": datasets.Value("string"),
            "changed_files_count": datasets.Value("int64"),
            "java_changed_files_count": datasets.Value("int64"),
            "kt_changed_files_count": datasets.Value("int64"),
            "py_changed_files_count": datasets.Value("int64"),
            "code_changed_files_count": datasets.Value("int64"),
            "pull_create_at": datasets.Value("string"),
            "stars": datasets.Value("int64"),
            "language": datasets.Value("string"),
            "languages": datasets.Value("string"),
            "license": datasets.Value("string"),
        }
    )
}


def update_hf_data(update: Callable[[datasets.Dataset, str, str], datasets.Dataset]) -> None:
    huggingface_token = os.environ['HUGGINGFACE_TOKEN']

    for config in CATEGORIES:
        for split in SPLITS:
            df = datasets.load_dataset(
                HUGGINGFACE_REPO, config,
                token=huggingface_token,
                split=split,
                ignore_verifications=True,
            )

            df = update(df, config, split)
            df.push_to_hub(HUGGINGFACE_REPO,
                           config,
                           private=True,
                           split=split,
                           token=huggingface_token)


def update_hf_data_splits(update: Callable[[datasets.Dataset, str, str], datasets.Dataset]) -> None:
    huggingface_token = os.environ['HUGGINGFACE_TOKEN']

    for config in CATEGORIES:
        df = datasets.load_dataset(
            HUGGINGFACE_REPO, config,
            token=huggingface_token,
            split='dev',
            ignore_verifications=True,
        )

        for split in SPLITS:
            df = update(df, config, split)
            df.push_to_hub(HUGGINGFACE_REPO,
                           config,
                           private=True,
                           split=split,
                           token=huggingface_token)
