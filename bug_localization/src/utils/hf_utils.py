import os
from typing import Callable

import datasets
import huggingface_hub
from datasets import Dataset

HUGGINGFACE_REPO = 'tiginamaria/bug-localization'
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
            "text_id": datasets.Value("string"),
            "repo_owner": datasets.Value("string"),
            "repo_name": datasets.Value("string"),
            "issue_url": datasets.Value("string"),
            "pull_url": datasets.Value("string"),
            "comment_url": datasets.Value("string"),
            "links_count": datasets.Value("int64"),
            'link_keyword': datasets.Value("string"),
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
            'repo_symbols_count': datasets.Value("int64"),
            'repo_tokens_count': datasets.Value("int64"),
            'repo_lines_count': datasets.Value("int64"),
            'repo_files_without_tests_count': datasets.Value("int64"),
            'changed_symbols_count': datasets.Value("int64"),
            'changed_tokens_count': datasets.Value("int64"),
            'changed_lines_count': datasets.Value("int64"),
            'changed_files_without_tests_count': datasets.Value("int64"),
            'issue_symbols_count': datasets.Value("int64"),
            'issue_tokens_count': datasets.Value("int64"),
            'issue_lines_count': datasets.Value("int64"),
            'issue_links_count': datasets.Value("int64"),
            'issue_code_blocks_count': datasets.Value("int64"),
            "pull_create_at": datasets.Value("timestamp[s]"),
            "stars": datasets.Value("int64"),
            "language": datasets.Value("string"),
            "languages": datasets.Value("string"),
            "license": datasets.Value("string"),
        }
    )
}


def load_data(category: str, split: str) -> Dataset:
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])

    return datasets.load_dataset(
        HUGGINGFACE_REPO, category,
        split=split,
        ignore_verifications=True,
    )


def upload_data(df: Dataset, category: str, split: str) -> None:
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])

    df.push_to_hub(HUGGINGFACE_REPO,
                   category,
                   split=split)


def update_hf_data(update: Callable[[datasets.Dataset, str, str], datasets.Dataset]) -> None:
    for category in CATEGORIES:
        for split in SPLITS:
            df = load_data(category, split)
            df = update(df, category, split)
            upload_data(df, category, split)


def update_hf_data_splits(update: Callable[[datasets.Dataset, str, str], datasets.Dataset]) -> None:
    for category in CATEGORIES:
        df = load_data(category, 'dev')
        for split in SPLITS:
            df = update(df, category, split)
            upload_data(df, category, split)
