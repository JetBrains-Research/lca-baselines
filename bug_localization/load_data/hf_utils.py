import datasets

HUGGINGFACE_REPO = 'JetBrains-Research/lca-bug-localization'

FEATURES = {
    'repos_paths': datasets.Features(
        {
            "repos": [datasets.Value("string")]
        }
    ),
    'bug_localization_data': datasets.Features(
        {
            "repo_owner": datasets.Value("string"),
            "repo_name": datasets.Value("string"),
            "issue_url": datasets.Value("string"),
            "pull_url": datasets.Value("string"),
            "comment_url": datasets.Value("string"),
            "issue_title": datasets.Value("string"),
            "issue_body": datasets.Value("string"),
            "base_sha": datasets.Value("string"),
            "head_sha": datasets.Value("string"),
            "diff_url": datasets.Value("string"),
            "changed_files": datasets.Value("string"),
            "changed_files_exts": datasets.Value("string"),
            "java_changed_files_count": datasets.Value("int64"),
            "kt_changed_files_count": datasets.Value("int64"),
            "py_changed_files_count": datasets.Value("int64"),
            "code_changed_files_count": datasets.Value("int64"),
            "pull_create_at": datasets.Value("string"),
            "stars": datasets.Value("int64")
        }
    )
}
