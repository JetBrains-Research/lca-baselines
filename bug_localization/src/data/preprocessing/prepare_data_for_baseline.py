import multiprocessing
import os
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig
from langdetect import detect

from src.utils.file_utils import get_file_exts
from src.utils.git_utils import get_diff_between_commits, parse_changed_files_from_diff
from src.utils.jsonl_utils import get_jsonl_data, get_repos


def has_test_files(changed_files: List[str]) -> bool:
    for file in changed_files:
        if "/test" in file.lower() or "test_" in file.lower():
            return True
    return False


def get_repo_records(repo: dict, config: DictConfig) -> List[dict]:
    repo_owner = repo['owner']
    repo_name = repo['name']
    issues_links = get_jsonl_data(config.issues_links_filtered_path, repo_owner, repo_name)
    if issues_links is None or len(issues_links) == 0:
        return []
    pulls = get_jsonl_data(config.pulls_path, repo_owner, repo_name)
    if pulls is None or pulls is None:
        print(f"Can not get pulls for repo {repo_owner}/{repo_name}")
        return []

    issues = get_jsonl_data(config.issues_path, repo_owner, repo_name)
    if pulls is None or issues is None:
        print(f"Can not get issues for repo {repo_owner}/{repo_name}")
        return []

    pulls_by_urls = {pull['html_url']: pull for pull in pulls}
    issues_by_urls = {issue['html_url']: issue for issue in issues}

    repo_path = os.path.join(config.repos_path, f"{repo_owner}__{repo_name}")

    records = []
    if issues_links is not None:
        for issues_link in issues_links:
            try:
                pull = pulls_by_urls[issues_link['issue_html_url']]
                issue = issues_by_urls[issues_link['linked_issue_html_url']]
                diff = get_diff_between_commits(repo_path, pull['base']['sha'], pull['head']['sha'])
                changed_files = parse_changed_files_from_diff(diff)
                files_exts = get_file_exts(changed_files)
            except Exception as e:
                print("Failed to get data", e)
                continue
            records.append(
                {
                    "text_id": f"{repo_owner}/{repo_name}/"
                               f"{issues_link['issue_html_url'].split('/')[-1]}/"
                               f"{issues_link['linked_issue_html_url'].split('/')[-1]}",
                    "repo_owner": repo_owner,
                    "repo_name": repo_name,
                    "issue_url": issues_link['linked_issue_html_url'],
                    "pull_url": issues_link['issue_html_url'],
                    "comment_url": issues_link['comment_html_url'],
                    "links_count": issues_link['links_count'],
                    "issue_title": str(issue['title']),
                    "issue_body": str(issue['body']),
                    "issue_body_langauge": str(detect(issue['body'])),
                    "base_sha": pull['base']['sha'],
                    "head_sha": pull['head']['sha'],
                    "diff_url": f"https://github.com/{repo_owner}/{repo_name}/compare/{pull['base']['sha']}...{pull['head']['sha']}",
                    "diff": str(diff),
                    "changed_files": str(changed_files),
                    "changed_files_count": len(changed_files),
                    "java_changed_files_count": files_exts.get('.java', 0),
                    "py_changed_files_count": files_exts.get('.py', 0),
                    "kt_changed_files_count": files_exts.get('.kt', 0),
                    "code_changed_files_count": sum(
                        [v for k, v in files_exts.items() if k in ['.java', '.py', '.kt']]),
                    "changed_files_exts": str(files_exts),
                    "pull_create_at": pull['created_at'],
                    "stars": repo['stars'],
                    "language": str(repo['language']),
                    "languages": str(repo['languages']),
                    "license": str(repo['license']),
                }
            )

    return records


def split_by_language(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    code_changed_files = ['py_changed_files_count', 'kt_changed_files_count', 'java_changed_files_count']
    df_by_language = {}

    print(f"Total samples: {df.shape[0]}")

    for lang_count_column in code_changed_files:
        df_lang = df[df[lang_count_column] == df['changed_files_count']]
        lang = lang_count_column.split('_')[0]
        print(f"There is {df_lang.shape[0]} {lang} samples in dataset")
        df_by_language[lang] = df_lang

    df_lang = df[~(df[code_changed_files].eq(df['changed_files_count'], axis=0)).any(axis=1)]
    print(f"There is {df_lang.shape[0]} mixed code or text samples in dataset")
    df_by_language['mixed'] = df_lang
    return df_by_language


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def main(config: DictConfig):
    cpus = multiprocessing.cpu_count()
    results = []
    params = [(repo, config) for repo in get_repos(config.repos_list_path)]

    with multiprocessing.Pool(processes=cpus) as pool:
        result = pool.starmap(get_repo_records, params)
        for r in result:
            results += r

    df = pd.DataFrame.from_records(results)
    df = df.sort_values('stars', ascending=False)
    df['id'] = df.index
    df_by_language = split_by_language(df)

    os.makedirs(config.bug_localization_data_path, exist_ok=True)
    for lang, df_lang in df_by_language.items():
        df_lang.to_csv(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{lang}.csv"),
                       escapechar="\\", index=False)
        df_lang.to_json(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{lang}.jsonl"),
                        orient="records", lines=True)


if __name__ == "__main__":
    main()
