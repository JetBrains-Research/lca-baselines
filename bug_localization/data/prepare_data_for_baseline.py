import multiprocessing
import os
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig

from utils.file_utils import get_file_exts
from utils.git_utils import get_diff_between_commits, parse_changed_files_from_diff
from utils.jsonl_utils import get_jsonl_data, get_repos


def has_test_files(changed_files: List[str]) -> bool:
    for file in changed_files:
        if "/test/" in file or "test_" in file:
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
            pull = pulls_by_urls[issues_link['issue_html_url']]
            issue = issues_by_urls[issues_link['linked_issue_html_url']]
            try:
                diff = get_diff_between_commits(repo_path, pull['base']['sha'], pull['head']['sha'])
                diff.encode('utf-8')
                changed_files = parse_changed_files_from_diff(diff)
                files_exts = get_file_exts(changed_files)
            except Exception as e:
                print("Failed to get diff", e)
                continue
            records.append(
                {
                    "repo_owner": repo_owner,
                    "repo_name": repo_name,
                    "issue_url": issues_link['linked_issue_html_url'],
                    "pull_url": issues_link['issue_html_url'],
                    "comment_url": issues_link['comment_html_url'],
                    "issue_title": issue['title'],
                    "issue_body": issue['body'],
                    "base_sha": pull['base']['sha'],
                    "head_sha": pull['head']['sha'],
                    "diff_url": f"https://github.com/{repo_owner}/{repo_name}/compare/{pull['base']['sha']}...{pull['head']['sha']}",
                    "diff": diff,
                    "changed_files": changed_files,
                    "changed_files_count": len(changed_files),
                    "java_changed_files_count": files_exts.get('.java', 0),
                    "py_changed_files_count": files_exts.get('.py', 0),
                    "kt_changed_files_count": files_exts.get('.kt', 0),
                    "code_changed_files_count": sum(
                        [v for k, v in files_exts.items() if k in ['.java', '.py', '.kt']]),
                    "changed_files_exts": files_exts,
                    "pull_create_at": pull['created_at'],
                    "stars": repo['stars'],
                    "language": repo['language'],
                    "languages": repo['languages'],
                    "license": repo['license'],
                }
            )

    return records


@hydra.main(config_path="./../configs", config_name="local_data", version_base=None)
def main(config: DictConfig):
    cpus = multiprocessing.cpu_count()
    results = []
    params = [(repo, config) for repo in get_repos(config.repos_list_path)]

    with multiprocessing.Pool(processes=cpus) as pool:
        result = pool.starmap(get_repo_records, params)
        for r in result:
            results += r

    df = pd.DataFrame.from_records(results)
    df.to_csv("bug_localization_data.csv", index=False)


if __name__ == "__main__":
    main()
