import multiprocessing
import os
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig

from lca.model.model import ParsedLinkedIssue, PullRequest, RepoShort, Issue
from lca.utils.ext_utils import get_file_exts, supported_code_extensions
from lca.utils.git_utils import get_diff_between_commits, \
    parse_changed_files_from_diff
from lca.utils.process_data_utils import get_repo_data, get_repos


def has_test_files(changed_files: List[str]) -> bool:
    for file in changed_files:
        if "/test/" in file or "test_" in file:
            return True
    return False


def get_repo_records(repo: RepoShort, config: DictConfig) -> List[dict]:
    issues_links: List[ParsedLinkedIssue] = get_repo_data(config.issues_links_filtered_path, repo, ParsedLinkedIssue)
    if issues_links is None or len(issues_links) == 0:
        return []
    pulls = get_repo_data(config.pulls_path, repo, PullRequest)
    if pulls is None or pulls is None:
        print(f"Can not get pulls for repo {repo.get_full_name()}")
        return []

    issues = get_repo_data(config.issues_path, repo, Issue)
    if pulls is None or issues is None:
        print(f"Can not get issues for repo {repo.get_full_name()}")
        return []

    pulls_by_urls = {pull.html_url: pull for pull in pulls}
    issues_by_urls = {issue.html_url: issue for issue in issues}

    repo_path = os.path.join(config.repos_path, f"{repo.get_dirname()}")

    records = []
    if issues_links is not None:
        for issues_link in issues_links:
            pull: PullRequest = pulls_by_urls[issues_link.issue_html_url]
            issue: Issue = issues_by_urls[issues_link.linked_issue_html_url]
            try:
                diff = get_diff_between_commits(repo_path, pull.base.sha, pull.head.sha)
                diff.encode('utf-8')
                changed_files = parse_changed_files_from_diff(diff)
                files_exts = get_file_exts(changed_files)
            except Exception as e:
                print("Failed to get diff", e)
                continue
            records.append(
                {
                    "repo_owner": repo.owner,
                    "repo_name": repo.name,
                    "issue_url": issues_link.linked_issue_html_url,
                    "pull_url": issues_link.issue_html_url,
                    "comment_url": issues_link.comment_html_url,
                    "issue_title": issue.title,
                    "issue_body": issue.body,
                    "base_sha": pull.base.sha,
                    "head_sha": pull.head.sha,
                    "diff_url": f"https://github.com/jina-ai/jina/compare/{pull.base.sha}...{pull.head.sha}",
                    "diff": diff,
                    "changed_files": changed_files,
                    "changed_files_count": len(changed_files),
                    "java_changed_files_count": files_exts.get('.java', 0),
                    "kt_changed_files_count": files_exts.get('.kt', 0),
                    "py_changed_files_count": files_exts.get('.py', 0),
                    "code_changed_files_count": sum(
                        [v for k, v in files_exts.items() if k in supported_code_extensions]),
                    "changed_files_exts": files_exts,
                    "pull_create_at": pull.created_at,
                    "stars": repo.stars
                }
            )

    return records


@hydra.main(config_path="../../../lca/configs", config_name="server", version_base=None)
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
