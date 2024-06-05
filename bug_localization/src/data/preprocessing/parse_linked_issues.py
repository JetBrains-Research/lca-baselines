import os
import re
from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig

from src.data.preprocessing.utils import remove_comments, remove_code
from src.utils.jsonl_utils import get_jsonl_data, save_jsonl_data
from src.utils.processing_utils import process_repos_data

KEYWORDS = {
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
    "solve",
    "solves",
    "solved",
}


def parse_linked_issues_from_comment(comment_text: str) -> List[Tuple[int, str, str]]:
    """
    Parse issue links from comments text according to documentation
    https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls
    :param comment_text: text of issue comment to parse linked issues from
    :return: list of issue id and link type pairs parsed from comments text
    """

    patterns = {
        # https://github.com/jlord/sheetsee.js/issues/26
        "issue_link": r"(\w+\s)?https:\/\/github\.com\/[^\/\s]+\/[^\/\s]+\/issues\/(?P<issue_number>\d+)(\s\w+)?",
        # #26
        "hash": r"(\w+\s)?#(?P<issue_number>\d+)(\s\w+)?",
        # GH-26
        "slash": r"(\w+\s)?gh\-(?P<issue_number>\d+)(\s\w+)?",
        # jlord/sheetsee.js#26
        "file": r"(\w+\s)?[^\/\s]+\/[^\/\s]+#(?P<issue_number>\d+)(\s\w+)?",
    }

    comment_text = remove_comments(comment_text)
    comment_text = remove_code(comment_text)

    linked_issues = []
    for p_type, p in patterns.items():
        try:
            issue_ids = re.findall(p, comment_text.lower())
        except Exception as e:
            print(f"Can not parse issue links from text:\n{comment_text}", e)
            continue
        for keyword_before, issue_id, keyword_after in issue_ids:
            if keyword_before.strip() in KEYWORDS:
                keyword = keyword_before
            elif keyword_after.strip() in KEYWORDS:
                keyword = keyword_after
            else:
                keyword = ""
            if not issue_id.isdigit():
                continue
            linked_issues.append((int(issue_id), keyword.strip(), p_type))

    return linked_issues


def parse_linked_issues_from_comments(
        repo_owner: str,
        repo_name: str,
        issues_comments_path: str,
        pull_requests_comments_path: str,
        pull_requests_path: str,
) -> list[dict]:
    issues_links = []
    comments = []

    issues_comments = get_jsonl_data(issues_comments_path, repo_owner, repo_name)
    if issues_comments is None:
        print(f"Issues comments are missed for repo {repo_owner}/{repo_name}")
    else:
        comments += issues_comments

    pull_requests_comments = get_jsonl_data(pull_requests_comments_path, repo_owner, repo_name)
    if pull_requests_comments is None:
        print(f"Pull requests comments are missed for repo {repo_owner}/{repo_name}")
    else:
        comments += pull_requests_comments

    pull_requests = get_jsonl_data(pull_requests_path, repo_owner, repo_name)
    if pull_requests is None:
        print(f"Pull requests are missed for repo {repo_owner}/{repo_name}")
    else:
        comments += pull_requests

    for comment in comments:
        if comment['body'] is None:
            print(f"Comment {comment['html_url']} body is None. Skipping...")
            continue

        comment_text = comment['body']
        if 'title' in comment:
            # Add pull request title to comment text
            comment_text = comment['title'] + '\n' + comment_text

        parsed_issue_links = parse_linked_issues_from_comment(comment_text)
        comment_html_url = comment['html_url']
        for issue_id, link_keyword, link_type in parsed_issue_links:
            issues_links.append(
                {
                    # https://github.com/umple/umple/issues/733#issuecomment-185940279
                    "comment_html_url": comment_html_url,
                    # https://github.com/umple/umple/issues/733
                    "issue_html_url": comment_html_url.split("#")[0],
                    # Issue as same rule for issues and pull linking, will de defined in processing stage
                    "linked_issue_html_url": f"https://github.com/{repo_owner}/{repo_name}/issues/{issue_id}",
                    # issue_link|hash|slash|file
                    "link_type": link_type,
                    # keyword
                    "link_keyword": link_keyword,
                }
            )

    return issues_links


def get_linked_issues_from_comments(
        repo: dict,
        config: DictConfig
) -> Optional[Exception]:
    repo_owner = repo['owner']
    repo_name = repo['name']
    print(f"Processing repo {repo_owner}/{repo_name}...")

    repo_linked_issues_path = os.path.join(config.issues_links_path, f"{repo_owner}__{repo_name}.jsonl")
    if os.path.exists(repo_linked_issues_path):
        print(f"Linked issues for repo {repo_owner}/{repo_name} already parsed. Skipping...")
        return None

    issues_links = parse_linked_issues_from_comments(repo_owner, repo_name,
                                                     config.issues_comments_path,
                                                     config.pull_requests_comments_path,
                                                     config.pulls_path)
    print(f"Collected {len(issues_links)} issue links")
    save_jsonl_data(repo_owner, repo_name, issues_links, config.issues_links_path)

    return None


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def main(config: DictConfig):
    os.makedirs(config.issues_links_path, exist_ok=True)

    process_repos_data(get_linked_issues_from_comments, config)


if __name__ == "__main__":
    main()
