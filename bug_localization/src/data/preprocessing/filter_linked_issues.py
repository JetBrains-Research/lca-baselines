import os
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Set, Optional

import hydra
from langdetect import detect_langs
from omegaconf import DictConfig

from src.data.preprocessing.utils import is_utf_8, has_media_in_text, remove_comments, remove_code
from src.utils.file_utils import get_file_exts, is_test_file
from src.utils.git_utils import get_repo_content_on_commit, get_diff_between_commits, parse_changed_files_from_diff, \
    parse_added_files_from_diff
from src.utils.jsonl_utils import get_jsonl_data, save_jsonl_data
from src.utils.processing_utils import process_repos_data


def url_to_id(url: str) -> int:
    return int(url.split('/')[-1])


class FilterStatus(Enum):
    OK = "ok"

    NOT_ENOUGH_INFO = "not_enough_info"

    ISSUE_NOT_A_BUG = "issue_not_a_bug"
    ISSUE_EMPTY = "issue_empty"
    ISSUE_NON_UTF8 = "issue_non_utf8"
    ISSUE_HAS_MEDIA = "issue_has_media"
    ISSUE_NOT_ENGLISH = "issue_not_english"

    DIFF_CAN_NOT_EXTRACT = "diff_can_not_extract"
    DIFF_CAN_NOT_EXTRACT_CHANGED_FILES = "diff_can_not_extract_changed_files"
    DIFF_NON_CODE_FILES = "diff_non_code_files"
    DIFF_NON_UTF8 = "diff_non_utf8"
    DIFF_CAN_NOT_EXTRACT_BASE_COMMIT = "diff_can_not_extract_base_commit"
    DIFF_HAS_NEW_FILES = "diff_has_new_files"

    PR_TO_MULTI_ISSUES = "pr_to_multi_issues"
    ISSUE_TO_MULTI_PRS = "issue_to_multi_prs"
    NO_FIX_KEYWORD = "no_fix_keyword"


def has_info_about_issues(issue_id: int, linked_issue_id: int, issue_links: dict) -> FilterStatus:
    # Check there is info about issue and linked issue
    if issue_id not in issue_links or linked_issue_id not in issue_links[issue_id]:
        return FilterStatus.NOT_ENOUGH_INFO
    return FilterStatus.OK


def apply_issue_filters(linked_issue: dict) -> FilterStatus:
    # Check issue is a bug
    if not any(["bug" in label['name'].lower() for label in linked_issue['labels']]):
        print(f"Issue is not a bug. "
              f"Skipping linked issue {linked_issue['html_url']} ...")
        return FilterStatus.ISSUE_NOT_A_BUG

    # Check issue body is not empty
    if linked_issue['body'] == '' or linked_issue['body'] is None:
        print(f"Issue body is empty. "
              f"Skipping linked issue {linked_issue['html_url']} ...")
        return FilterStatus.ISSUE_EMPTY

    # Check issue body is utf8
    if not is_utf_8(linked_issue['body']):
        print(f"Issue body contains non-utf8 symbols. "
              f"Skipping linked issue {linked_issue['html_url']} ...")
        return FilterStatus.ISSUE_NON_UTF8

    # Check issue body is utf8
    if has_media_in_text(linked_issue['body']):
        print(f"Issue body contains media. "
              f"Skipping linked issue {linked_issue['html_url']} ...")
        return FilterStatus.ISSUE_HAS_MEDIA

    # Check issue language is english
    issue_body = linked_issue['body']
    issue_body = remove_comments(issue_body)
    issue_body = remove_code(issue_body)
    try:
        issue_languages = detect_langs(issue_body)
        if len(issue_languages) > 1 or issue_languages[0].lang != 'en':
            print(f"Issue written not in english {issue_languages}. "
                  f"Skipping linked issue {linked_issue['html_url']} ...")
            return FilterStatus.ISSUE_NOT_ENGLISH
    except Exception as e:
        print(f"Failed to get issue language {e}. "
              f"Skipping linked issue {linked_issue['html_url']} ...")
        return FilterStatus.ISSUE_NOT_ENGLISH

    return FilterStatus.OK


def apply_diff_filters(repo_path: str, pull_request: dict) -> FilterStatus:
    # Check diff between base and head commit can be extracted
    try:
        diff = get_diff_between_commits(repo_path, pull_request['base']['sha'], pull_request['head']['sha'])
    except Exception as e:
        print(f"Can not get diff {e}. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_CAN_NOT_EXTRACT

    if diff is None or diff == "":
        print(f"Diff is empty or None. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_CAN_NOT_EXTRACT

    # Check diff between base and head commit can be extracted
    try:
        changed_files = [f for f in parse_changed_files_from_diff(diff) if not is_test_file(f)]
        added_files = [f for f in parse_added_files_from_diff(diff) if not is_test_file(f)]

        if len(changed_files) < 0:
            print(f"No changed files found. "
                  f"Skipping pull request {pull_request['html_url']} ...")
            return FilterStatus.DIFF_CAN_NOT_EXTRACT
    except Exception as e:
        print(f"Can not get changed files {e}. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_CAN_NOT_EXTRACT

    # Get only diffs without new files (new tests are excepted)
    if len(added_files) > 0:
        print(f"Diff contains new files. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_HAS_NEW_FILES

    # Keep only diff with python, java, kotlin files
    changed_files_exts = get_file_exts(changed_files)
    if not any(key in [".py", ".java", ".kt"] for key in changed_files_exts.keys()):
        print(f"Diff contains no py|java|kt files. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_NON_CODE_FILES

    # Check issue body is utf8
    if not is_utf_8(diff):
        print(f"Diff contains non utf-8 symbols. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_NON_UTF8

    # Check repo content on pull base commit can be extracted
    try:
        repo_content = get_repo_content_on_commit(repo_path, pull_request['base']['sha'])
    except Exception as e:
        print(f"Failed to get repo content {e}. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_CAN_NOT_EXTRACT_CHANGED_FILES

    # Can read all files in diff
    if any(changed_file not in repo_content or repo_content[changed_file] is None for changed_file in changed_files):
        print(f"Failed to get all files from diff. "
              f"Skipping pull request {pull_request['html_url']} ...")
        return FilterStatus.DIFF_CAN_NOT_EXTRACT_CHANGED_FILES

    return FilterStatus.OK


def apply_issue_links_filter(parsed_issue_link: dict, pull_id: int, pull_links: set, linked_issue_id: int,
                             issue_links: set) \
        -> FilterStatus:
    # If more than one issue to pull request -- skip as it probably contains changes from several issues
    if len(pull_links) > 1 or (len(pull_links) == 1 and linked_issue_id not in pull_links):
        print(f"Pull request connected to multiple issues")
        return FilterStatus.PR_TO_MULTI_ISSUES

    # If more than one pull request to one issue -- skip as it probably fixed in several pull requests
    if len(issue_links) > 1 or (len(issue_links) == 1 and pull_id not in issue_links):
        print(f"Issue connected to multiple pull requests")
        return FilterStatus.ISSUE_TO_MULTI_PRS

    if parsed_issue_link['link_keyword'] == "":
        return FilterStatus.NO_FIX_KEYWORD

    return FilterStatus.OK


def filter_linked_issue(parsed_issue_link: dict,
                        issue_links: dict[int, set[int]],
                        pulls_by_id: dict[int, dict],
                        issues_by_id: dict[int, dict],
                        repo_path: str) -> tuple[FilterStatus, Optional[int], Optional[int]]:
    issue_url = parsed_issue_link['issue_html_url']
    linked_issue_url = parsed_issue_link["linked_issue_html_url"]
    issue_id = url_to_id(issue_url)
    linked_issue_id = url_to_id(linked_issue_url)

    # Apply info filter
    status = has_info_about_issues(issue_id, linked_issue_id, issue_links)
    if status != FilterStatus.OK:
        return status, None, None

    if issue_id in pulls_by_id:
        pull_id, linked_issue_id = issue_id, linked_issue_id
    else:
        pull_id, linked_issue_id = linked_issue_id, issue_id

    pull_request = pulls_by_id[pull_id]
    linked_issue = issues_by_id[linked_issue_id]

    # Apply issue filter
    status = apply_issue_filters(linked_issue)
    if status != FilterStatus.OK:
        return status, pull_id, linked_issue_id

    # Apply diff filter
    status = apply_diff_filters(repo_path, pull_request)
    if status != FilterStatus.OK:
        return status, pull_id, linked_issue_id

    # Apply links filter
    status = apply_issue_links_filter(parsed_issue_link,
                                      pull_id, issue_links.get(pull_id, set()),
                                      linked_issue_id, issue_links.get(linked_issue_id, set()))
    if status != FilterStatus.OK:
        return status, pull_id, linked_issue_id

    return FilterStatus.OK, pull_id, linked_issue_id


def filter_linked_issues(
        parsed_issues_links: List[dict], pulls: List[dict], issues: List[dict], repo_path: str
) -> List[dict]:
    pulls_by_id = {url_to_id(pull['html_url']): pull for pull in pulls}
    issues_by_id = {url_to_id(issue['html_url']): issue for issue in issues if
                    url_to_id(issue['html_url']) not in pulls_by_id}

    # Pull to issue relation without duplications
    issue_links: Dict[int, Set[int]] = defaultdict(set)

    for parsed_issue_link in parsed_issues_links:
        issue_id = url_to_id(parsed_issue_link['issue_html_url'])
        linked_issue_id = url_to_id(parsed_issue_link['linked_issue_html_url'])
        if (issue_id in pulls_by_id and linked_issue_id in issues_by_id) or (
                issue_id in issues_by_id and linked_issue_id in pulls_by_id):
            print(f"Link {issue_id} <-> {linked_issue_id}")
            issue_links[issue_id].add(linked_issue_id)

    filtered_parsed_issue_links: list[dict] = []

    for parsed_issue_link in parsed_issues_links:
        status, pull_id, linked_issue_id = (
            filter_linked_issue(parsed_issue_link, issue_links, pulls_by_id, issues_by_id, repo_path))

        if linked_issue_id in issue_links.get(pull_id, set()) and pull_id in issue_links.get(linked_issue_id, set()):
            links_count = 2
        else:
            links_count = 1

        filtered_parsed_issue_links.append({
            "comment_html_url": parsed_issue_link['comment_html_url'],
            "issue_html_url": pulls_by_id[pull_id]['html_url'] if pull_id else parsed_issue_link['issue_html_url'],
            "linked_issue_html_url": issues_by_id[linked_issue_id]['html_url'] if linked_issue_id else
            parsed_issue_link['linked_issue_html_url'],
            "link_type": parsed_issue_link['link_type'],
            "link_keyword": parsed_issue_link['link_keyword'],
            "links_count": links_count,
            "status": status.value
        })
    print(f"Left issues links: {len(filtered_parsed_issue_links)}")
    return list(filtered_parsed_issue_links)


def prepare_data(repo: dict, config: DictConfig):
    repo_owner = repo['owner']
    repo_name = repo['name']
    print(f"Processing repo {repo_owner}/{repo_name}...")

    if os.path.exists(os.path.join(config.issues_links_filtered_path, f"{repo_owner}__{repo_name}.jsonl")):
        print(f"Repo {repo_owner}/{repo_name} already processed")
        return
    repo_path = os.path.join(config.repos_path, f"{repo_owner}__{repo_name}")

    pulls = get_jsonl_data(config.pulls_path, repo_owner, repo_name)
    issues = get_jsonl_data(config.issues_path, repo_owner, repo_name)
    parsed_issues_links = get_jsonl_data(config.issues_links_path, repo_owner, repo_name)

    if pulls is None or issues is None or parsed_issues_links is None:
        print(f"Not enough info for repo {repo_owner}/{repo_name}. Skipping...")
        return

    filtered_parsed_issue_links = filter_linked_issues(parsed_issues_links, pulls, issues, repo_path)

    save_jsonl_data(
        repo_owner, repo_name,
        filtered_parsed_issue_links,
        config.issues_links_filtered_path,
    )


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def main(config: DictConfig):
    os.makedirs(config.issues_links_filtered_path, exist_ok=True)
    process_repos_data(prepare_data, config)


if __name__ == "__main__":
    main()
