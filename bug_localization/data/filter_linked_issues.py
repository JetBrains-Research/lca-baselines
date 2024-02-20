import os
import re
from collections import defaultdict
from typing import Dict, List, Set

import hydra
from omegaconf import DictConfig

from utils.file_utils import get_file_exts
from utils.git_utils import get_repo_content_on_commit, get_changed_files_between_commits
from utils.jsonl_utils import get_jsonl_data, save_jsonl_data
from utils.processing_utils import process_repos_data


def url_to_id(url: str) -> int:
    return int(url.split('/')[-1])


def has_bug_label(issue: dict) -> bool:
    return any(["bug" in label['name'].lower() for label in issue['labels']])


def has_image_in_text(issue: dict) -> bool:
    try:
        images = re.findall(r"!\[.*?\]\((.*?\.(jpg|png|gif|jpeg|svg|bmp|tiff|webp|heic|psd|raw))\)", issue['body'],
                            re.I)
        return len(images) > 0
    except Exception as e:
        print("Can not parse images from text", e)
        return False


def filter_linked_issues(
        parsed_issues_links: List[dict], pulls: List[dict], issues: List[dict], repo_path: str
) -> List[dict]:
    pulls_by_id = {url_to_id(pull['html_url']): pull for pull in pulls}
    issues_by_id = {url_to_id(issue['html_url']): issue for issue in issues if issue['html_url'] not in pulls_by_id}

    # Pull to issue relation without duplications
    issues_to_linked_pulls: Dict[int, Set[dict]] = defaultdict(set)
    pulls_to_linked_issues: Dict[int, Set[dict]] = defaultdict(set)

    for parsed_issue_link in parsed_issues_links:
        issue_id = url_to_id(parsed_issue_link['issue_html_url'])
        linked_issue_id = url_to_id(parsed_issue_link['linked_issue_html_url'])
        if issue_id in pulls_by_id and linked_issue_id in issues_by_id:
            pulls_to_linked_issues[issue_id].add(parsed_issue_link)
        elif issue_id in issues_by_id and linked_issue_id in pulls_by_id:
            issues_to_linked_pulls[issue_id].add(parsed_issue_link)
        else:
            print(f'Not enough information or not issue <-> pull request link. Skipping {parsed_issue_link}')

    filtered_parsed_issue_links: set[dict] = set()

    for pull_id, parsed_issue_links in pulls_to_linked_issues.items():
        pull_request = pulls_by_id[pull_id]

        # If more than one issue -- skip as pull request not atomic
        if len(parsed_issue_links) != 1:
            print(f"Skipping pull request {pull_request['html_url']} "
                  f"as it connected to more then one issue...")
            continue

        parsed_issue_link = parsed_issue_links.pop()
        linked_issue_id = url_to_id(parsed_issue_link['linked_issue_html_url'])
        if len(issues_to_linked_pulls[linked_issue_id]) != 1:
            print(f"Skipping pull request {pull_request['html_url']} "
                  f"as linked issue connected to more then one pull request...")
            continue

        linked_issue = issues_by_id[linked_issue_id]

        # Check issue is a bug
        if not has_bug_label(linked_issue):
            print(f"Skipping pull request {pull_request['html_url']}. Issue is not a bug...")
            continue

        # Check issue text has no images
        if has_image_in_text(linked_issue):
            print(f"Skipping pull request {pull_request['html_url']}. Issue has images which we can not process...")
            continue

        # Check diff between base and head commit can be extracted
        changed_files = get_changed_files_between_commits(repo_path, pull_request['base_sha'], pull_request['head_sha'])
        if changed_files is None:
            print(f"Skipping pull request {pull_request['html_url']}. Can not get changed files...")
            continue

        # Check repo content on pull base commit can be extracted
        repo_content = get_repo_content_on_commit(repo_path, pull_request['base_sha'])
        if repo_content is None:
            print(f"Skipping pull request {pull_request['html_url']}. Сan not get repo content...")
            continue

        # Filter only python kotlin, java, python files
        changed_files_exts = get_file_exts(changed_files)
        if not any(key in [".py", ".java", ".kt"] for key in changed_files_exts.keys()):
            print(f"Skipping pull request {pull_request['html_url']}. No py|kt|java files in diff...")
            continue

        filtered_parsed_issue_links.add(parsed_issue_link)

    return list(filtered_parsed_issue_links)


def prepare_data(repo_owner: str, repo_name: str, config: DictConfig):
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


@hydra.main(config_path="../../../lca/configs", config_name="server", version_base=None)
def main(config: DictConfig):
    os.makedirs(config.issues_links_filtered_path, exist_ok=True)
    process_repos_data(prepare_data, config)


if __name__ == "__main__":
    main()
