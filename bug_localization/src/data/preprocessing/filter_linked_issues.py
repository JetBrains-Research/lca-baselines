import os
import re
from collections import defaultdict
from typing import Dict, List, Set

import hydra
from omegaconf import DictConfig

from src.utils.file_utils import get_file_exts
from src.utils.git_utils import get_repo_content_on_commit, get_diff_between_commits, parse_changed_files_from_diff
from src.utils.jsonl_utils import get_jsonl_data, save_jsonl_data
from src.utils.processing_utils import process_repos_data


def url_to_id(url: str) -> int:
    return int(url.split('/')[-1])


def is_utf_8(text: str) -> int:
    try:
        encoded = text.encode('utf-8')
    except UnicodeEncodeError:
        return False
    else:
        return True


def has_bug_label(issue: dict) -> bool:
    return any(["bug" in label['name'].lower() for label in issue['labels']])


def has_media_in_text(issue: dict) -> bool:
    try:
        # URL
        images = re.findall(
            r"!\[.*?\]\((.*?\.(jpg|png|gif|jpeg|svg|bmp|tiff|webp|heic|psd|raw|mp3|mp4|mov|wmv|avi|mkv))\)",
            issue['body'],
            re.I
        )
        # HTML
        images += re.findall(
            r'src="(.*?\.(jpg|png|gif|jpeg|svg|bmp|tiff|webp|heic|psd|raw|mp3|mp4|mov|wmv|avi|mkv))"',
            issue['body'],
            re.I
        )
        return len(images) > 0
    except Exception as e:
        print("Can not parse images from text", e)
        return False


def filter_linked_issues(
        parsed_issues_links: List[dict], pulls: List[dict], issues: List[dict], repo_path: str
) -> List[dict]:
    pulls_by_id = {url_to_id(pull['html_url']): pull for pull in pulls}
    issues_by_id = {url_to_id(issue['html_url']): issue for issue in issues if
                    url_to_id(issue['html_url']) not in pulls_by_id}

    # Pull to issue relation without duplications
    issue_to_linked_issues: Dict[int, Set[int]] = defaultdict(set)

    for parsed_issue_link in parsed_issues_links:
        issue_id = url_to_id(parsed_issue_link['issue_html_url'])
        linked_issue_id = url_to_id(parsed_issue_link['linked_issue_html_url'])
        if (issue_id in pulls_by_id and linked_issue_id in issues_by_id) or (
                issue_id in issues_by_id and linked_issue_id in pulls_by_id):
            print(f"Link {issue_id} <-> {linked_issue_id}")
            issue_to_linked_issues[issue_id].add(linked_issue_id)

    filtered_parsed_issue_links: list[dict] = []
    filtered_parsed_issue_links_unique: set[tuple[int, int]] = set()

    for parsed_issue_link in parsed_issues_links:
        issue_id = url_to_id(parsed_issue_link['issue_html_url'])
        linked_issue_id = url_to_id(parsed_issue_link['linked_issue_html_url'])

        if issue_id not in issue_to_linked_issues or linked_issue_id not in issue_to_linked_issues[issue_id]:
            print(f'Not enough information or not an issue <-> pull request link. '
                  f'Skipping {parsed_issue_link["issue_html_url"]} <-> {parsed_issue_link["linked_issue_html_url"]}')
            continue

        if issue_id in pulls_by_id:
            pull_id, linked_issue_id = issue_id, linked_issue_id
        else:
            pull_id, linked_issue_id = linked_issue_id, issue_id

        pull_request = pulls_by_id[pull_id]

        # If more than one issue to pull request -- skip as it probably contains changes from several issues
        if (len(issue_to_linked_issues.get(pull_id, set())) > 1 or
                (len(issue_to_linked_issues.get(pull_id, set())) == 1 and
                 linked_issue_id not in issue_to_linked_issues[pull_id])):
            print(f"Pull request connected to more then one issue. "
                  f"Skipping pull request {pull_request['html_url']} ...")
            continue

        # If more than one pull request to one issue -- skip as it probably fixed in several pull requests
        if (len(issue_to_linked_issues.get(linked_issue_id, set())) > 1 or
                (len(issue_to_linked_issues.get(linked_issue_id, set())) == 1 and
                 pull_id not in issue_to_linked_issues[linked_issue_id])):
            print(f"Linked issue connected to more then one pull request. "
                  f"Skipping pull request {pull_request['html_url']} ...")
            continue

        if linked_issue_id in issue_to_linked_issues.get(pull_id, set()) and pull_id not in issue_to_linked_issues.get(
                linked_issue_id, set()):
            links_count = 2
        else:
            links_count = 1

        linked_issue = issues_by_id[linked_issue_id]

        # Check issue is a bug
        if not has_bug_label(linked_issue):
            print(f"Issue is not a bug. "
                  f"Skipping pull request {pull_request['html_url']} ...")
            continue

        if linked_issue['body'] == '' or linked_issue['body'] is None or not is_utf_8(linked_issue['body']):
            print(f"Issue is empty or contains not-utf-8 characters in description. "
                  f"Skipping issue {linked_issue['html_url']} ...")
            continue

        # Check issue text has no media content (images, video, audio)
        if has_media_in_text(linked_issue):
            print(f"Issue has media file which we can not process. "
                  f"Skipping issue {linked_issue['html_url']} ...")
            continue

        # Check diff between base and head commit can be extracted
        try:
            diff = get_diff_between_commits(repo_path, pull_request['base']['sha'], pull_request['head']['sha'])
            changed_files = parse_changed_files_from_diff(diff)
            if diff == '' or diff is None or not is_utf_8(diff):
                print(f"Diff is empty or contains non-utf-8 characters. "
                      f"Skipping pull request {pull_request['html_url']} ...")
                continue
        except Exception as e:
            print(f"Can not get changed files. "
                  f"Skipping pull request {pull_request['html_url']} due to exception {e}...")
            continue

        # Keep only diff with python, java, kotlin files
        changed_files_exts = get_file_exts(changed_files)
        if not any(key in [".py", ".java", ".kt"] for key in changed_files_exts.keys()):
            print(f"No py|kt|java files in diff. "
                  f"Skipping pull request {pull_request['html_url']} ...")
            continue

        # Check repo content on pull base commit can be extracted
        try:
            repo_content = get_repo_content_on_commit(repo_path, pull_request['base']['sha'])
        except Exception as e:
            print(f"Сan not get repo content. "
                  f"Skipping pull request {pull_request['html_url']} due to exception {e}...")
            continue

        if (pull_id, linked_issue_id) not in filtered_parsed_issue_links_unique:
            filtered_parsed_issue_links_unique.add((pull_id, linked_issue_id))
            filtered_parsed_issue_links.append({
                "comment_html_url": parsed_issue_link['comment_html_url'],
                "issue_html_url": pull_request['html_url'],
                "linked_issue_html_url": linked_issue['html_url'],
                "link_type": parsed_issue_link['link_type'],
                "links_count": links_count,
            })

        # Get only diffs without new files
        if any(changed_file not in repo_content for changed_file in changed_files):
            continue

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
