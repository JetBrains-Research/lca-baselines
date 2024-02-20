import os
from collections import defaultdict
from typing import Dict, List, Set

import hydra
from omegaconf import DictConfig


def filter_linked_issues(
        parsed_issues_links: List[ParsedLinkedIssue], pulls: List[PullRequest], issues: List[Issue], repo_path: str
) -> List[ParsedLinkedIssue]:
    pulls_by_url = {pull.html_url: pull for pull in pulls}
    issues_by_url = {issue.html_url: issue for issue in issues if issue.html_url not in pulls_by_url}

    # Pull to issue relation without duplications
    pull_to_linked_issues: Dict[str, Set[ParsedLinkedIssue]] = defaultdict(set)
    linked_issues_to_pulls: Dict[str, Set[ParsedLinkedIssue]] = defaultdict(set)

    for parsed_issue_link in parsed_issues_links:
        # Check pull request exists
        if parsed_issue_link.issue_html_url not in pulls_by_url:
            continue
        # Check issue exists
        if parsed_issue_link.linked_issue_html_url not in issues_by_url:
            continue
        pull_to_linked_issues[parsed_issue_link.issue_html_url].add(parsed_issue_link)
        linked_issues_to_pulls[parsed_issue_link.linked_issue_html_url].add(parsed_issue_link)

    filtered_parsed_issue_links: List[ParsedLinkedIssue] = []
    for pull_url, parsed_issue_links in pull_to_linked_issues.items():
        pull_request = pulls_by_url[pull_url]

        # Is more than one issue -- skip as pull request not atomic
        if len(parsed_issue_links) != 1:
            print(f"Skipping pull request {pull_request.html_url} "
                  f"as it connected to more then one issue...")
            continue

        parsed_issue_link = parsed_issue_links.pop()

        if len(linked_issues_to_pulls[parsed_issue_link.linked_issue_html_url]) != 1:
            print(f"Skipping pull request {pull_request.html_url} "
                  f"as linked issue connected to more then one pull request...")
            continue

        linked_issue = issues_by_url[parsed_issue_link.linked_issue_html_url]

        # Check issue is a bug
        if not linked_issue.has_bug_label():
            print(f"Skipping pull request {pull_request.html_url}. Issue is not a bug...")
            continue

        # Check issue text has no images
        if linked_issue.has_image_in_text():
            print(f"Skipping pull request {pull_request.html_url}. Issue has images which we can not process...")
            continue

        # Check diff between base and head commit can be extracted
        changed_files = get_changed_files(repo_path, pull_request)
        if changed_files is None:
            print(f"Skipping pull request {pull_request.html_url}. Can not get changed files...")
            continue

        # Check repo content on pull base commit can be extracted
        repo_content = get_repo_content_on_base_commit(repo_path, pull_request)
        if repo_content is None:
            print(f"Skipping pull request {pull_request.html_url}. Сan not get repo content...")
            continue

        # Filter only python kotlin and java files
        changed_files_exts = get_file_exts(changed_files)
        if not any(key in [".py", ".java", ".kt"] for key in changed_files_exts.keys()):
            print(f"Skipping pull request {pull_request.html_url}. No py|kt|java files in diff...")
            continue

        filtered_parsed_issue_links.append(parsed_issue_link)

    return filtered_parsed_issue_links


def prepare_data(repo: RepoShort, config: DictConfig):
    print(f"Processing repo {repo.get_full_name()}...")

    if os.path.exists(os.path.join(config.issues_links_filtered_path, repo.get_jsonl_filename())):
        print(f"Repo {repo.get_full_name()} already processed")
        return
    repo_path = os.path.join(config.repos_path, f"{repo.get_dirname()}")

    pulls = get_repo_data(config.pulls_path, repo, PullRequest)
    issues = get_repo_data(config.issues_path, repo, Issue)
    parsed_issues_links = get_repo_data(config.issues_links_path, repo, ParsedLinkedIssue)

    if pulls is None or issues is None or parsed_issues_links is None:
        print(f"Not enough info for repo {repo.get_full_name()}. Skipping...")
        return

    filtered_parsed_issue_links = filter_linked_issues(parsed_issues_links, pulls, issues, repo_path)

    save_repo_data_to_jsonl(
        repo,
        [issues_link.to_dict() for issues_link in filtered_parsed_issue_links],
        config.issues_links_filtered_path,
    )


@hydra.main(config_path="../../../lca/configs", config_name="server", version_base=None)
def main(config: DictConfig):
    os.makedirs(config.issues_links_filtered_path, exist_ok=True)
    process_repos_data(prepare_data, config)


if __name__ == "__main__":
    main()
