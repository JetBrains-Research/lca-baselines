import os
import re
from typing import Dict, List, Tuple, Optional

import git


def get_changed_files_between_commits(repo_path: str, first_commit_sha: str, second_commit_sha: str,
                                      extensions: Optional[list[str]] = None) -> List[str]:
    """
    Get changed files between `first_commit` and `second_commit`
    :param repo_path: path to directory where repo is cloned
    :param first_commit_sha: sha of first commit
    :param second_commit_sha: sha of second commit
    :param extensions: list of file extensions to get
    :return: list of changed files
    """

    pull_request_diff = get_diff_between_commits(repo_path, first_commit_sha, second_commit_sha)
    changed_files = parse_changed_files_from_diff(pull_request_diff)
    if not extensions:
        return changed_files

    changed_files_with_extensions = []
    for changed_file in changed_files:
        if any(changed_file.endswith(ext) for ext in extensions):
            changed_files_with_extensions.append(changed_file)

    return changed_files_with_extensions


def get_changed_files_in_commit(repo_path: str, commit_sha: str) -> List[str]:
    """
    Get changed files in commit
    :param repo_path: path to directory where repo is cloned
    :param commit_sha: sha of commit
    :return: list of changed files
    """

    pull_request_diff = get_diff_commit(repo_path, commit_sha)
    return parse_changed_files_from_diff(pull_request_diff)


def get_changed_files_and_lines_between_commits(repo_path: str, first_commit_sha: str, second_commit_sha: str) \
        -> Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    For each changed files get changed lines in commit
    :param repo_path: path to directory where repo is cloned
    :param first_commit_sha: sha of first commit
    :param second_commit_sha: sha of second commit
    :return: dict from file path to lines for each changed files according to diff
    """

    pull_request_diff = get_diff_between_commits(repo_path, first_commit_sha, second_commit_sha)
    return parse_changed_files_and_lines_from_diff(pull_request_diff)


def get_diff_between_commits(repo_path: str, first_commit_sha: str, second_commit_sha: str) -> str:
    """
    Get git diff between `first_commit` and `second_commit` https://matthew-brett.github.io/pydagogue/git_diff_dots.html
    :param repo_path: path to directory where repo is cloned
    :param first_commit_sha: sha of first commit
    :param second_commit_sha: sha of second commit
    :return: git diff in standard string format
    """

    repo = git.Repo(repo_path)

    return repo.git.diff("{}...{}".format(first_commit_sha, second_commit_sha))


def get_diff_commit(repo_path: str, commit_sha: str) -> str:
    """
    Get git diff for commit
    :param repo_path: path to directory where repo is cloned
    :param commit_sha: sha of commit
    :return: git diff in standard string format
    """

    repo = git.Repo(repo_path)
    return repo.git.show(commit_sha)


def parse_changed_files_from_diff(diff_str: str) -> List[str]:
    """
    Parse change file names from diff
    :param diff_str: diff in string format gather from `get_git_diff_between_commits`
    :return: list of changed files according to diff
    """
    changed_files = set()
    for line in diff_str.splitlines():
        if line.startswith("+++ b/"):
            file_name = line[6:]
            changed_files.add(file_name)

    return list(changed_files)


def parse_changed_files_and_lines_from_diff(diff_str: str) -> Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Parse change file names and lines in it from diff
    :param diff_str: diff in string format gather from `get_git_diff_between_commits`
    :return: dict from file path to lines for each changed files according to diff
    """
    changed_files = dict()
    diff_lines = diff_str.splitlines()
    changed_line_regex = re.compile(r"@@ ([-+]\d+,\d+) ([-+]\d+,\d+) @@")

    i = 0
    prev_file_name = None
    while i < len(diff_lines):
        line = diff_lines[i]
        if line.startswith("+++ b/"):
            file_name = line[6:]
            changed_files[file_name] = []
            prev_file_name = file_name

        if prev_file_name is not None:

            matches = changed_line_regex.findall(line)

            for match in matches:
                start1, count1 = map(int, match[0][1:].split(","))
                start2, count2 = map(int, match[1][1:].split(","))
                changed_files[prev_file_name].append(((start1, count1), (start2, count2)))

        i += 1

    return changed_files


def get_repo_content_on_commit(repo_path: str, commit_sha: str,
                               extensions: Optional[list[str]] = None,
                               ignore_tests: bool = False) -> Dict[str, str]:
    """
    Get repo content on specific commit
    :param repo_path: path to directory where repo is cloned
    :param commit_sha: commit shat on what stage get repo's content
    :return: for all files in repo on`commit_sha` stage map from file path (relative from repo root) to it's content
    """
    repo = git.Repo(repo_path)
    repo.git.checkout(commit_sha, f=True)
    commit = repo.commit(commit_sha)

    file_contents = {}
    for blob in commit.tree.traverse():
        if blob.type == "blob":
            file_path = str(blob.path)
            if extensions is not None and not any(file_path.endswith(ext) for ext in extensions):
                continue
            if ignore_tests and any(test_dir in file_path.lower() for test_dir in ['test/', 'tests/']):
                continue
            with open(os.path.join(repo_path, file_path), "r") as file:
                try:
                    content = file.read()
                    file_contents[file_path] = str(content)
                except Exception as e:
                    file_contents[file_path] = ""
                    # print(f"Can not read file with ext {file_path}. Replace with empty string...", e)

    repo.git.checkout('HEAD', '.')
    return file_contents
