import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import git
import unidiff

from src.utils.file_utils import is_test_file


def get_changed_files_between_commits(repo_path: str, first_commit_sha: str, second_commit_sha: str,
                                      extensions: Optional[list[str]] = None,
                                      ignore_tests: bool = False) -> List[str]:
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
    filtered_changed_files = []

    for changed_file in changed_files:
        if ignore_tests and is_test_file(changed_file):
            continue

        if extensions and any(changed_file.endswith(ext) for ext in extensions):
            filtered_changed_files.append(changed_file)

    return filtered_changed_files


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
    source_files = {
        patched_file.source_file.split("a/", 1)[-1]
        for patched_file in unidiff.PatchSet.from_string(diff_str)
    }

    return list(source_files)


def parse_added_files_from_diff(diff_str: str) -> List[str]:
    source_files = {
        patched_file.target_file.split("b/", 1)[-1]
        for patched_file in unidiff.PatchSet.from_string(diff_str) if patched_file.is_added_file
    }

    return list(source_files)


def parse_changed_files_and_lines_from_diff(diff_str: str) -> Dict[str, list[tuple[int, str, str]]]:
    """
    Parse change file names and lines in it from diff
    :param diff_str: diff in string format gather from `get_git_diff_between_commits`
    :return: dict from file path to lines for each changed files according to diff
    """
    changed_files_and_lines = defaultdict(list)
    patch_set = unidiff.PatchSet(diff_str)
    for patched_file in patch_set:
        for hunk in patched_file:
            for line in hunk:
                if line.is_added:
                    changed_files_and_lines[patched_file.path].append((line.target_line_no - 1, 'a', line.value))
                elif line.is_removed:
                    changed_files_and_lines[patched_file.path].append((line.source_line_no - 1, 'r', line.value))

    return dict(changed_files_and_lines)


def get_repo_content_on_commit(repo_path: str, commit_sha: str,
                               extensions: Optional[list[str]] = None,
                               ignore_tests: bool = False) -> Dict[str, Optional[str]]:
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
            if ignore_tests and is_test_file(file_path):
                continue
            full_file_path = os.path.join(repo_path, file_path)
            if not os.path.isfile(full_file_path):
                continue
            try:
                with open(full_file_path, "r") as file:
                    try:
                        content = file.read()
                        file_contents[file_path] = str(content)
                    except Exception as e:
                        file_contents[file_path] = None
                        # print(f"Can not read file with ext {file_path}. Replace with empty string...", e)
            except Exception as e:
                file_contents[file_path] = None
    repo.git.checkout('HEAD', '.')
    return file_contents
