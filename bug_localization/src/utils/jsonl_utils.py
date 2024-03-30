import codecs
import json
import os
from typing import Optional

PERMISSIVE_LICENSES = ['MIT License',
                       'Apache License 2.0',
                       'BSD 3-Clause "New" or "Revised" License',
                       'BSD 2-Clause "Simplified" License']

EXCLUDE_REPOS = ["moj-analytical-services/splink"]


def get_repos(repos_path: str,
              licences: Optional[list[str]] = PERMISSIVE_LICENSES,
              exclude_repos: Optional[list[str]] = EXCLUDE_REPOS) -> list[dict]:
    """
    Parse list of repos (owner, name) from given file (support both old format with .txt and new with .jsonl)
    :param repos_path: path to file with repos list
    :param licences: permissive licences that repos should have
    :param exclude_repos: repos that should be filtered out
    :return: a list of repos
    """
    _, extension = os.path.splitext(repos_path)

    if extension == ".txt":
        repos = get_repos_from_txt_file(repos_path)
    elif extension == ".json":
        repos = get_repos_from_json_file(repos_path)
    else:
        raise Exception("Unsupported repo file format")

    if licences:
        repos = [repo for repo in repos if repo['license'] in licences]
    if exclude_repos:
        repos = [repo for repo in repos if f"{repo['owner']}/{repo['name']}" not in exclude_repos]
    return repos


def get_repos_from_txt_file(repos_path: str) -> list[dict]:
    """
    Parse list of repos (owner, name) from given txt file `repos_path`.
    Repos are stored in format <owner>/<name> separated by new lines.
    :param repos_path: path to parse repos from
    :return: list of repos
    """
    _, extension = os.path.splitext(repos_path)
    assert extension == ".txt"

    repos = []
    with open(repos_path, "r", encoding="utf-8") as f_repos:
        for line in f_repos:
            owner, name = line.strip().split("/")
            repos.append({
                "owner": owner,
                "name": name,
                "language": None,
                "languages": None,
                "license": None,
                "stars": None,
            })
    return repos


def get_repos_from_json_file(repos_path: str) -> list[dict]:
    """
    Parse list of repos (owner, name) from given json file `repos_path`.
    Repos are stored as list of jsons in "items" field.
    Owner and name are stored in "name" field in format <owner>/<name>.
    :param repos_path: path to parse repos from
    :return: list of repos
    """
    _, extension = os.path.splitext(repos_path)
    assert extension == ".json"

    repos = []
    # The only working way to read repos from jsonl file
    repos_json = json.load(codecs.open(repos_path, "r", "latin-1"))
    for repo_info in repos_json["items"]:
        owner, name = repo_info["name"].strip().split("/")
        stars = repo_info["stargazers"]
        repos.append({
            "owner": owner,
            "name": name,
            "language": repo_info["mainLanguage"],
            "languages": repo_info["languages"],
            "license": repo_info['license'],
            "stars": stars
        })

    return repos


def get_jsonl_data(data_path: str, repo_owner: str, repo_name: str) -> Optional[list[dict]]:
    """
    Read data in jsonl format from file with <owner>__<name>.jsonl name (fixed name for repo data).
    :param data_path: path to file with repo data
    :param repo_owner: repo owner
    :param repo_name: repo name
    :return: list of parsed dicts from jsonl file or None is such file does not exist
    """
    # Parse from <owner>__<name>.jsonl
    file_path = os.path.join(data_path, f"{repo_owner}__{repo_name}.jsonl")

    if not os.path.exists(file_path):
        print(f"Path {file_path} does not exists")
        return None

    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    return data_list


def save_jsonl_data(repo_owner: str, repo_name: str, data: list[dict], output_dir_path: str) -> None:
    """
    Save repo data to a jsonl file
    :param repo_owner: repo owner
    :param repo_name: repo name
    :param data: repo data to be saved
    :param output_dir_path: directory to save jsonl file with repo data
    """
    data_path = os.path.join(output_dir_path, f"{repo_owner}__{repo_name}.jsonl")
    with open(data_path, "w") as f_data_output:
        for item in data:
            f_data_output.write(json.dumps(item) + "\n")
