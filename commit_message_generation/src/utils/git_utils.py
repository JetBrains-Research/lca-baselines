import logging
import os
import tarfile

import chardet
import git
from git import GitCommandError
from huggingface_hub import hf_hub_download


def read_file(path: str) -> str:
    with open(path, "rb") as f:
        raw_data = f.read()

    # Detect encoding
    result = chardet.detect(raw_data)
    encoding = result["encoding"]

    # Fallback to UTF-8 if no encoding detected
    if not encoding:
        encoding = "utf-8"

    with open(path, "r", encoding=encoding) as f:
        content = f.read()
    return content


def clone_repo(repository: str, hf_repo_id: str, hf_path_in_repo: str, local_data_dir: str) -> bool:
    extracted_dir = os.path.join(local_data_dir, "extracted_repos")
    repo_path = os.path.join(extracted_dir, repository.replace("/", "__"))

    if not os.path.exists(repo_path):
        try:
            logging.info(f"[{repository}] Downloading from GitHub.")
            _ = git.Repo.clone_from(url=f"https://github.com/{repository}", to_path=repo_path)
            return True
        except Exception as e:
            logging.error(f"[{repository}] Couldn't clone repository from GitHub: {e}.")
            logging.info(f"[{repository}] Downloading from HuggingFace Hub.")
            try:
                local_file_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=os.path.join(hf_path_in_repo, f"{repository.replace('/', '__')}.tar.gz"),
                    repo_type="dataset",
                    local_dir=local_data_dir,
                )
                with tarfile.open(local_file_path, "r:gz") as tar:
                    tar.extractall(path=extracted_dir)
                return True
            except Exception as e:
                logging.error(f"[{repository}] Couldn't download repository from HuggingFace Hub: {e}.")
                return False
    else:
        return True


def checkout_repo_to_commit(repo_path: str, commit_hash: str) -> bool:
    try:
        repo = git.Repo(repo_path)
        repo.git.reset("--hard")
        repo.git.clean("-fd")
        repo.git.checkout(commit_hash)
        return True
    except GitCommandError as e:
        logging.error(
            f"[{os.path.basename(repo_path).replace('__', '/')}] Couldn't checkout repo to the commit {commit_hash}: {e}."
        )
        return False
