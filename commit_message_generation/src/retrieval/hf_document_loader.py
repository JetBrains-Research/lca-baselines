import logging
import os
import tarfile
from typing import Iterator, Optional, Set

import chardet
import git
from git import GitCommandError
from huggingface_hub import hf_hub_download
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class HuggingFaceGitReposLoader(BaseLoader):
    """
    A custom Document Loader that:
      1) Downloads Git repo (packaged as .tar.gz) from a Hugging Face dataset.
      2) Extract it to a local folder.
      3) Iterates through all .py files inside this repo, yielding Document objects.
    """

    def __init__(
        self,
        repository: str,
        hash: str,
        hf_repo_id: str,
        hf_path_in_repo: str,
        local_data_dir: str,
        files_to_exclude: Optional[Set[Optional[str]]] = None,
    ):
        """
        Args:
            repository: Current repository.
            hash: Hash of the current commit.
            hf_repo_id: The HuggingFace dataset ID.
            hf_path_in_repo: Path to folder in HuggingFace dataset containing the .tar.gz repositories.
            local_data_dir: Local directory where the downloaded and extracted repos will be stored.
            files_to_exclude: List of files that should not be returned as documents.
        """
        self.repository = repository
        self.hash = hash
        self.hf_repo_id = hf_repo_id
        self.hf_path_in_repo = hf_path_in_repo
        self.local_data_dir = local_data_dir
        self.files_to_exclude = files_to_exclude
        self.extracted_dir = os.path.join(self.local_data_dir, "extracted_repos")
        os.makedirs(self.extracted_dir, exist_ok=True)

    def lazy_load(self) -> Iterator[Document]:
        repo_path = os.path.join(self.extracted_dir, self.repository.replace("/", "__"))

        # 1) Download corresponding .tar.gz file and extract it
        repo = None
        if not os.path.exists(repo_path):
            try:
                repo = git.Repo.clone_from(url=f"https://github.com/{self.repository}", to_path=repo_path)
            except Exception as e:
                logging.error(f"[{self.repository}] Couldn't clone repository from GitHub: {e}.")
                try:
                    local_file_path = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename=os.path.join(self.hf_path_in_repo, f"{self.repository.replace('/', '__')}.tar.gz"),
                        repo_type="dataset",
                        local_dir=self.local_data_dir,
                    )
                    with tarfile.open(local_file_path, "r:gz") as tar:
                        tar.extractall(path=self.extracted_dir)

                except Exception as e:
                    logging.error(f"[{self.repository}] Couldn't download repository from HuggingFace Hub: {e}.")

        # 2) Checkout repo to the commit before the current one
        try:
            if repo is None:
                repo = git.Repo(repo_path)
            repo.git.reset("--hard")
            repo.git.clean("-fd")
            parent_hash = repo.commit(self.hash).parents[0]
            repo.git.checkout(parent_hash)
        except GitCommandError as e:
            logging.error(f"[{self.repository}] Couldn't checkout repo to the commit before the current one: {e}.")

        # 3) Walk through the repo and yield Documents for each .py file
        for root, _, files in os.walk(repo_path):
            for file_name in files:
                if file_name.endswith(".py"):
                    if self.files_to_exclude and file_name in self.files_to_exclude:
                        continue
                    full_path = os.path.join(root, file_name)
                    try:
                        with open(full_path, "rb") as f:
                            raw_data = f.read()

                        # Detect encoding
                        result = chardet.detect(raw_data)
                        encoding = result["encoding"]

                        # Fallback to UTF-8 if no encoding detected
                        if not encoding:
                            encoding = "utf-8"

                        with open(full_path, "r", encoding=encoding) as f:
                            content = f.read()

                        yield Document(page_content=content, metadata={"source": full_path})
                    except Exception as e:
                        logging.error(f"[{self.repository}] Error reading file {full_path}: {e}")
