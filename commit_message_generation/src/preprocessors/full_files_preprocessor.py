import logging
import os
from typing import Any, Dict, List, Optional

import git
from git import GitCommandError

from ..utils import TokenizationUtils
from ..utils.git_utils import checkout_repo_to_commit, clone_repo, read_file
from ..utils.typing_utils import UnifiedCommitExample
from . import SimpleCMGPreprocessor


class FullFilesCMGPreprocessor(SimpleCMGPreprocessor):
    """Returns full contents for files before and after commit."""

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        max_num_tokens: Optional[int],
        hf_repo_id: str,
        hf_path_in_repo: str,
        local_data_dir: str,
        include_path: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_provider=model_provider,
            include_path=include_path,
        )
        self.hf_repo_id = hf_repo_id
        self.hf_path_in_repo = hf_path_in_repo
        self.local_data_dir = local_data_dir
        self.max_num_tokens = max_num_tokens
        self._tokenization_utils = TokenizationUtils(model_name=model_name, model_provider=model_provider)

    def _load_repo(self, commit: UnifiedCommitExample) -> List[Dict[str, Optional[str]]]:
        repo_path = os.path.join(self.local_data_dir, "extracted_repos", commit["repo"].replace("/", "__"))

        # 1) Download repo
        clone_repo(
            repository=commit["repo"],
            hf_repo_id=self.hf_repo_id,
            hf_path_in_repo=self.hf_path_in_repo,
            local_data_dir=self.local_data_dir,
        )

        # 2) Checkout repo to the commit before the current one
        try:
            repo = git.Repo(repo_path)
            parent_hash = repo.commit(commit["hash"]).parents[0].hexsha
            checkout_repo_to_commit(repo_path=repo_path, commit_hash=parent_hash)
        except GitCommandError as e:
            logging.error(f"[{commit['repo']}] Couldn't checkout repo to the commit before the current one: {e}.")

        # 3) Get contents before the commit
        processed_mods: List[Dict[str, Optional[str]]] = []
        for file in commit["mods"]:
            if file["old_path"] is not None:
                full_path = os.path.join(repo_path, file["old_path"])
                try:
                    contents = read_file(full_path)
                    processed_mods.append({"old_path": file["old_path"], "old_contents": contents})
                except Exception as e:
                    logging.error(f"[{commit['repo']}] Error reading file {full_path}: {e}")
                    processed_mods.append({"old_path": file["old_path"], "old_contents": None})
            else:
                processed_mods.append({"old_path": file["old_path"], "old_contents": None})

        # 4) Checkout repo to current commit
        try:
            checkout_repo_to_commit(repo_path=repo_path, commit_hash=commit["hash"])
        except GitCommandError as e:
            logging.error(f"[{commit['repo']}] Couldn't checkout repo to the commit before the current one: {e}.")

        # 5) Get contents after before the commit
        for i, file in enumerate(commit["mods"]):
            if file["new_path"] is not None:
                full_path = os.path.join(repo_path, file["new_path"])
                try:
                    contents = read_file(full_path)
                    processed_mods[i]["new_path"] = file["new_path"]
                    processed_mods[i]["new_contents"] = contents
                except Exception as e:
                    logging.error(f"[{commit['repo']}] Error reading file {full_path}: {e}")
                    processed_mods[i]["new_path"] = file["new_path"]
                    processed_mods[i]["new_contents"] = None
            else:
                processed_mods[i]["new_path"] = file["new_path"]
                processed_mods[i]["new_contents"] = None
        return processed_mods

    def __call__(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, Any]:
        processed_commit_mods = self._load_repo(commit)
        if self.max_num_tokens:
            if (
                sum(
                    self._tokenization_utils.count_tokens(file["old_contents"])
                    for file in processed_commit_mods
                    if file["old_contents"]
                )
                + sum(
                    self._tokenization_utils.count_tokens(file["new_contents"])
                    for file in processed_commit_mods
                    if file["new_contents"]
                )
                > self.max_num_tokens
            ):
                num_files = sum(1 for file in processed_commit_mods if file["old_contents"]) + sum(
                    1 for file in processed_commit_mods if file["new_contents"]
                )
                tokens_per_file = self.max_num_tokens // num_files

                for file in processed_commit_mods:
                    if file["old_contents"] is not None:
                        file["old_contents"] = (
                            self._tokenization_utils.truncate(file["old_contents"], max_num_tokens=tokens_per_file)
                            + "\n\n[... the rest is omitted ...]"
                        )

                    if file["new_contents"] is not None:
                        file["new_contents"] = (
                            self._tokenization_utils.truncate(file["new_contents"], max_num_tokens=tokens_per_file)
                            + "\n\n[... the rest is omitted ...]"
                        )

        return {"files": processed_commit_mods}
