from typing import List

from src.utils import CommitDiff

from .base_preprocessor import CMGDiffPreprocessor


class SimpleCMGPreprocessor(CMGDiffPreprocessor):
    """Concatenates all file diffs into a single diff."""

    def __init__(self, model_name: str, model_provider: str, include_path: bool = True, *args, **kwargs):
        self._include_path = include_path

    def __call__(self, commit_diff: List[CommitDiff], **kwargs) -> str:
        diff = []
        for mod in commit_diff:
            if mod["change_type"] == "UNKNOWN":
                continue
            elif mod["change_type"] == "ADD":
                file_diff = f"new file {mod['new_path']}"
            elif mod["change_type"] == "DELETE":
                file_diff = f"deleted file {mod['old_path']}"
            elif mod["change_type"] == "RENAME":
                file_diff = f"rename from {mod['old_path']}\nrename to {mod['new_path']}"
            elif mod["change_type"] == "COPY":
                file_diff = f"copy from {mod['old_path']}\ncopy to {mod['new_path']}"
            else:
                file_diff = f"{mod['new_path']}"
            if self._include_path:
                diff.append(file_diff)
            diff.append(mod["diff"])

        return "\n".join(diff)
