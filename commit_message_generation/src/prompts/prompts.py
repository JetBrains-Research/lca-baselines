from typing import Any, Dict

from .base_prompt import CMGPrompt


class SimpleCMGPrompt(CMGPrompt):
    def _base_prompt(self, processed_commit: Dict[str, Any]) -> str:
        assert "mods" in processed_commit, "SimpleCMGPrompt expects 'mods' key to be present after preprocessing."
        return f"Write a commit message for a given diff.\nDiff:\n{processed_commit['mods']}\nCommit message:\n"


class DetailedCMGPrompt(CMGPrompt):
    def _base_prompt(self, processed_commit: Dict[str, Any]) -> str:
        assert "mods" in processed_commit, "DetailedCMGPrompt expects 'mods' key to be present after preprocessing."
        return f"Write a commit message for a given diff. Start with a heading that serves as a summary of the whole diff: a single sentence in an imperative form, no more than 50 characters long. If you have details to add, do it after a blank line. Do your best to be specific, do not use 'refactor' unless you are absolutely sure that this change is ONLY a refactoring. Your goal is to communicate what the change does without having to look at the source code. Do not go into low-level details like all the changed files, do not be overly verbose. Avoid adding any external references like issue tags, URLs or emails.\nDiff:\n{processed_commit['mods']}\nCommit message:\n"


class DetailedCMGPromptWContext(CMGPrompt):
    def _base_prompt(self, processed_commit: Dict[str, Any]) -> str:
        assert "mods" in processed_commit, (
            "DetailedCMGPromptWContext expects 'mods' key to be present after preprocessing."
        )
        assert "context" in processed_commit, (
            "DetailedCMGPromptWContext expects 'context' key to be present after preprocessing."
        )
        base_prompt = [
            "Write a commit message for a given diff. Start with a heading that serves as a summary of the whole diff: a single sentence in an imperative form, no more than 50 characters long. If you have details to add, do it after a blank line. Do your best to be specific, do not use 'refactor' unless you are absolutely sure that this change is ONLY a refactoring. Your goal is to communicate what the change does without having to look at the source code. Do not go into low-level details like all the changed files, do not be overly verbose. Avoid adding any external references like issue tags, URLs or emails.\n\nAdditional context from the repository:\n"
        ]

        for i, file in enumerate(processed_commit["context"]):
            base_prompt.append(f"FILE {i + 1}: {file['source']}")
            base_prompt.append(file["content"])

        base_prompt.append("\n")
        base_prompt.append(f"Diff:\n\n{processed_commit['mods']}\n\nCommit message:\n")
        return "\n".join(base_prompt)


class DetailedCMGPromptForFullFiles(CMGPrompt):
    def _base_prompt(self, processed_commit: Dict[str, Any]) -> str:
        assert "files" in processed_commit, (
            "DetailedCMGPromptForFullFiles expects 'context' key to be present after preprocessing."
        )
        base_prompt = [
            "Write a commit message for a given modification. Start with a heading that serves as a summary of the whole modification: a single sentence in an imperative form, no more than 50 characters long. If you have details to add, do it after a blank line. Do your best to be specific, do not use 'refactor' unless you are absolutely sure that this change is ONLY a refactoring. Your goal is to communicate what the change does without having to look at the source code. Do not go into low-level details like all the changed files, do not be overly verbose. Avoid adding any external references like issue tags, URLs or emails.\n\nYou will see full contents of the changed files before and after the modification.\n"
        ]

        for i, file in enumerate(processed_commit["files"]):
            assert file["old_path"] == file["new_path"]
            base_prompt.append(f"FILE {i + 1}: {file['old_path']}")
            base_prompt.append("BEFORE THE MODIFICATION:")
            base_prompt.append(file["old_contents"] if file["old_contents"] else "N/A")

            base_prompt.append("AFTER THE MODIFICATION:")
            base_prompt.append(file["new_contents"] if file["new_contents"] else "N/A")

        base_prompt.append("\nCommit message:\n")
        return "\n".join(base_prompt)
