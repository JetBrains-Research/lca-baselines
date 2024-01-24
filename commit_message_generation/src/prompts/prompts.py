from .base_prompt import CMGPrompt


class SimpleCMGPrompt(CMGPrompt):
    def _base_prompt(self, diff: str) -> str:
        return f"Write a commit message for a given diff.\nDiff:\n{diff}\nCommit message:\n"


class DetailedCMGPrompt(CMGPrompt):
    def _base_prompt(self, diff: str) -> str:
        return f"Write a commit message for a given diff. Start with a heading that serves as a summary of the whole diff: a single sentence in an imperative form, no more than 50 characters long. If you have details to add, do it after a blank line. Do your best to be specific, do not use 'refactor' unless you are absolutely sure that this change is ONLY a refactoring. Your goal is to communicate what the change does without having to look at the source code. Do not go into low-level details like all the changed files, do not be overly verbose. Avoid adding any external references like issue tags, URLs or emails.\nDiff:\n{diff}\nCommit message:\n"
