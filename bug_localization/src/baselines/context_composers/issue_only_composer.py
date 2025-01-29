from src.baselines.context_composers.base_context_composer import BaseContextComposer
from src.baselines.utils.type_utils import ChatMessage
from src.utils.tokenization_utils import TokenizationUtils


class IssueOnlyComposer(BaseContextComposer):

    def __init__(self, name: str, system_prompt_path: str, user_prompt_path: str):
        self._name = name
        self._system_prompt = self._read_prompt(system_prompt_path)
        self._user_prompt = self._read_prompt(user_prompt_path)

    def compose_chat(self, dp: dict, model_name: str) -> list[ChatMessage]:
        tokenization_utils = TokenizationUtils(model_name)
        return tokenization_utils.truncate([
            {
                "role": "system",
                "content": self._system_prompt,
            },
            {
                "role": "user",
                "content": self._user_prompt.format(
                    repo_name=f"{dp['repo_owner']}/{dp['repo_name']}",
                    issue_description=f"{dp['issue_title']}\n{dp['issue_body']}")
            },
        ])
