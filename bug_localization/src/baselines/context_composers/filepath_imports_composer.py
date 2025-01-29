from src import logger
from src.baselines.context_composers.base_context_composer import BaseContextComposer
from src.baselines.context_composers.tree_sitter.utils import extract_imports
from src.baselines.utils.type_utils import ChatMessage
from src.utils.tokenization_utils import TokenizationUtils


class FilepathImportsComposer(BaseContextComposer):

    def __init__(self, name: str, system_prompt_path: str, user_prompt_path: str):
        self._name = name
        self._system_prompt = self._read_prompt(system_prompt_path)
        self._user_prompt = self._read_prompt(user_prompt_path)

    @staticmethod
    def _get_filepath_imports_list(dp: dict):
        filepath_imports_list = ""
        for path, content in dp['repo_content'].items():
            filepath_imports_list += path + "\n"
            file_extension = path.split(".")[-1]
            filepath_imports_list += "\n".join(extract_imports(content, file_extension)) + "\n"
        return filepath_imports_list

    def compose_chat(self, dp: dict, model_name: str) -> list[ChatMessage]:
        tokenization_utils = TokenizationUtils(model_name)
        messages = [
            {
                "role": "system",
                "content": self._system_prompt,
            },
            {
                "role": "user",
                "content": self._user_prompt.format(
                    repo_name=f"{dp['repo_owner']}/{dp['repo_name']}",
                    issue_description=f"{dp['issue_title']}\n{dp['issue_body']}",
                    filepath_imports=self._get_filepath_imports_list(dp)
                )
            },
        ]
        logger.info(f"Messages tokens count: {tokenization_utils.count_messages_tokens(messages)}")

        truncated_messages = tokenization_utils.truncate(messages)
        logger.info(f"Truncated messages tokens count: {tokenization_utils.count_messages_tokens(truncated_messages)}")

        return truncated_messages
