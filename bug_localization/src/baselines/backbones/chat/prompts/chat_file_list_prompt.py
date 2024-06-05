from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.baselines.backbones.chat.prompts.chat_prompt_templates import FILE_LIST_PROMPT_TEMPLATE


class ChatFileListPrompt(ChatBasePrompt):
    def base_prompt(self, issue_description: str, project_content: dict[str, str]) -> str:
        file_paths = '\n'.join(project_content.keys())

        return FILE_LIST_PROMPT_TEMPLATE.format(file_paths, issue_description)
