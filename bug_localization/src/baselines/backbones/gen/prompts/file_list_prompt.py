from src.baselines.backbones.gen.prompts.base_prompt import BasePrompt
from src.baselines.backbones.gen.prompts.prompt_templates import FILE_LIST_PROMPT_TEMPLATE


class FileListPrompt(BasePrompt):
    def base_prompt(self, issue_description: str, project_content: dict[str, str]) -> str:
        file_paths = '\n'.join(project_content.keys())

        return FILE_LIST_PROMPT_TEMPLATE.format(file_paths, issue_description)
