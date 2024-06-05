from src.baselines.backbones.agent.prompts.agent_base_prompt import AgentBasePrompt
from src.baselines.backbones.agent.prompts.agent_prompt_templates import AGENT_PROMPT_TEMPLATE


class AgentSimplePrompt(AgentBasePrompt):

    def base_prompt(self, issue_description: str, project_content: dict[str, str]) -> str:
        return AGENT_PROMPT_TEMPLATE.format(issue_description)
