import json
from typing import Dict, Any, Optional

from openai import OpenAI
from tenacity import wait_random_exponential, retry, stop_after_attempt

from src.baselines.backbones.agent.env.fs_env import FileSystemEnv
from src.baselines.backbones.agent.prompts.agent_base_prompt import AgentBasePrompt
from src.baselines.backbones.base_backbone import BaseBackbone


class OpenAIAgentBackbone(BaseBackbone):
    def __init__(self, name: str, model_name: str, prompt: AgentBasePrompt, api_key: Optional[str] = None):
        super().__init__(name)
        self._model_name = model_name
        self._prompt = prompt
        self._api_key = api_key

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, client: OpenAI, messages, tools=None):
        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                tools=tools,
            )
            return response
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            return e

    def _run_tool_calls_loop(self, issue_description: str, repo_content: Dict[str, str]) \
            -> tuple[list[tuple[str, str]], str]:
        env = FileSystemEnv(repo_content)
        messages = self._prompt.chat(issue_description, repo_content)

        all_tool_calls = []
        run_tool_calls = True
        final_message = None

        client = OpenAI()

        while run_tool_calls:
            try:
                chat_response = self.chat_completion_request(
                    client, messages, tools=env.get_tools()
                )
                print(chat_response.choices[0].message)
                tool_calls = chat_response.choices[0].message.tool_calls
            except Exception as e:
                print(e)
                run_tool_calls = False
                continue

            if not tool_calls:
                final_message = str(chat_response.choices[0].message)
                run_tool_calls = False
                continue

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function_response = env.run_command(function_name, function_args)
                print(function_response)
                messages.append(
                    {
                        "role": "function",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response,
                    }
                )
                all_tool_calls.append((str(tool_call), function_response))

        return all_tool_calls, final_message

    def localize_bugs(self, issue_description: str, repo_content: Dict[str, str], **kwargs) -> Dict[str, Any]:
        all_tool_calls, final_message = self._run_tool_calls_loop(issue_description, repo_content)
        return {
            "all_tool_calls": all_tool_calls,
            "final_message": final_message
        }
