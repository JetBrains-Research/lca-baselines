import json
import re
from typing import List, Dict, Any, Optional

from src.baselines.backbones.gen.prompts.base_prompt import BasePrompt
from src.utils.tokenization_utils import TokenizationUtils


def check_match_context_size(tokenization_utils: TokenizationUtils,
                             prompt: BasePrompt,
                             issue_description: str,
                             project_content: Dict[str, str],
                             is_chat: bool):
    if is_chat:
        messages = prompt.chat(issue_description, project_content)
        return tokenization_utils.messages_match_context_size(messages)

    text = prompt.complete(issue_description, project_content)
    return tokenization_utils.text_match_context_size(text)


def batch_project_context(model: str,
                          prompt: BasePrompt,
                          issue_description: str,
                          project_content: Dict[str, str],
                          is_chat: bool) -> List[Dict[str, str]]:
    tokenization_utils = TokenizationUtils(model)
    file_paths = list(project_content.keys())

    has_big_message = True
    n = len(file_paths)
    step = len(file_paths)

    while has_big_message:
        has_big_message = False
        for i in range(0, n, step):
            project_content_subset = {f: c for f, c in project_content.items() if f in file_paths[i:i + step]}
            if not check_match_context_size(tokenization_utils, prompt, issue_description, project_content_subset,
                                            is_chat):
                has_big_message = True
                step //= 2
                break

    batched_project_content = [
        {f: c for f, c in project_content.items() if f in file_paths[i:i + step]} for i in range(0, n, step)
    ]
    assert len(file_paths) == sum(len(b) for b in batched_project_content)

    return batched_project_content


def parse_json_response(response: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        print("Failed to parse raw json from response", response)

    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, response, re.MULTILINE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.decoder.JSONDecodeError:
            print("Failed to parse code json from response", response)

    return None


def parse_list_files_completion(raw_completion: str, repo_content: dict[str, str]) -> List[str]:
    json_data = parse_json_response(raw_completion)
    list_files = []

    # If data in json format
    if json_data:
        if 'files' in json_data:
            for file in json_data['files']:
                if file in repo_content.keys():
                    list_files.append(file)
        else:
            print("No 'file' key in json output")

    # If data in list files format
    else:
        for file in raw_completion.split('\n'):
            if file in repo_content.keys():
                list_files.append(file)

    return list_files
