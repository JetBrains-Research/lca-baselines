from typing import Dict, Optional

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion

from baselines.model.baseline_models import ScoreBaseline


class OpenAIBaseline(ScoreBaseline):

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-3.5-turbo",
                 ):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @staticmethod
    def name():
        return 'openai'

    @staticmethod
    def _build_messages(issue_text: str, file_path: str, file_content: str) -> list[ChatCompletionMessageParam]:
        return [
            {
                "role": "system",
                "content": "You are python java and kotlin developer or "
                           "QA who is in a duty and looking through bugs reports in GitHub repo"
            },
            {
                "role": "user",
                "content": ("You are provided with an issue with bug description from a GitHub repository "
                            "along with a file content taken from the same repository. "
                            "Assess the probability that code changes that fix this bug will affect this file. "
                            "Let define the probability as INT VALUE ranging from 0 (very unlikely will affect) "
                            "to 10 (definitely will affect) with increments of 1. "
                            "Provide the response in format of SINGLE INT VALUE, representing this probability, "
                            "EXCLUDING ANY supplementary text, explanations, or annotations. "
                            f"Here is issue description:\n{issue_text}\n"
                            f"Here are {file_path} file with code:\n{file_content}")
            }
        ]

    def _parse_scores(self, outputs: list[Optional[ChatCompletion]]) -> list[int]:
        pass

    def score(self, issue_text: str, file_paths: list[str], file_contents: Dict[str, str]) -> np.ndarray[int]:
        outputs = []
        for file_path in file_paths:
            assert file_path in file_contents
            file_content = file_contents[file_path]
            messages = self._build_messages(issue_text, file_path, file_content)
            try:
                outputs = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )

            except Exception as e:
                print(e)
                outputs.append(None)
                continue

        scores = self._parse_scores(outputs)

        return np.array(scores)
