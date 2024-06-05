import dataclasses
from typing import Callable

@dataclasses.dataclass
class DatapointBase:
    repo_id: int
    repo_name: str
    completion_lines: dict[str, list[int]]
    # completion_lines_raw: dict[str, list[int]]
    context_dict: dict[str, str] | None = None  # keys are filepaths, values are file contents
    completion_dict: dict[str, str] | None = None
    context: str | None = None
    completion: str | None = None
    context_len: int | None = None
    model_input: list[int] | None = None

    def to_model_input(self, tokenizer_call: Callable):
        tokenized_content = tokenizer_call(self)
        self.context_len = len(tokenized_content.context)
        self.model_input = tokenized_content.context + tokenized_content.completion
        return self

    def get_prefix(self, line_num):
        if self.completion_dict is not None:
            completion_content = list(self.completion_dict.values())[0]
        else:
            completion_content = self.completion
        file_lines = completion_content.split('\n')
        prefix = '\n'.join(file_lines[:line_num])
        return prefix

    def get_line(self, line_num):
        if self.completion_dict is not None:
            completion_content = list(self.completion_dict.values())[0]
        else:
            completion_content = self.completion

        file_lines = completion_content.split('\n')
        return file_lines[line_num]

    def get_context(self):
        return self.context_dict

    def get_completion(self):
        return self.completion_dict
