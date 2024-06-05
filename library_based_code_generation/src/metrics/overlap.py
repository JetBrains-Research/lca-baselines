from ..context.parsed_file import ParsedFile
from .metric import Metric


class Overlap(Metric):
    def __init__(self):
        pass

    def score(self, generated_file: str, reference_code: str, unique_apis: list[str]) -> float:
        parsed_generated_file = ParsedFile(code=generated_file)
        generated_function_calls = parsed_generated_file.called_functions
        guessed_apis = set(generated_function_calls) & set(unique_apis)
        return len(guessed_apis) / len(unique_apis)

    def name(self) -> str:
        return "API_recall"
