from abc import ABC, abstractmethod

import numpy as np
from rank_bm25 import BM25Okapi

from .utils import split_identifier


class ExampleGenerationModel(ABC):
    @abstractmethod
    def generate(self, task_description: str) -> str:
        pass

    @abstractmethod
    def name(self):
        pass

    def get_prompt(self, instruction: str):
        return f"Generate Python code based on the following instruction. Output ONLY code. DO NOT include explanations or other textual content.\nInstruction: {instruction}"

    def get_bm25_prompt(self, instruction: str, project_apis: list[str], n_selections: int = 20):
        corpus = []

        for name in project_apis:
            corpus.append(split_identifier(name))

        bm25 = BM25Okapi(corpus)

        clean_instruction = "".join(c for c in instruction if c.isalnum() or c == " ").lower().split(" ")
        doc_scores = bm25.get_scores(clean_instruction)
        predictions = []
        for ind in list(reversed(np.argsort(doc_scores)))[:n_selections]:
            predictions.append(project_apis[ind])

        bm25_instruction = (
            instruction
            + "\n\n"
            + "You can find the following APIs from the library helpful:\n"
            + ", ".join(predictions)
        )
        return self.get_prompt(bm25_instruction)
