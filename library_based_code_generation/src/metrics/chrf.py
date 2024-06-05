from sacrebleu import CHRF

from .metric import Metric


class ChrF(Metric):
    def __init__(self):
        self.chrf = CHRF()

    def score(self, generated_file: str, reference_code: str, unique_apis: list[str]) -> float:
        return self.chrf.sentence_score(generated_file, [reference_code]).score / 100

    def name(self) -> str:
        return "ChrF"
