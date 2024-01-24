from typing import Dict, List

import evaluate  # type: ignore[import-untyped]

from .b_norm import BNorm


class CMGMetrics:
    def __init__(self):
        self.bnorm = BNorm()
        self.bleu = evaluate.load("sacrebleu")
        self.chrf = evaluate.load("chrf")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.bertscore_normalized = evaluate.load("bertscore")

    def reset(self):
        self.bnorm.reset()

    def update(self, predictions: List[str], references: List[str]) -> None:
        self.bnorm.update(predictions=predictions, references=references)
        self.bleu.add_batch(predictions=predictions, references=[[ref] for ref in references])
        self.chrf.add_batch(predictions=predictions, references=[[ref] for ref in references])
        self.rouge.add_batch(predictions=predictions, references=references)
        self.bertscore.add_batch(predictions=predictions, references=references)
        self.bertscore_normalized.add_batch(predictions=predictions, references=references)

    def compute(self) -> Dict[str, float]:
        rouge = self.rouge.compute()
        bertscore = self.bertscore.compute(lang="en")
        bertscore_normalized = self.bertscore_normalized.compute(lang="en", rescale_with_baseline=True)
        return {
            "bnorm": self.bnorm.compute(),
            "bleu": self.bleu.compute(tokenize="13a")["score"],
            "chrf": self.chrf.compute()["score"],
            "rouge1": rouge["rouge1"] * 100,
            "rouge2": rouge["rouge2"] * 100,
            "rougeL": rouge["rougeL"] * 100,
            "bertscore": sum(bertscore["f1"]) / len(bertscore["f1"]),
            "bertscore_normalized": sum(bertscore_normalized["f1"]) / len(bertscore_normalized["f1"]),
        }
