from typing import List

from torchmetrics import Metric

from .reused_implementations import bleuFromMaps, splitPuncts


class BNorm(Metric):
    def __init__(self):
        """
        B-Norm is a variation of BLEU. It uses smoothing by Lin and Och, 2004
        and does some additional preprocessing steps.

        It was recommended for evaluation of commit message generation approaches in the
        "On the Evaluation of Commit Message Generation Models: An Experimental Study" paper accepted to ICSME 2021.

        This class is a TorchMetrics wrapper over implementation provided in the replication package:
        https://github.com/DeepSoftwareAnalytics/CommitMsgEmpirical/blob/main/metrics/B-Norm.py
        """
        super().__init__()
        self.add_state("bnorm_scores", default=[])

    def update(self, predictions: List[str], references: List[str]):
        prediction_map = {i: [splitPuncts(pred.strip().lower())] for i, pred in enumerate(predictions)}
        gold_map = {i: [splitPuncts(ref.strip().lower())] for i, ref in enumerate(references)}
        self.bnorm_scores.append(bleuFromMaps(gold_map, prediction_map)[0])  # type: ignore

    def compute(self) -> float:
        return sum(self.bnorm_scores) / len(self.bnorm_scores)  # type: ignore
