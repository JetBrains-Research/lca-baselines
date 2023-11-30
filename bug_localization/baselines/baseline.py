import numpy as np


class ScoreBaseline:

    @staticmethod
    def name():
        pass

    def score(self, issue_text: str, file_contents: dict[str, str]) -> np.ndarray[int]:
        pass
