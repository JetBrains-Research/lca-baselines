from typing import Optional

import numpy as np


class BaseTokenizer:

    @staticmethod
    def name():
        pass

    def fit(self, texts: list[str]):
        pass

    def tokenize(self, text: str) -> np.ndarray[str]:
        pass
