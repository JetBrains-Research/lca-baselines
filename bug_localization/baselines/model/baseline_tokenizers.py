from typing import Optional

import numpy as np


class BaseTokenizer:

    @staticmethod
    def name():
        pass

    def fit(self, texts: np.ndarray[str]):
        pass

    def tokenize(self, text: str) -> np.ndarray[str]:
        pass
