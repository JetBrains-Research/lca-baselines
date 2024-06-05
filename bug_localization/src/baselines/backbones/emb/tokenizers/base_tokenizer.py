from abc import ABC, abstractmethod

import numpy as np


class BaseTokenizer(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, file_contents: np.ndarray[str]):
        pass

    @abstractmethod
    def tokenize(self, file_content: str) -> np.ndarray[str]:
        pass
