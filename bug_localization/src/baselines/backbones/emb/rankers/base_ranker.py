from abc import ABC

import numpy as np


class BaseRanker(ABC):

    def rank(self, file_names: np.ndarray[str], vect_file_contents: np.ndarray[float]) -> dict:
        pass
