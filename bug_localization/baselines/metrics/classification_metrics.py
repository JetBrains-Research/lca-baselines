import numpy as np
import sklearn


def auc(y_true: np.ndarray[int], y_pred: np.ndarray[float]) -> float:
    return sklearn.metrics.auc(y_true, y_pred)
