import numpy as np
import sklearn


def pr_auc_score(y_true: np.ndarray[int], y_pred: np.ndarray[float]) -> float:
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    return sklearn.metrics.auc(fpr, tpr)


def roc_auc_score(y_true: np.ndarray[int], y_pred: np.ndarray[float]) -> float:
    return sklearn.metrics.roc_auc_score(y_true, y_pred)


def f1_score(y_true: np.ndarray[int], y_pred: np.ndarray[float]) -> tuple[float, float]:
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    f1_scores = 2 * tpr * fpr / (tpr + fpr)
    best_f1 = np.max(f1_scores)
    best_thresh = thresholds[np.argmax(f1_scores)]

    return best_f1, best_thresh
