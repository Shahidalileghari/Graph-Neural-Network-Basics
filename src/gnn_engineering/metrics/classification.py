from __future__ import annotations
import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Labels and predictions must have equal, non-empty shapes.")
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, labels=None) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Labels and predictions must have equal, non-empty shapes.")
    classes = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    scores = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2.0 * tp / denom))
    return float(np.mean(scores)) if scores else 0.0
