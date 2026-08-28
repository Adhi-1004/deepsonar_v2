from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def summarise(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
    }


def per_class_f1(
    truth: np.ndarray, predictions: np.ndarray, classes: list[str]
) -> dict[str, float]:
    scores = f1_score(truth, predictions, average=None, zero_division=0, labels=range(len(classes)))
    return {name: float(score) for name, score in zip(classes, scores, strict=False)}


def confusion(truth: np.ndarray, predictions: np.ndarray, classes: list[str]) -> np.ndarray:
    return confusion_matrix(truth, predictions, labels=range(len(classes)))


def aggregate(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({k for run in runs for k in run})
    out = {}
    for key in keys:
        values = np.array([run[key] for run in runs if key in run], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return out
