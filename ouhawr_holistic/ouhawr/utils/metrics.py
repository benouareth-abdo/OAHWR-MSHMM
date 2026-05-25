"""
ouhawr/utils/metrics.py

Evaluation metrics for word recognition:
  - Top-1 recognition rate
  - Top-N recognition rate (N-best)
  - Per-class accuracy
  - Confusion matrix summary
"""

import numpy as np
from typing import List, Tuple


def top1_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Standard recognition rate: fraction of correct top-1 predictions."""
    return float(np.mean(y_true == y_pred))


def topn_accuracy(
    y_true: np.ndarray,
    nbest: List[List[Tuple[object, float]]],
) -> float:
    """
    Top-N recognition rate: fraction of samples where the correct class
    appears anywhere in the N-best list.
    """
    correct = 0
    for gt, hyps in zip(y_true, nbest):
        if gt in [h[0] for h in hyps]:
            correct += 1
    return correct / len(y_true)


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Return a dict {class_label: accuracy} for each class."""
    classes = np.unique(y_true)
    acc = {}
    for cls in classes:
        mask = y_true == cls
        acc[cls] = float(np.mean(y_pred[mask] == cls))
    return acc


def recognition_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbest: List[List[Tuple[object, float]]],
) -> str:
    """Return a formatted recognition report string."""
    t1 = top1_accuracy(y_true, y_pred)
    tn = topn_accuracy(y_true, nbest)
    n = len(nbest[0]) if nbest else 0
    lines = [
        "=" * 40,
        "  Recognition Report",
        "=" * 40,
        f"  Top-1 accuracy : {t1 * 100:.2f} %",
        f"  Top-{n} accuracy: {tn * 100:.2f} %",
        f"  Samples        : {len(y_true)}",
        f"  Classes        : {len(np.unique(y_true))}",
        "=" * 40,
    ]
    return "\n".join(lines)
