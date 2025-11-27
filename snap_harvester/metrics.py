from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict:
    """Compute core metrics for a binary classifier."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_hat = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}

    # ROC AUC can fail if only one class is present
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        metrics["brier"] = float("nan")

    if y_true.size == 0:
        metrics["accuracy"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_hat))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_hat,
            average="binary",
            zero_division=0,
        )
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1"] = float(f1)

    metrics["n"] = int(y_true.size)
    metrics["pos_rate"] = float(y_true.mean()) if y_true.size > 0 else 0.0
    metrics["avg_prob"] = float(y_prob.mean()) if y_prob.size > 0 else 0.0
    return metrics
