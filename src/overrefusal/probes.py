"""Linear probes on activations.

The groups are defined by source (pseudo_harm = OR-Bench/FalseReject/XSTest,
harmless = Alpaca/ToxicChat/WildGuard...), so a probe can reach high accuracy
by recognising *which dataset* a prompt comes from rather than what it asks.
`leave_one_source_out` is the control: train without a source, test on it.
A probe that has learned the concept still labels the unseen source correctly.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def make() -> LogisticRegression:
    # Same settings as the original probes, so old numbers stay comparable.
    return LogisticRegression(C=0.1, max_iter=1000, random_state=42)


def cv_accuracy(X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    return float((cross_val_predict(make(), X, y, cv=folds) == y).mean())


def cv_auroc(X: np.ndarray, y: np.ndarray, folds: int = 5) -> tuple[float, np.ndarray]:
    """Out-of-fold AUROC for a binary target, plus the out-of-fold scores."""
    scores = cross_val_predict(make(), X, y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, scores)), scores


def fit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return make().fit(X, y)


def transfer_accuracy(clf, X: np.ndarray, y: np.ndarray) -> float:
    return float((clf.predict(X) == y).mean())


def leave_one_source_out(X: np.ndarray, y: np.ndarray, sources: np.ndarray) -> dict[str, float]:
    """Accuracy on each source when that source was never seen in training.

    Only sources whose group still has other training sources are tested;
    otherwise the held-out group would have no examples at all.
    """
    out = {}
    for s in np.unique(sources):
        test = sources == s
        train_groups = set(y[~test])
        if not set(y[test]) <= train_groups:
            continue
        out[s] = transfer_accuracy(fit(X[~test], y[~test]), X[test], y[test])
    return out


def majority_baseline(y: np.ndarray) -> float:
    return float(np.bincount(y).max() / len(y))
