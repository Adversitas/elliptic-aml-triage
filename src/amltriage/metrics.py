"""Metrics that matter for alert triage: illicit-class F1, PR-AUC, alert-budget precision."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising illicit-class F1 (used on a validation window only)."""
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    precision = tp / k
    recall = tp / max(1, y_true.sum())
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    return float(scores[order][best])


def illicit_prf(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], average=None, zero_division=0
    )
    return {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0])}


def alert_budget(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray, budget: float):
    """Precision / recall when investigators review the top ``budget`` share of each group.

    Groups are time steps: a compliance team has a fixed capacity per period, so the
    relevant question is how many true illicit cases land in each period's review queue.
    """
    hits, reviewed, positives = 0, 0, int(y_true.sum())
    for g in np.unique(groups):
        m = groups == g
        k = max(1, int(round(budget * m.sum())))
        top = np.argsort(-scores[m])[:k]
        hits += int(y_true[m][top].sum())
        reviewed += k
    return {
        "budget": budget,
        "precision": hits / reviewed,
        "recall": hits / max(1, positives),
    }


def summarise(y_true, scores, threshold, groups, budgets=(0.02, 0.05, 0.10)) -> dict:
    out = illicit_prf(y_true, scores, threshold)
    out["pr_auc"] = float(average_precision_score(y_true, scores))
    out["threshold"] = float(threshold)
    for b in budgets:
        res = alert_budget(y_true, scores, groups, b)
        out[f"precision@{int(b * 100)}%"] = res["precision"]
        out[f"recall@{int(b * 100)}%"] = res["recall"]
    return out


def per_step_f1(y_true, scores, threshold, steps) -> dict[int, float]:
    res = {}
    for s in np.unique(steps):
        m = steps == s
        res[int(s)] = (
            illicit_prf(y_true[m], scores[m], threshold)["f1"] if y_true[m].any() else float("nan")
        )
    return res


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of ``actual`` against the ``expected`` (training) sample."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / max(1, len(actual))
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
