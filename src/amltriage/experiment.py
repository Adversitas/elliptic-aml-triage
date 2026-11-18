"""Benchmark, explanation and drift experiments under a strict temporal protocol.

Protocol (no information from the future ever reaches a decision):

* **threshold selection**: fit on steps 1-29, choose the F1-optimal threshold on 30-34;
* **final model**: refit on steps 1-34 and apply that threshold, unchanged, to steps 35-49.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import graph_features
from .data import EllipticData
from .metrics import best_f1_threshold, per_step_f1, psi, summarise
from .models import FEATURE_SETS, fit_predict

TRAIN_END, VAL_START = 34, 30


def with_graph_features(data: EllipticData, cache: Path | None = None) -> pd.DataFrame:
    if cache is not None and cache.exists():
        gf = pd.read_parquet(cache)
    else:
        gf = graph_features.compute(data.nodes, data.edges)
        if cache is not None:
            gf.to_parquet(cache, index=False)
    nodes = data.nodes.reset_index(drop=True)
    return pd.concat([nodes, gf.reset_index(drop=True)], axis=1)


def _xy(frame: pd.DataFrame, cols: list[str]):
    return frame[cols].to_numpy(dtype=np.float32), frame["label"].to_numpy(dtype=np.int8)


def run_one(nodes: pd.DataFrame, feature_set: str, model: str, seed: int, balanced: bool = True):
    cols = FEATURE_SETS[feature_set]
    lab = nodes[nodes["label"] >= 0]
    fit_part = lab[lab["time_step"] < VAL_START]
    val_part = lab[(lab["time_step"] >= VAL_START) & (lab["time_step"] <= TRAIN_END)]
    train = lab[lab["time_step"] <= TRAIN_END]
    test = lab[lab["time_step"] > TRAIN_END]

    t0 = time.perf_counter()
    val_scores, _ = fit_predict(model, *_xy(fit_part, cols), _xy(val_part, cols)[0], seed, balanced)
    threshold = best_f1_threshold(val_part["label"].to_numpy(), val_scores)
    x_tr, y_tr = _xy(train, cols)
    x_te, y_te = _xy(test, cols)
    scores, fitted = fit_predict(model, x_tr, y_tr, x_te, seed, balanced)
    elapsed = time.perf_counter() - t0

    steps = test["time_step"].to_numpy()
    res = summarise(y_te, scores, threshold, steps)
    res.update(
        feature_set=feature_set,
        model=model,
        seed=seed,
        balanced=balanced,
        n_features=len(cols),
        fit_seconds=round(elapsed, 2),
        per_step_f1=per_step_f1(y_te, scores, threshold, steps),
    )
    return res, fitted, scores


def benchmark(nodes, feature_sets, models, seeds, balanced=True) -> list[dict]:
    rows = []
    for fs in feature_sets:
        for m in models:
            for s in seeds if m != "logreg" else seeds[:1]:
                res, _, _ = run_one(nodes, fs, m, s, balanced)
                rows.append(res)
                print(
                    f"{fs:12s} {m:7s} seed={s} balanced={balanced} "
                    f"F1={res['f1']:.3f} PR-AUC={res['pr_auc']:.3f} ({res['fit_seconds']}s)"
                )
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["feature_set", "model", "balanced"]
    metrics = ["precision", "recall", "f1", "pr_auc", "precision@5%", "recall@5%"]
    agg = df.groupby(keys)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg["n_seeds"] = df.groupby(keys).size().to_numpy()
    return agg.sort_values("f1_mean", ascending=False)


def explain(nodes: pd.DataFrame, feature_set: str = "all+graph", seed: int = 0, top: int = 10):
    """SHAP explanations for the LightGBM model: global importance + per-alert reason codes."""
    import shap

    res, model, scores = run_one(nodes, feature_set, "lgbm", seed)
    cols = FEATURE_SETS[feature_set]
    lab = nodes[nodes["label"] >= 0]
    test = lab[lab["time_step"] > TRAIN_END].reset_index(drop=True)
    x_te = test[cols].to_numpy(dtype=np.float32)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(x_te)
    sv = sv[1] if isinstance(sv, list) else sv
    importance = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)

    alerts = np.argsort(-scores)[:top]
    reasons = []
    for i in alerts:
        contrib = pd.Series(sv[i], index=cols).sort_values(ascending=False)
        reasons.append(
            {
                "txId": int(test.loc[i, "txId"]),
                "time_step": int(test.loc[i, "time_step"]),
                "score": float(scores[i]),
                "label": int(test.loc[i, "label"]),
                "top_reasons": [
                    {"feature": f, "shap": float(v), "value": float(test.loc[i, f])}
                    for f, v in contrib.head(3).items()
                ],
            }
        )
    return {
        "global_importance": {k: float(v) for k, v in importance.head(20).items()},
        "graph_feature_rank": {
            f: int(list(importance.index).index(f)) + 1 for f in cols if f.startswith("g_")
        },
        "alerts": reasons,
        "metrics": {k: v for k, v in res.items() if k != "per_step_f1"},
    }


def drift(nodes: pd.DataFrame, features: list[str], per_step: dict[int, float]) -> pd.DataFrame:
    """PSI of each test step against the training window, next to that step's F1."""
    lab = nodes[nodes["label"] >= 0]
    train = lab[lab["time_step"] <= TRAIN_END]
    rows = []
    for step, f1 in sorted(per_step.items()):
        cur = lab[lab["time_step"] == step]
        psis = [psi(train[f].to_numpy(), cur[f].to_numpy()) for f in features]
        rows.append(
            {
                "time_step": step,
                "n_labelled": len(cur),
                "n_illicit": int(cur["label"].sum()),
                "illicit_f1": f1,
                "mean_psi_top_features": float(np.mean(psis)),
                "max_psi_top_features": float(np.max(psis)),
            }
        )
    return pd.DataFrame(rows)
