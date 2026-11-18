import json

import numpy as np
import pandas as pd
import pytest

from amltriage import graph_features
from amltriage.data import agg_feature_names, local_feature_names, temporal_split
from amltriage.experiment import benchmark, run_one, with_graph_features
from amltriage.metrics import alert_budget, best_f1_threshold, illicit_prf, psi
from amltriage.report import write_report


def synthetic(n_steps=40, per_step=60, seed=0):
    """A tiny Elliptic-shaped dataset: illicit nodes have a shifted local_0 and fan-out."""
    rng = np.random.default_rng(seed)
    rows, edges, tx = [], [], 0
    for step in range(1, n_steps + 1):
        ids = []
        for _ in range(per_step):
            label = rng.choice([1, 0, -1], p=[0.15, 0.55, 0.30])
            feats = rng.normal(size=93 + 72).astype(np.float32)
            if label == 1:
                feats[:5] += 2.0
            rows.append([tx, step, *feats, label])
            ids.append((tx, label))
            tx += 1
        for a, la in ids:
            k = 4 if la == 1 else 1
            for b, _ in [ids[i] for i in rng.integers(0, len(ids), k)]:
                if a != b:
                    edges.append((a, b))
    cols = ["txId", "time_step", *local_feature_names(), *agg_feature_names(), "label"]
    nodes = pd.DataFrame(rows, columns=cols)
    nodes["label"] = nodes["label"].astype(np.int8)
    return nodes, pd.DataFrame(edges, columns=["src", "dst"])


class _Data:
    def __init__(self, nodes, edges):
        self.nodes, self.edges = nodes, edges


def test_temporal_split_never_mixes_periods():
    nodes, _ = synthetic()
    train, test = temporal_split(nodes, train_end=34)
    assert train["time_step"].max() <= 34 < test["time_step"].min()
    assert (train["label"] >= 0).all() and (test["label"] >= 0).all()


def test_graph_features_on_a_hand_made_graph():
    nodes = pd.DataFrame(
        {"txId": [10, 11, 12, 13], "time_step": [1, 1, 1, 2], "local_0": [1.0, 3.0, 5.0, 0.0]}
    )
    edges = pd.DataFrame({"src": [10, 10, 11], "dst": [11, 12, 12]})
    gf = graph_features.compute(nodes, edges)
    assert gf["g_out_deg"].tolist() == [2, 1, 0, 0]
    assert gf["g_in_deg"].tolist() == [0, 1, 2, 0]
    assert gf["g_out2_reach"].tolist() == [2, 1, 0, 0]
    assert gf["g_component_size"].tolist() == [3, 3, 3, 1]
    assert gf.loc[2, "g_nbr_in_mean_local_0"] == pytest.approx(2.0)
    assert gf.loc[3, "g_nbr_absdiff_local_0"] == 0.0  # isolated node
    assert gf["g_pagerank"].mean() == pytest.approx(1.0, rel=1e-5)
    assert not gf.isna().any().any()


def test_pagerank_matches_networkx_style_reference():
    import scipy.sparse as sp

    adj = sp.csr_matrix(np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=float))
    pr = graph_features.pagerank(adj)
    assert pr.sum() == pytest.approx(1.0)
    assert pr[2] > pr[0] > pr[1]


def test_best_threshold_and_prf():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    t = best_f1_threshold(y, s)
    assert t == pytest.approx(0.7)
    assert illicit_prf(y, s, t) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_alert_budget_counts_per_period():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.8, 0.2, 0.3])
    g = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    res = alert_budget(y, s, g, budget=0.25)  # one review per period
    assert res["precision"] == 0.5 and res["recall"] == 0.5


def test_psi_is_zero_for_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(loc=1.5, size=5000)) > 0.5


def test_end_to_end_benchmark_on_synthetic_graph(tmp_path):
    nodes, edges = synthetic()
    full = with_graph_features(_Data(nodes, edges))
    res, _, scores = run_one(full, "local+graph", "logreg", seed=0)
    assert res["f1"] > 0.6  # the signal planted in local_0..4 is easy to find
    assert len(scores) == int(((full["label"] >= 0) & (full["time_step"] > 34)).sum())
    rows = benchmark(full, ["local"], ["logreg", "rf"], seeds=[0])
    assert {r["model"] for r in rows} == {"logreg", "rf"}
    (tmp_path / "benchmark_balanced.json").write_text(json.dumps(rows))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "| local | rf |" in text
