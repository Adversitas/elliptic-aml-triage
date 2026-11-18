"""Label-free graph features computed from the transaction flow graph.

Only the graph structure and node features are used, never labels, so the features can be
computed for a new time step before anyone has investigated it. That keeps them legal to use
at scoring time and free of target leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

GRAPH_FEATURES = [
    "g_in_deg",
    "g_out_deg",
    "g_in2_reach",
    "g_out2_reach",
    "g_fan_ratio",
    "g_pagerank",
    "g_component_size",
    "g_nbr_in_mean_local_0",
    "g_nbr_out_mean_local_0",
    "g_nbr_absdiff_local_0",
]


def _adjacency(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[sp.csr_matrix, np.ndarray]:
    index = pd.Series(np.arange(len(nodes)), index=nodes["txId"].to_numpy())
    keep = edges["src"].isin(index.index) & edges["dst"].isin(index.index)
    src = index[edges.loc[keep, "src"]].to_numpy()
    dst = index[edges.loc[keep, "dst"]].to_numpy()
    n = len(nodes)
    adj = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    adj.data[:] = 1.0  # collapse duplicate edges
    return adj, index.to_numpy()


def pagerank(adj: sp.csr_matrix, alpha: float = 0.85, tol: float = 1e-8, max_iter: int = 200):
    """Power-iteration PageRank on a directed sparse graph (dangling mass spread uniformly)."""
    n = adj.shape[0]
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_deg, out=np.zeros_like(out_deg), where=out_deg > 0)
    transition = sp.diags(inv) @ adj  # row-stochastic for non-dangling rows
    rank = np.full(n, 1.0 / n)
    dangling = out_deg == 0
    for _ in range(max_iter):
        new = alpha * (transition.T @ rank + rank[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(new - rank).sum() < tol:
            return new
        rank = new
    return rank


def compute(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Return a frame aligned with ``nodes`` holding the GRAPH_FEATURES columns."""
    adj, _ = _adjacency(nodes, edges)
    adj_t = adj.T.tocsr()
    in_deg = np.asarray(adj.sum(axis=0)).ravel()
    out_deg = np.asarray(adj.sum(axis=1)).ravel()

    two_out = adj @ adj
    two_in = adj_t @ adj_t
    out2 = np.asarray(((adj + two_out) > 0).sum(axis=1)).ravel()
    in2 = np.asarray(((adj_t + two_in) > 0).sum(axis=1)).ravel()

    und = ((adj + adj_t) > 0).astype(np.float32)
    _, comp = connected_components(und, directed=False)
    comp_size = np.bincount(comp)[comp]

    # Edges never cross time steps, so one global run behaves like independent per-step runs
    # (each step keeps a share of the mass proportional to its size). Multiplying by N makes
    # 1.0 mean "average importance", whatever the size of the step.
    pr_scaled = pagerank(adj) * len(nodes)

    x0 = nodes["local_0"].to_numpy(dtype=np.float64)
    in_mean = np.divide(adj_t @ x0, in_deg, out=np.full(len(x0), np.nan), where=in_deg > 0)
    out_mean = np.divide(adj @ x0, out_deg, out=np.full(len(x0), np.nan), where=out_deg > 0)
    nbr_mean = np.where(np.isnan(in_mean), out_mean, in_mean)
    nbr_mean = np.where(np.isnan(nbr_mean), x0, nbr_mean)  # isolated node: no contrast
    # degree columns already say "no neighbour"; 0 keeps linear models usable
    in_mean, out_mean = np.nan_to_num(in_mean), np.nan_to_num(out_mean)

    return pd.DataFrame(
        {
            "g_in_deg": in_deg,
            "g_out_deg": out_deg,
            "g_in2_reach": in2,
            "g_out2_reach": out2,
            "g_fan_ratio": (out_deg + 1) / (in_deg + 1),
            "g_pagerank": pr_scaled,
            "g_component_size": comp_size,
            "g_nbr_in_mean_local_0": in_mean,
            "g_nbr_out_mean_local_0": out_mean,
            "g_nbr_absdiff_local_0": np.abs(x0 - nbr_mean),
        },
        index=nodes.index,
    ).astype(np.float32)
