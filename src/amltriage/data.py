"""Download and load the Elliptic Bitcoin transaction graph.

The dataset (Weber et al., 2019) has 203,769 transactions over 49 time steps, 234,355
directed payment-flow edges and 166 anonymised features per node. 4,545 nodes are labelled
illicit, 42,019 licit and the rest are unknown. Edges only connect transactions of the same
time step, which is what makes a time-based split leakage-free.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.pyg.org/datasets/elliptic"
FILES = ("elliptic_txs_features", "elliptic_txs_edgelist", "elliptic_txs_classes")

N_LOCAL = 93  # local features after the time step column
N_AGG = 72  # neighbourhood aggregates pre-computed by the dataset authors


def local_feature_names() -> list[str]:
    return [f"local_{i}" for i in range(N_LOCAL)]


def agg_feature_names() -> list[str]:
    return [f"agg_{i}" for i in range(N_AGG)]


def download(raw_dir: Path) -> list[Path]:
    """Fetch the three zipped CSV files and extract them into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name in FILES:
        csv_path = raw_dir / f"{name}.csv"
        if not csv_path.exists():
            zip_path = raw_dir / f"{name}.csv.zip"
            urllib.request.urlretrieve(f"{BASE_URL}/{name}.csv.zip", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(raw_dir)
            zip_path.unlink()
        out.append(csv_path)
    return out


@dataclass
class EllipticData:
    nodes: pd.DataFrame  # one row per transaction: txId, time_step, label, features
    edges: pd.DataFrame  # src, dst (txId)

    @property
    def labelled(self) -> pd.DataFrame:
        return self.nodes[self.nodes["label"] >= 0]


def prepare(raw_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Convert the raw CSVs into two compact parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    feats.columns = ["txId", "time_step", *local_feature_names(), *agg_feature_names()]
    feats[feats.columns[2:]] = feats[feats.columns[2:]].astype(np.float32)
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    # 1 = illicit, 2 = licit, "unknown" -> -1 ; we map to 1 / 0 / -1
    mapping = {"1": 1, "2": 0, "unknown": -1}
    classes["label"] = classes["class"].astype(str).map(mapping).astype(np.int8)
    nodes = feats.merge(classes[["txId", "label"]], on="txId", how="left")
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    edges.columns = ["src", "dst"]
    nodes_path, edges_path = out_dir / "nodes.parquet", out_dir / "edges.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    return nodes_path, edges_path


def load(processed_dir: Path) -> EllipticData:
    nodes = pd.read_parquet(processed_dir / "nodes.parquet")
    edges = pd.read_parquet(processed_dir / "edges.parquet")
    return EllipticData(nodes=nodes, edges=edges)


def temporal_split(nodes: pd.DataFrame, train_end: int = 34) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard Elliptic protocol: train on steps 1..train_end, test on the rest."""
    lab = nodes[nodes["label"] >= 0]
    return lab[lab["time_step"] <= train_end], lab[lab["time_step"] > train_end]
