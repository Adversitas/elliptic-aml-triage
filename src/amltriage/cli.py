"""Command line interface: ``amltriage download|prepare|benchmark|explain|report``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data as data_mod
from .experiment import aggregate, benchmark, drift, explain, with_graph_features
from .models import FEATURE_SETS, MODELS
from .report import write_report


def _nodes(processed: Path):
    d = data_mod.load(processed)
    return with_graph_features(d, cache=processed / "graph_features.parquet")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="amltriage", description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="fetch the Elliptic CSVs (about 150 MB zipped)")
    sub.add_parser("prepare", help="convert CSVs to parquet")
    b = sub.add_parser("benchmark", help="train/evaluate models x feature sets")
    b.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    b.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    b.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    b.add_argument("--unbalanced", action="store_true", help="disable class re-weighting")
    e = sub.add_parser("explain", help="SHAP reason codes + drift analysis (LightGBM)")
    e.add_argument("--feature-set", default="all+graph")
    sub.add_parser("report", help="rebuild reports/results.md from the JSON outputs")
    a = p.parse_args(argv)

    a.reports.mkdir(parents=True, exist_ok=True)
    if a.cmd == "download":
        for path in data_mod.download(a.raw):
            print(path)
    elif a.cmd == "prepare":
        for path in data_mod.prepare(a.raw, a.processed):
            print(path)
    elif a.cmd == "benchmark":
        nodes = _nodes(a.processed)
        rows = benchmark(nodes, a.feature_sets, a.models, a.seeds, balanced=not a.unbalanced)
        tag = "unbalanced" if a.unbalanced else "balanced"
        (a.reports / f"benchmark_{tag}.json").write_text(json.dumps(rows, indent=1))
        print(aggregate(rows).to_string(index=False))
    elif a.cmd == "explain":
        nodes = _nodes(a.processed)
        res = explain(nodes, a.feature_set)
        top = list(res["global_importance"])[:10]
        rows = json.loads((a.reports / "benchmark_balanced.json").read_text())
        ref = next(
            r
            for r in rows
            if r["model"] == "lgbm" and r["feature_set"] == a.feature_set and r["seed"] == 0
        )
        per_step = {int(k): v for k, v in ref["per_step_f1"].items()}
        res["drift"] = drift(nodes, top, per_step).to_dict(orient="records")
        (a.reports / "explain.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res["global_importance"], indent=1))
    elif a.cmd == "report":
        out = write_report(a.reports)
        print(out)


if __name__ == "__main__":
    main()
