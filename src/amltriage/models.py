"""Model zoo and feature sets."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import agg_feature_names, local_feature_names
from .graph_features import GRAPH_FEATURES

FEATURE_SETS = {
    "local": local_feature_names(),
    "local+graph": local_feature_names() + GRAPH_FEATURES,
    "all": local_feature_names() + agg_feature_names(),
    "all+graph": local_feature_names() + agg_feature_names() + GRAPH_FEATURES,
}

MODELS = ("logreg", "rf", "lgbm", "xgb")


def make_model(name: str, seed: int = 0, balanced: bool = True):
    """Build an unfitted classifier. ``balanced`` re-weights the minority (illicit) class."""
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced" if balanced else None),
        )
    if name == "rf":
        # 50 max features / 100 trees mirrors the strongest baseline of Weber et al. (2019)
        return RandomForestClassifier(
            n_estimators=100,
            max_features=50,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample" if balanced else None,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            random_state=seed,
            class_weight="balanced" if balanced else None,
            verbose=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.5,
            random_state=seed,
            n_jobs=-1,
            eval_metric="aucpr",
            scale_pos_weight=None,
            tree_method="hist",
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def fit_predict(name, x_train, y_train, x_test, seed=0, balanced=True) -> np.ndarray:
    model = make_model(name, seed=seed, balanced=balanced)
    if name == "xgb" and balanced:
        pos = max(1, int(y_train.sum()))
        model.set_params(scale_pos_weight=(len(y_train) - pos) / pos)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1], model
