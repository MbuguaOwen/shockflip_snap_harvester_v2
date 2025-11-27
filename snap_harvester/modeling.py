from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingClassifier

from .metrics import binary_classification_metrics


def build_feature_matrix(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Turn the meta dataset into a numeric feature matrix."""
    meta_cfg = cfg.get("meta", {})
    label_col = meta_cfg.get("label_col", "y_swing")
    r_col = meta_cfg.get("r_multiple_col", "r_final")
    mfe_col = meta_cfg.get("mfe_col", "mfe_r")
    mae_col = meta_cfg.get("mae_col", "mae_r")

    exclude = set(meta_cfg.get("exclude_feature_cols", []))
    exclude.update(
        {
            label_col,
            r_col,
            mfe_col,
            mae_col,
            "exit_time",
            "p_hat",
        }
    )

    numeric_cols = []
    cat_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    parts: list[pd.DataFrame] = []

    if numeric_cols:
        num_df = df[numeric_cols].astype("float32")
        parts.append(num_df)

    if cat_cols:
        max_card = int(meta_cfg.get("max_category_cardinality", 20))
        small_cat = {}
        for col in cat_cols:
            nunique = df[col].nunique(dropna=True)
            if nunique <= max_card:
                small_cat[col] = df[col].astype("category")

        if small_cat:
            cat_df = pd.DataFrame(small_cat, index=df.index)
            dummies = pd.get_dummies(cat_df, prefix=list(small_cat.keys()), drop_first=True)
            parts.append(dummies)

    if not parts:
        raise ValueError("No usable features found. Check your config.exclude_feature_cols.")

    X = pd.concat(parts, axis=1)
    X = X.fillna(0.0)
    return X


def train_and_evaluate_hgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame | None,
    y_eval: np.ndarray | None,
    params: dict,
    threshold: float = 0.5,
) -> Tuple[HistGradientBoostingClassifier, dict, dict]:
    """Train a HistGradientBoostingClassifier and compute train / eval metrics."""
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    y_prob_train = model.predict_proba(X_train)[:, 1]
    train_metrics = binary_classification_metrics(y_train, y_prob_train, threshold=threshold)

    eval_metrics: dict = {}
    if X_eval is not None and y_eval is not None and len(y_eval) > 0:
        y_prob_eval = model.predict_proba(X_eval)[:, 1]
        eval_metrics = binary_classification_metrics(y_eval, y_prob_eval, threshold=threshold)

    return model, train_metrics, eval_metrics
