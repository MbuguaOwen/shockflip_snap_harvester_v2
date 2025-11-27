#!/usr/bin/env python
"""
Research helper: tree model on barrier_y_H30_R3p0_sl0p5 and quick SL/TP grid stats.

Usage:
  python scripts/research_barrier_3R.py \
    --meta-path results/meta/snap_meta_events.csv \
    --target barrier_y_H30_R3p0_sl0p5 \
    --run-grid
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(path: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in {path}")

    drop_cols = [
        target,
        "timestamp",
        "event_bar_time",
        "exit_time",
        "event_id",
        "risk_profile",
        "y_swing_1p5",
        "r_final",
        "mfe_r",
        "mae_r",
        "p_hat",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target].astype(int)

    # One-hot encode low-card categorical columns
    cat_cols = []
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype).startswith("category"):
            if X[col].nunique() <= 20:
                cat_cols.append(col)
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    return X, y


def train_tree_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)

    print(f"[Barrier 3R Model] AUC={auc:.3f}  AP={ap:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    importances = clf.feature_importances_
    feat_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 20 features by importance:")
    for name, imp in feat_imp[:20]:
        print(f"{name:30s} {imp:.4f}")

    return clf


def safe_win_stats(df: pd.DataFrame, sl: float, tp: float) -> Tuple[float, float]:
    """
    Conservative "safe win" probability and pessimistic expectancy:
      - Counts trades that never went below -sl and reached tp in MFE.
      - Expectancy: safe TP, everything else stopped at -sl.
    """
    if "mfe_r" not in df.columns or "mae_r" not in df.columns:
        raise ValueError("mfe_r and mae_r columns are required for grid stats.")
    safe = ((df["mfe_r"] >= tp) & (df["mae_r"] >= -sl)).sum()
    frac = safe / len(df)
    exp_r = frac * tp - (1 - frac) * sl
    return frac, exp_r


def run_grid(df: pd.DataFrame, sl_grid: List[float], tp_grid: List[float]) -> None:
    rows = []
    for sl in sl_grid:
        for tp in tp_grid:
            frac, exp_r = safe_win_stats(df, sl, tp)
            rows.append((sl, tp, frac, exp_r))

    rows_sorted = sorted(rows, key=lambda x: x[3], reverse=True)

    print("\nSL/TP grid (sorted by pessimistic E[R]):")
    print("SL\tTP\tsafe_win_rate\tE[R]_pessimistic")
    for sl, tp, frac, exp_r in rows_sorted:
        print(f"{sl:.1f}\t{tp:.1f}\t{frac:.2%}\t{exp_r:.2f}R")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Research: tree model on 3R barrier and SL/TP grid.")
    ap.add_argument(
        "--meta-path",
        default="results/meta/snap_meta_events.csv",
        help="Path to meta dataset (geometry profile recommended).",
    )
    ap.add_argument(
        "--target",
        default="barrier_y_H30_R3p0_sl0p5",
        help="Target column to model.",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split size (default 0.3).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default 42).",
    )
    ap.add_argument(
        "--run-grid",
        action="store_true",
        help="Also run SL/TP grid stats using mfe_r/mae_r.",
    )
    ap.add_argument(
        "--sl-grid",
        type=str,
        default="1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated SL (R units) for grid search.",
    )
    ap.add_argument(
        "--tp-grid",
        type=str,
        default="3.0,4.0,5.0,6.0",
        help="Comma-separated TP (R units) for grid search.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Load] {args.meta_path}")
    X, y = load_data(args.meta_path, target=args.target)
    print(f"[Data] X shape = {X.shape}, positives = {y.sum()} / {len(y)}")

    _ = train_tree_model(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.run_grid:
        sl_grid = [float(x) for x in args.sl_grid.split(",") if x.strip()]
        tp_grid = [float(x) for x in args.tp_grid.split(",") if x.strip()]
        df_full = pd.read_csv(args.meta_path)
        run_grid(df_full, sl_grid=sl_grid, tp_grid=tp_grid)


if __name__ == "__main__":
    main()
