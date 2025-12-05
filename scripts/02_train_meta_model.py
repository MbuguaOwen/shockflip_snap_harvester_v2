import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib

# Allow running the script directly from the repo root by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.config import load_config
from snap_harvester.logging_utils import get_logger
from snap_harvester.modeling import build_feature_matrix, train_and_evaluate_hgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HGB meta-model for Snap Harvester.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/snap_harvester_example.yaml).",
    )
    args = parser.parse_args()

    logger = get_logger("train_meta_model")
    cfg = load_config(args.config)

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    meta_cfg = cfg["meta"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    meta_path = Path(paths_cfg["meta_out"])
    df = pd.read_csv(meta_path)
    time_col = data_cfg["events_time_col"]
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # Build features + labels
    X = build_feature_matrix(df, cfg)
    logger.info("Train features (%d): %s", X.shape[1], list(X.columns))
    y_col = meta_cfg["label_col"]
    y = df[y_col].astype(int).values

    train_start = pd.to_datetime(train_cfg["train_start"], utc=True)
    train_end = pd.to_datetime(train_cfg["train_end"], utc=True)
    oos_start = pd.to_datetime(train_cfg["oos_start"], utc=True)
    oos_end = pd.to_datetime(train_cfg["oos_end"], utc=True)

    t = df[time_col]

    train_mask = (t >= train_start) & (t <= train_end)
    oos_mask = (t >= oos_start) & (t <= oos_end)

    X_train = X.loc[train_mask]
    y_train = y[train_mask.values]

    X_oos = X.loc[oos_mask]
    y_oos = y[oos_mask.values]

    logger.info("Train size: %d, OOS size: %d", X_train.shape[0], X_oos.shape[0])

    params = model_cfg.get("params", {})
    model, train_metrics, oos_metrics = train_and_evaluate_hgb(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_oos,
        y_eval=y_oos,
        params=params,
        threshold=0.5,
    )

    # Save model
    model_out = Path(paths_cfg["model_out"])
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    logger.info("Saved HGB model -> %s", model_out)

    # Add probabilities for all rows
    y_prob_all = model.predict_proba(X)[:, 1]
    df["p_hat"] = y_prob_all

    phase = np.full(len(df), "other", dtype=object)
    phase[train_mask.values] = "train"
    phase[oos_mask.values] = "oos"
    df["phase"] = phase

    meta_with_preds_out = Path(paths_cfg["meta_with_preds_out"])
    meta_with_preds_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_with_preds_out, index=False)
    logger.info("Saved meta dataset with predictions -> %s", meta_with_preds_out)

    # Save metrics
    metrics_out = Path(paths_cfg["metrics_out"])
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "train": train_metrics,
        "oos": oos_metrics,
    }
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved metrics -> %s", metrics_out)


if __name__ == "__main__":
    main()
