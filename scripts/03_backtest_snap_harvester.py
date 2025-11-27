import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Allow running the script directly from the repo root by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.config import load_config
from snap_harvester.logging_utils import get_logger
from snap_harvester.backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Snap Harvester on OOS slice.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/snap_harvester_example.yaml).",
    )
    args = parser.parse_args()

    logger = get_logger("backtest_snap_harvester")
    cfg = load_config(args.config)

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    time_col = data_cfg["events_time_col"]
    meta_with_preds_path = Path(paths_cfg["meta_with_preds_out"])
    df = pd.read_csv(meta_with_preds_path)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    oos_start = pd.to_datetime(train_cfg["oos_start"], utc=True)
    oos_end = pd.to_datetime(train_cfg["oos_end"], utc=True)
    mask_oos = (df[time_col] >= oos_start) & (df[time_col] <= oos_end)
    df_oos = df.loc[mask_oos].copy()

    logger.info("OOS rows for backtest: %d", len(df_oos))

    results = run_backtest(df_oos, cfg)

    out_path = Path(paths_cfg["backtest_summary_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved backtest summary -> %s", out_path)


if __name__ == "__main__":
    main()
