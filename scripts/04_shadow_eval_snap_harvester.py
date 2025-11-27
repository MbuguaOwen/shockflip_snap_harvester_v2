import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Allow running from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.config import load_config
from snap_harvester.logging_utils import get_logger
from snap_harvester.backtest import run_backtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/snap_harvester_example.yaml",
        help="Path to Snap Harvester config",
    )
    ap.add_argument(
        "--symbol",
        default="ETHUSDT",
        help="Symbol to evaluate (default: ETHUSDT)",
    )
    ap.add_argument(
        "--start",
        required=True,
        help="Start date (inclusive), e.g. 2025-08-01",
    )
    ap.add_argument(
        "--end",
        required=True,
        help="End date (inclusive), e.g. 2025-10-31",
    )
    ap.add_argument(
        "--out",
        default="results/backtest/snap_backtest_shadow_eth_aug_oct.json",
        help="Where to save the shadow backtest summary",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("shadow_eval")

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    time_col = data_cfg["events_time_col"]
    sym_col = data_cfg["events_symbol_col"]

    meta_with_preds_path = Path(paths_cfg["meta_with_preds_out"])
    logger.info("Loading meta with preds from %s", meta_with_preds_path)
    df = pd.read_csv(meta_with_preds_path, parse_dates=[time_col])

    # Filter to symbol + date window
    mask = (
        (df[sym_col] == args.symbol)
        & (df[time_col] >= args.start)
        & (df[time_col] <= args.end)
    )
    df_shadow = df.loc[mask].copy()
    logger.info(
        "Shadow eval window: symbol=%s, %s to %s, rows=%d",
        args.symbol, args.start, args.end, len(df_shadow),
    )

    # Run the same backtest logic (naive + ML thresholds)
    results = run_backtest(df_shadow, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved shadow backtest -> %s", out_path)


if __name__ == "__main__":
    main()
