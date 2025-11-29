#!/usr/bin/env python3
"""
Study tail MFE/MAE for ShockFlip / Snap events.

Usage example:
  python scripts/study_tail_mfe.py \
    --config configs/snap_harvester_example.yaml \
    --events_csv results/meta/snap_meta_events_2024_jan_nov.csv \
    --out_csv results/meta/snap_tail_mfe_2024_jan_nov.csv \
    --max_horizon_bars 720
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow running the script directly from the repo root by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.config import load_config
from snap_harvester.logging_utils import get_logger
from snap_harvester.research.tail_mfe import compute_tail_mfe_for_events, load_minutes_bars_for_symbol

COLUMN_ALIASES = {
    "event_ts": ["event_ts", "timestamp", "ts", "time", "event_time"],
    "symbol": ["symbol", "sym"],
    "entry_price": ["entry_price", "price", "close", "entry", "fill_price", "open"],
    "direction": ["direction", "side", "dir", "position", "position_side"],
    "atr": ["atr", "atr_value"],
}

DIR_POS = {"long", "buy", "b", "+1", "1", "bull"}
DIR_NEG = {"short", "sell", "s", "-1", "bear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute tail MFE stats for Snap events.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--events_csv", required=True, help="Input events CSV (ShockFlip / Snap).")
    parser.add_argument("--out_csv", required=True, help="Output CSV with tail stats.")
    parser.add_argument(
        "--max_horizon_bars",
        type=int,
        default=720,
        help="Look-ahead horizon in bars (default: 720 -> 12h on 1m bars).",
    )
    return parser.parse_args()


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _normalize_direction(val) -> int | None:
    if pd.isna(val):
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in DIR_POS:
            return 1
        if s in DIR_NEG:
            return -1
    try:
        num = float(val)
    except Exception:
        return None
    if not np.isfinite(num):
        return None
    if num > 0:
        return 1
    if num < 0:
        return -1
    return None


def normalize_events(df: pd.DataFrame, logger) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _find_column(df, aliases)
        if found is None:
            raise ValueError(f"Missing required column for {canonical!r} (aliases tried: {aliases})")
        rename_map[found] = canonical

    events = df.rename(columns=rename_map).copy()
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True, errors="coerce")
    events["entry_price"] = pd.to_numeric(events["entry_price"], errors="coerce")
    events["atr"] = pd.to_numeric(events["atr"], errors="coerce")
    events["symbol"] = events["symbol"].astype(str)
    events["direction"] = events["direction"].apply(_normalize_direction).astype("Int64")

    bad_mask = (
        events["event_ts"].isna()
        | events["entry_price"].isna()
        | events["atr"].isna()
        | events["direction"].isna()
    )
    if bad_mask.any():
        logger.warning("Dropping %d events with invalid ts/price/atr/direction", int(bad_mask.sum()))
        events = events.loc[~bad_mask].copy()

    return events.sort_values(["symbol", "event_ts"]).reset_index(drop=True)


def summarize_tail(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"[TailMFE] N events: {n}")
    if n == 0:
        return
    for thr in (5, 10, 20, 40):
        col = f"hit_{thr}r"
        prob = df[col].mean() if col in df.columns else float("nan")
        print(f"[TailMFE] P(MFE >= {thr}R): {prob:.3f}")


def main() -> None:
    args = parse_args()
    logger = get_logger("study_tail_mfe")

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    integrity_cfg = cfg.get("integrity", {})
    price_tolerance = float(integrity_cfg.get("price_tolerance", 1e-6))

    logger.info("Loading events from %s", args.events_csv)
    raw_events = pd.read_csv(args.events_csv)
    events = normalize_events(raw_events, logger=logger)
    if events.empty:
        raise SystemExit("No valid events after normalization.")

    horizon = int(args.max_horizon_bars)
    tail_frames: list[pd.DataFrame] = []
    start_bound = events["event_ts"].min() - pd.Timedelta(minutes=1)
    end_bound = events["event_ts"].max() + pd.Timedelta(minutes=horizon + 1)

    for symbol, sym_events in events.groupby("symbol"):
        logger.info("Processing %s (%d events)", symbol, len(sym_events))
        try:
            bars = load_minutes_bars_for_symbol(
                symbol,
                data_cfg,
                price_tolerance=price_tolerance,
                logger=logger,
                start=start_bound,
                end=end_bound,
            )
        except FileNotFoundError as exc:
            logger.error("Skipping %s: %s", symbol, exc)
            continue

        sym_out = compute_tail_mfe_for_events(
            bars=bars,
            events=sym_events,
            max_horizon_bars=horizon,
            time_col=data_cfg["bars_time_col"],
            event_time_col="event_ts",
            entry_col="entry_price",
            direction_col="direction",
            atr_col="atr",
            high_col=data_cfg["bars_high_col"],
            low_col=data_cfg["bars_low_col"],
            logger=logger,
        )
        if sym_out.empty:
            logger.warning("No results for %s (all events skipped).", symbol)
            continue
        tail_frames.append(sym_out)

    if tail_frames:
        tail_df = pd.concat(tail_frames, ignore_index=True)
    else:
        tail_df = pd.DataFrame()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tail_df.to_csv(out_path, index=False)
    logger.info("Saved tail MFE stats -> %s (n=%d)", out_path, len(tail_df))

    summarize_tail(tail_df)


if __name__ == "__main__":
    main()
