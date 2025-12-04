#!/usr/bin/env python3
"""
Compute MFE/MAE/R paths per ShockFlip event and sweep barrier outcomes.

Outputs:
  - results/event_study/snap_mfe_mae_2024_BTC_agg.parquet
  - results/event_study/barrier_sweep_2024_BTC_agg.csv

Usage example:
python scripts/run_barrier_event_study.py \
  --config configs/snap_harvester_2024_btc_agg.yaml \
  --symbol BTCUSDT \
  --horizons 6,10,20,30,60,120,240 \
  --tp_grid 1.5,2.0,2.5,3.0,4.0 \
  --sl_grid 0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from snap_harvester.config import load_config

EPS = 1e-9


def _parse_floats(csv_list: str) -> List[float]:
    return [float(x) for x in csv_list.split(",") if x.strip()]


def _parse_ints(csv_list: str) -> List[int]:
    return [int(x) for x in csv_list.split(",") if x.strip()]


def load_bars(cfg: dict, symbol: str) -> pd.DataFrame:
    pattern = cfg["data"]["bars_path_template"].replace("{symbol}", symbol)
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No minute bar files found with pattern: {pattern}")

    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(df)
    bars = pd.concat(frames, ignore_index=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    return bars


def load_events(cfg: dict, symbol: str) -> pd.DataFrame:
    events_path = cfg["data"]["events_path"]
    paths = sorted(Path(".").glob(events_path))
    if not paths:
        raise FileNotFoundError(f"No events files found with pattern: {events_path}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    events = pd.concat(frames, ignore_index=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    if "symbol" in events.columns:
        events = events[events["symbol"].str.upper() == symbol.upper()]
    events = events.reset_index(drop=True)
    return events


def compute_event_study(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    horizons: List[int],
    price_col: str,
    atr_col: str,
    time_col: str,
    side_col: str,
) -> pd.DataFrame:
    bars_index = {ts: i for i, ts in enumerate(bars[time_col])}
    close = bars[price_col].to_numpy()
    ts_array = bars[time_col].to_numpy()

    records = []
    horizons_sorted = sorted(horizons)

    for _, ev in events.iterrows():
        ts = ev[time_col]
        side = int(ev[side_col])
        entry_price = float(ev[price_col])
        atr_entry = float(ev[atr_col]) if pd.notnull(ev[atr_col]) else np.nan

        base = {
            "event_id": ev.get("event_id", ""),
            "timestamp": ts,
            "symbol": ev.get("symbol", ""),
            "side": side,
            "entry_price": entry_price,
            "atr_entry": atr_entry,
        }

        if ts not in bars_index or not np.isfinite(atr_entry) or atr_entry <= 0:
            for h in horizons_sorted:
                base[f"MFE_H{h}"] = np.nan
                base[f"MAE_H{h}"] = np.nan
                base[f"R_H{h}"] = np.nan
            records.append(base)
            continue

        start_idx = bars_index[ts]
        for h in horizons_sorted:
            end_idx = min(start_idx + h, len(bars) - 1)
            if end_idx <= start_idx:
                mfe = mae = r_h = np.nan
            else:
                # forward path strictly after entry bar
                segment = close[start_idx + 1 : end_idx + 1]
                if len(segment) == 0:
                    mfe = mae = r_h = np.nan
                else:
                    r_path = side * (segment - entry_price) / atr_entry
                    mfe = float(np.max(r_path))
                    mae = float(np.min(r_path))
                    r_h = float(r_path[-1])
            base[f"MFE_H{h}"] = mfe
            base[f"MAE_H{h}"] = mae
            base[f"R_H{h}"] = r_h

        records.append(base)

    return pd.DataFrame.from_records(records)


def _first_index_ge(arr: np.ndarray, threshold: float) -> int | None:
    idx = np.argmax(arr >= threshold)
    if idx == 0 and arr[0] < threshold:
        return None
    return int(idx)


def _first_index_le(arr: np.ndarray, threshold: float) -> int | None:
    idx = np.argmax(arr <= threshold)
    if idx == 0 and arr[0] > threshold:
        return None
    return int(idx)


def sweep_barriers(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    horizons: List[int],
    tp_grid: List[float],
    sl_grid: List[float],
    price_col: str,
    atr_col: str,
    time_col: str,
    side_col: str,
) -> pd.DataFrame:
    bars_index = {ts: i for i, ts in enumerate(bars[time_col])}
    close = bars[price_col].to_numpy()

    combos = list(itertools.product(horizons, tp_grid, sl_grid))
    agg = {
        (h, tp, sl): {
            "N": 0,
            "sum_R": 0.0,
            "wins": 0,
            "losses": 0,
            "sum_win": 0.0,
            "sum_loss": 0.0,
        }
        for h, tp, sl in combos
    }

    for _, ev in events.iterrows():
        ts = ev[time_col]
        side = int(ev[side_col])
        entry_price = float(ev[price_col])
        atr_entry = float(ev[atr_col]) if pd.notnull(ev[atr_col]) else np.nan

        if ts not in bars_index or not np.isfinite(atr_entry) or atr_entry <= 0:
            continue

        start_idx = bars_index[ts]
        for h in horizons:
            end_idx = min(start_idx + h, len(bars) - 1)
            if end_idx <= start_idx:
                continue
            segment = close[start_idx + 1 : end_idx + 1]
            if len(segment) == 0:
                continue
            r_path = side * (segment - entry_price) / atr_entry
            r_h = float(r_path[-1])

            for tp in tp_grid:
                idx_tp = _first_index_ge(r_path, tp)
                for sl in sl_grid:
                    idx_sl = _first_index_le(r_path, -sl)

                    if idx_tp is not None and (idx_sl is None or idx_tp <= idx_sl):
                        outcome = tp
                    elif idx_sl is not None:
                        outcome = -sl
                    else:
                        outcome = r_h

                    key = (h, tp, sl)
                    rec = agg[key]
                    rec["N"] += 1
                    rec["sum_R"] += outcome
                    if outcome > 0:
                        rec["wins"] += 1
                        rec["sum_win"] += outcome
                    elif outcome < 0:
                        rec["losses"] += 1
                        rec["sum_loss"] += outcome

    rows = []
    for (h, tp, sl), rec in sorted(agg.items()):
        n = rec["N"]
        if n == 0:
            continue
        total_R = rec["sum_R"]
        p_win = rec["wins"] / n
        p_loss = rec["losses"] / n
        avg_win = rec["sum_win"] / rec["wins"] if rec["wins"] > 0 else 0.0
        avg_loss = rec["sum_loss"] / rec["losses"] if rec["losses"] > 0 else 0.0
        g_ratio = (p_win * avg_win) / ((p_loss * abs(avg_loss)) + EPS)
        rows.append(
            {
                "H": h,
                "TP_R": tp,
                "SL_R": sl,
                "N": n,
                "total_R": total_R,
                "mean_R": total_R / n,
                "p_win": p_win,
                "p_loss": p_loss,
                "avg_win_R": avg_win,
                "avg_loss_R": avg_loss,
                "G_ratio": g_ratio,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrier event study for aggTrades ShockFlip events.")
    ap.add_argument("--config", required=True, help="Path to agg config (YAML).")
    ap.add_argument("--symbol", default=None, help="Symbol override (defaults to first in config).")
    ap.add_argument("--horizons", default="6,10,20,30,60,120,240", help="Comma list of horizons (bars).")
    ap.add_argument("--tp_grid", default="1.5,2.0,2.5,3.0,4.0", help="Comma list of TP in R.")
    ap.add_argument("--sl_grid", default="0.25,0.5,0.75,1.0", help="Comma list of SL in R.")
    ap.add_argument(
        "--out_event_study",
        default="results/event_study/snap_mfe_mae_2024_BTC_agg.parquet",
        help="Parquet path for event-level MFE/MAE/R.",
    )
    ap.add_argument(
        "--out_barrier_sweep",
        default="results/event_study/barrier_sweep_2024_BTC_agg.csv",
        help="CSV path for barrier sweep results.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    symbol = args.symbol or cfg["data"]["symbols"][0]
    horizons = _parse_ints(args.horizons)
    tp_grid = _parse_floats(args.tp_grid)
    sl_grid = _parse_floats(args.sl_grid)

    price_col = cfg["data"]["bars_close_col"]
    atr_col = cfg["data"]["events_atr_col"]
    time_col = cfg["data"]["events_time_col"]
    side_col = cfg["data"]["events_side_col"]

    print(f"[Load] bars for {symbol} ...")
    bars = load_bars(cfg, symbol)
    print(f"[Load] events for {symbol} ...")
    events = load_events(cfg, symbol)

    print(f"[Event Study] computing MFE/MAE for horizons={horizons} ...")
    event_study = compute_event_study(
        events=events,
        bars=bars,
        horizons=horizons,
        price_col=price_col,
        atr_col=atr_col,
        time_col=time_col,
        side_col=side_col,
    )
    out_event_path = Path(args.out_event_study)
    out_event_path.parent.mkdir(parents=True, exist_ok=True)
    event_study.to_parquet(out_event_path, index=False)
    print(f"[Save] event study -> {out_event_path} (n={len(event_study)})")

    print("[Barrier Sweep] sweeping TP/SL grid ...")
    sweep_df = sweep_barriers(
        events=events,
        bars=bars,
        horizons=horizons,
        tp_grid=tp_grid,
        sl_grid=sl_grid,
        price_col=price_col,
        atr_col=atr_col,
        time_col=time_col,
        side_col=side_col,
    )
    out_sweep_path = Path(args.out_barrier_sweep)
    out_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_sweep_path, index=False)
    print(f"[Save] barrier sweep -> {out_sweep_path} (rows={len(sweep_df)})")


if __name__ == "__main__":
    main()
