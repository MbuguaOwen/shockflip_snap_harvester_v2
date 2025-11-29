#!/usr/bin/env python3
"""
Build a base Snap Harvester trade tape from meta-events.

Input:  meta-events CSV with ShockFlip features + barrier label
Output: trade tape with implied_exit_r and implied_exit_price

Assumptions:
- Barrier column is barrier_y_H30_R3p0_sl0p5 (H=30, TP=3R, SL=0.5R)
- Entry is at bar close (column 'close')
- ATR column is 'atr'
- Direction column is 'side' in {+1, -1}
"""

import argparse
import os
import numpy as np
import pandas as pd


def build_base_trades(
    meta_path: str,
    out_path: str,
    barrier_col: str = "barrier_y_H30_R3p0_sl0p5",
    tp_r: float = 3.0,
    sl_r: float = 0.5,
) -> None:
    if not os.path.exists(meta_path):
        raise SystemExit(f"[Error] meta file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # 1) Find barrier column if name is different
    if barrier_col not in df.columns:
        candidates = [c for c in df.columns if c.startswith("barrier_y_")]
        if not candidates:
            raise SystemExit(
                f"[Error] No barrier_y_* column found in {meta_path}. "
                f"Available columns: {list(df.columns)[:20]}..."
            )
        if len(candidates) > 1:
            print(f"[Warn] {len(candidates)} barrier columns found, using {candidates[0]!r}")
        barrier_col = candidates[0]

    print(f"[Cfg] Using barrier column: {barrier_col}")

    required_cols = ["timestamp", "side", "close", "atr"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[Error] Missing required cols in meta file: {missing}")

    # 2) Basic fields
    ts = df["timestamp"]
    side = df["side"].astype(int)
    entry_price = df["close"].astype(float)
    atr = df["atr"].astype(float)
    barrier_y = df[barrier_col].astype(int)

    # 3) Convert barrier outcome -> R outcome
    #  - If barrier wins (1): +tp_r R
    #  - If barrier loses (0): -sl_r R
    implied_exit_r = np.where(barrier_y == 1, tp_r, -sl_r)

    # 4) Convert R outcome -> price outcome
    implied_exit_price = entry_price + side * implied_exit_r * atr

    # 5) Derived targets
    sl_price = entry_price - side * sl_r * atr
    tp_price = entry_price + side * tp_r * atr
    be_price = entry_price  # BE is just entry for these static barriers

    # 6) Build trade tape (carry over optional columns if present)
    trades = pd.DataFrame(
        {
            "trade_id": np.arange(len(df), dtype=int),
            "event_id": df.get("event_id"),
            "timestamp": ts,
            "symbol": df.get("symbol", "BTCUSDT"),
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "be_price": be_price,
            "tp_price": tp_price,
            "atr": atr,
            barrier_col: barrier_y,
            "implied_exit_r": implied_exit_r,
            "implied_exit_price": implied_exit_price,
            "y_swing_1p5": df.get("y_swing_1p5"),
            "r_final": df.get("r_final"),
            "mfe_r": df.get("mfe_r"),
            "mae_r": df.get("mae_r"),
            "exit_time": df.get("exit_time"),
            "hit_be": df.get("hit_be"),
            "hit_tp": df.get("hit_tp"),
            "hit_sl": df.get("hit_sl"),
        }
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"[Save] {len(trades):,} trades -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build base Snap Harvester trade tape from meta-events."
    )
    ap.add_argument("--meta", required=True, help="Meta-events CSV (with barrier_y_*).")
    ap.add_argument("--out", required=True, help="Output trade tape CSV.")
    ap.add_argument(
        "--barrier_col",
        default="barrier_y_H30_R3p0_sl0p5",
        help="Barrier label column name (auto-detect if missing).",
    )
    ap.add_argument("--tp_r", type=float, default=3.0, help="TP multiple in R.")
    ap.add_argument("--sl_r", type=float, default=0.5, help="SL multiple in R.")
    args = ap.parse_args()

    build_base_trades(
        meta_path=args.meta,
        out_path=args.out,
        barrier_col=args.barrier_col,
        tp_r=args.tp_r,
        sl_r=args.sl_r,
    )


if __name__ == "__main__":
    main()
