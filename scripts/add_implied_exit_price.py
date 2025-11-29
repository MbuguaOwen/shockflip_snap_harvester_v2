#!/usr/bin/env python3
"""
Add an implied_exit_price column to a meta events CSV using entry, side, ATR, and realized R.
"""

import argparse
from pathlib import Path

import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.utils.ticks import get_tick_size, price_to_tick, tick_to_price


def add_implied_exit_price(
    in_csv: str,
    out_csv: str,
    entry_col: str = "close",
    side_col: str = "side",
    atr_col: str = "atr",
    r_col: str = "r_final",
    symbol_col: str = "symbol",
    tp_r: float = 3.0,
    sl_r: float = 0.5,
) -> None:
    in_path = Path(in_csv)
    out_path = Path(out_csv)

    df = pd.read_csv(in_path)

    missing = [c for c in [entry_col, side_col, atr_col, r_col, symbol_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{in_path} missing required columns: {missing}")

    entry_price = df[entry_col].astype(float)
    side = df[side_col].astype(float)
    atr = df[atr_col].astype(float)
    realized_r = df[r_col].astype(float)
    symbols = df[symbol_col].astype(str)

    if not ((side == 1) | (side == -1)).all():
        bad = side.loc[~((side == 1) | (side == -1))].unique()
        raise ValueError(f"side column must be +/-1, found {bad}")

    df["entry_price"] = entry_price
    df["sl_price"] = entry_price - side * sl_r * atr
    df["tp_price"] = entry_price + side * tp_r * atr
    df["be_price"] = entry_price

    df["implied_exit_price"] = entry_price + side * atr * realized_r

    # Snap to native tick grid per symbol
    snapped_prices = []
    for price, sym in zip(df["implied_exit_price"], symbols):
        tick = price_to_tick(price, get_tick_size(sym))
        snapped_prices.append(tick_to_price(tick, get_tick_size(sym)))
    df["implied_exit_price"] = snapped_prices

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[Save] wrote {len(df):,} rows with implied_exit_price -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Add implied_exit_price to a meta events CSV.")
    ap.add_argument("--in_csv", required=True, help="Input events CSV")
    ap.add_argument("--out_csv", required=True, help="Output CSV with implied_exit_price")
    ap.add_argument("--entry_col", default="close", help="Entry price column name")
    ap.add_argument("--side_col", default="side", help="Side column (+/-1)")
    ap.add_argument("--atr_col", default="atr", help="ATR column")
    ap.add_argument("--r_col", default="r_final", help="Realized R column")
    ap.add_argument("--symbol_col", default="symbol", help="Symbol column")
    ap.add_argument("--tp_r", type=float, default=3.0, help="TP multiple in R (default 3.0)")
    ap.add_argument("--sl_r", type=float, default=0.5, help="SL multiple in R (default 0.5)")
    args = ap.parse_args()

    add_implied_exit_price(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        entry_col=args.entry_col,
        side_col=args.side_col,
        atr_col=args.atr_col,
        r_col=args.r_col,
        symbol_col=args.symbol_col,
        tp_r=args.tp_r,
        sl_r=args.sl_r,
    )


if __name__ == "__main__":
    main()
