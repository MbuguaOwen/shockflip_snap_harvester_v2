#!/usr/bin/env python3
"""
Convert raw Binance trades/aggTrades CSV (no header) -> ShockFlip tick format.

Input (mode=trades):
  trade_id, price, qty, quote_qty, timestamp_ms, is_buyer_maker, is_best_match

Input (mode=aggtrades):
  agg_trade_id, price, qty, first_trade_id, last_trade_id, timestamp_ms, is_buyer_maker, is_best_match

Output:
  CSV with columns: timestamp, price, qty, is_buyer_maker
"""
import argparse
import pandas as pd
from typing import List


def _column_names(mode: str) -> List[str]:
    if mode == "trades":
        return [
            "trade_id",
            "price",
            "qty",
            "quote_qty",
            "timestamp",
            "is_buyer_maker",
            "is_best_match",
        ]
    if mode == "aggtrades":
        return [
            "agg_trade_id",
            "price",
            "qty",
            "first_trade_id",
            "last_trade_id",
            "timestamp",
            "is_buyer_maker",
            "is_best_match",
        ]
    raise ValueError("mode must be 'trades' or 'aggtrades'")


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk[["timestamp", "price", "qty", "is_buyer_maker"]].copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("Int64")
    out["price"] = pd.to_numeric(out["price"], errors="coerce").astype("float64")
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").astype("float64")
    out["is_buyer_maker"] = (
        out["is_buyer_maker"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype("int8")
    )
    return out.dropna(subset=["timestamp", "price", "qty"])


def convert(in_path: str, out_path: str, mode: str, chunksize: int = 1_000_000) -> None:
    cols = _column_names(mode)
    first = True
    total = 0

    for chunk in pd.read_csv(in_path, header=None, names=cols, chunksize=chunksize):
        cleaned = _clean_chunk(chunk)
        cleaned.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        total += len(cleaned)
        first = False

    print(f"[Save] {total:,} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert Binance trades/aggTrades CSV to ShockFlip tick format.")
    ap.add_argument("--in_path", required=True, help="Input raw Binance CSV (no header).")
    ap.add_argument("--out_path", required=True, help="Output CSV path (timestamp,price,qty,is_buyer_maker).")
    ap.add_argument("--mode", choices=["trades", "aggtrades"], required=True, help="Input file schema.")
    ap.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per chunk when streaming large files.")
    args = ap.parse_args()
    convert(args.in_path, args.out_path, args.mode, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
