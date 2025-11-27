import argparse
import os
import sys
from typing import Dict

import pandas as pd


def compute_pf(pnl: pd.Series) -> float:
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    if neg < 0:
        return float(pos / -neg) if (-neg) > 0 else 0.0
    # No losing trades → define PF as infinity
    return float("inf") if pos > 0 else 0.0


def summarize_side(df: pd.DataFrame) -> Dict[str, float]:
    n = int(len(df))
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "pf": 0.0, "sum_pnl": 0.0}
    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    win_rate = float((pnl > 0).mean())
    pf = compute_pf(pnl)
    return {
        "n": n,
        "win_rate": win_rate,
        "pf": pf,
        "sum_pnl": float(pnl.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PF by side from trades CSV.")
    parser.add_argument(
        "--trades",
        type=str,
        default="results/backtest/OOS/trades.csv",
        help="Path to trades CSV (must include 'side' and 'pnl' columns).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save a CSV summary (PF per side).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.trades):
        print(f"[Error] trades file not found: {args.trades}")
        sys.exit(1)

    df = pd.read_csv(args.trades)
    if "side" not in df.columns or "pnl" not in df.columns:
        print("[Error] trades CSV must contain 'side' and 'pnl' columns")
        sys.exit(1)

    df_long = df[df["side"] == 1]
    df_short = df[df["side"] == -1]

    long_stats = summarize_side(df_long)
    short_stats = summarize_side(df_short)

    print("PF breakdown (by side):")
    print(
        f"  Longs  - n={long_stats['n']} win={long_stats['win_rate']:.3f} PF={long_stats['pf'] if long_stats['pf']!=float('inf') else 'inf'} sum_pnl={long_stats['sum_pnl']:.6f}"
    )
    print(
        f"  Shorts - n={short_stats['n']} win={short_stats['win_rate']:.3f} PF={short_stats['pf'] if short_stats['pf']!=float('inf') else 'inf'} sum_pnl={short_stats['sum_pnl']:.6f}"
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_df = pd.DataFrame(
            [
                {"side": "long", **long_stats},
                {"side": "short", **short_stats},
            ]
        )
        out_df.to_csv(args.out, index=False)
        print(f"[Save] PF summary -> {args.out}")


if __name__ == "__main__":
    main()

