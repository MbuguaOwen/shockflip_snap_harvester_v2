#!/usr/bin/env python3
"""
Summarize routed trade PnL by risk_profile and symbol.

Usage:
  python scripts/summarize_routed_pnl.py \
    --routed results/meta/snap_routed_tape_2024_BTC.csv \
    --out_csv results/meta/snap_routed_pnl_summary_2024_BTC.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def summarize_routed_pnl(
    routed_path: str,
    out_csv: str | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    routed_path = Path(routed_path)
    df = pd.read_csv(routed_path)

    if "r_final" not in df.columns:
        raise ValueError("routed file must contain 'r_final' column")

    if group_cols is None:
        # default: group by symbol + risk_profile
        group_cols = []
        if "symbol" in df.columns:
            group_cols.append("symbol")
        if "risk_profile" in df.columns:
            group_cols.append("risk_profile")
        if not group_cols:
            # fallback: all trades as a single group
            group_cols = ["__all__"]
            df["__all__"] = "ALL"

    grouped = df.groupby(group_cols, dropna=False)

    rows = []
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        n = len(g)
        wins = (g["r_final"] > 0).sum()
        losses = (g["r_final"] < 0).sum()
        flats = (g["r_final"] == 0).sum()

        total_r = g["r_final"].sum()
        mean_r = g["r_final"].mean()
        median_r = g["r_final"].median()
        p_win = wins / n if n > 0 else 0.0

        row = {
            "n_trades": n,
            "wins": int(wins),
            "losses": int(losses),
            "flats": int(flats),
            "p_win": p_win,
            "total_r": total_r,
            "mean_r": mean_r,
            "median_r": median_r,
        }

        # unpack group keys into columns
        for col, val in zip(group_cols, key):
            row[col] = val

        rows.append(row)

    summary = pd.DataFrame(rows)

    # nice ordering
    ordered_cols = [
        *(c for c in ["symbol", "risk_profile"] if c in summary.columns),
        "n_trades",
        "wins",
        "losses",
        "flats",
        "p_win",
        "total_r",
        "mean_r",
        "median_r",
    ]
    other_cols = [c for c in summary.columns if c not in ordered_cols]
    summary = summary[ordered_cols + other_cols]

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)

    # pretty print
    print(f"[Summary] {len(summary)} group(s)")
    for _, row in summary.iterrows():
        label_parts = []
        if "symbol" in summary.columns:
            label_parts.append(f"symbol={row['symbol']}")
        if "risk_profile" in summary.columns:
            label_parts.append(f"risk_profile={row['risk_profile']}")
        label = ", ".join(label_parts) if label_parts else "ALL"

        print(f"\n[{label}] n={int(row['n_trades'])}")
        print(
            f"  wins={int(row['wins'])}, losses={int(row['losses'])}, "
            f"flats={int(row['flats'])}, p_win={row['p_win']:.3f}"
        )
        print(
            f"  total_r={row['total_r']:.2f}, "
            f"mean_r={row['mean_r']:.3f}, median_r={row['median_r']:.3f}"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routed",
        required=True,
        help="CSV with routed trades (output of build_routed_trade_tape.py)",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional CSV path to save the summary",
    )
    args = parser.parse_args()

    summarize_routed_pnl(
        routed_path=args.routed,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
