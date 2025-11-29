#!/usr/bin/env python3
"""
Summarize Tail MFE/MAE statistics per symbol and overall.

Example:
  python scripts/summarize_tail_mfe.py ^
    --tail_csv results/meta/snap_tail_mfe_aug_oct.csv ^
    --out_csv  results/meta/snap_tail_mfe_summary_aug_oct.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROB_COLS = ["hit_5r", "hit_10r", "hit_20r", "hit_40r"]
QUANTILES = [0.5, 0.75, 0.9, 0.95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tail MFE/MAE stats.")
    parser.add_argument("--tail_csv", required=True, help="Tail MFE CSV produced by study_tail_mfe.py.")
    parser.add_argument("--out_csv", required=True, help="Output CSV path for the summary table.")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, src_path: Path) -> pd.DataFrame:
    if "mae_r" not in df.columns:
        logger.error("mae_r column missing in %s", src_path)
        sys.exit(1)
    df["mae_r"] = pd.to_numeric(df["mae_r"], errors="coerce")
    df["mfe_r"] = pd.to_numeric(df.get("mfe_r"), errors="coerce")
    for col in PROB_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
    df["symbol"] = df["symbol"].astype(str)
    return df


def quantile_dict(series: pd.Series, prefix: str) -> dict[str, float]:
    if series.empty:
        return {f"{prefix}_q{int(q * 100)}": np.nan for q in QUANTILES}
    return {f"{prefix}_q{int(q * 100)}": float(series.quantile(q)) for q in QUANTILES}


def summarize_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"symbol": label, "n_events": len(df)}
    for col in PROB_COLS:
        row[f"p_{col}"] = float(df[col].mean()) if len(df) else np.nan

    row.update(quantile_dict(df["mae_r"].dropna(), "mae_r"))

    for thr in (10, 20, 40):
        subset = df[df[f"hit_{thr}r"] == 1]
        row.update(quantile_dict(subset["mae_r"].dropna(), f"mae_r_{thr}r"))
    return row


def print_human(row: dict) -> None:
    n = row["n_events"]
    print(f"[Summary][{row['symbol']}] n={n}")
    if n == 0:
        print("  No events.")
        return
    print(
        "  P(hit >= 10R): {p10:.2f}, P(hit >= 20R): {p20:.2f}, P(hit >= 40R): {p40:.2f}".format(
            p10=row.get("p_hit_10r", float("nan")),
            p20=row.get("p_hit_20r", float("nan")),
            p40=row.get("p_hit_40r", float("nan")),
        )
    )
    def fmt(prefix: str) -> str:
        return "q50={q50:.3g}, q75={q75:.3g}, q90={q90:.3g}, q95={q95:.3g}".format(
            q50=row.get(f"{prefix}_q50", float("nan")),
            q75=row.get(f"{prefix}_q75", float("nan")),
            q90=row.get(f"{prefix}_q90", float("nan")),
            q95=row.get(f"{prefix}_q95", float("nan")),
        )

    print("  MAE all (R):   ", fmt("mae_r"))
    print("  MAE 10R wins:  ", fmt("mae_r_10r"))
    print("  MAE 20R wins:  ", fmt("mae_r_20r"))
    print("  MAE 40R wins:  ", fmt("mae_r_40r"))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    tail_path = Path(args.tail_csv)
    if not tail_path.exists():
        logger.error("Tail CSV not found: %s", tail_path)
        sys.exit(1)

    df = pd.read_csv(tail_path)
    df = ensure_columns(df, tail_path)

    summary_rows = []
    for sym in sorted(df["symbol"].unique()):
        summary_rows.append(summarize_group(df[df["symbol"] == sym], sym))
    summary_rows.append(summarize_group(df, "ALL"))

    summary_df = pd.DataFrame(summary_rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    logger.info("Saved summary to %s (rows=%d)", out_path, len(summary_df))

    for row in summary_rows:
        print_human(row)


if __name__ == "__main__":
    main()
