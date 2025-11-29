import argparse
from pathlib import Path

import pandas as pd


def build_routed_trade_tape(
    trades_path: str,
    preds_path: str,
    out_path: str,
    risk_profile: str = "strategy",
) -> None:
    """
    Build final routed trade tape with p_hat.

    Inputs
    ------
    trades_path : CSV/Parquet with one row per event/trade, e.g.:
        event_id,timestamp,symbol,side,entry_price,exit_price,
        sl_price,be_price,tp_price,y_swing_1p5,r_final,
        mfe_r,mae_r,exit_time,hit_be,hit_tp,hit_sl

    preds_path : CSV/Parquet with at least:
        event_id,p_hat

    Out
    ---
    CSV at `out_path` with header:
        event_id,risk_profile,timestamp,symbol,side,
        entry_price,implied_exit_price,sl_price,be_price,tp_price,
        y_swing_1p5,r_final,mfe_r,mae_r,exit_time,
        hit_be,hit_tp,hit_sl,p_hat
    """
    trades_path = Path(trades_path)
    preds_path = Path(preds_path)
    out_path = Path(out_path)

    # Load trades
    if trades_path.suffix == ".parquet":
        trades = pd.read_parquet(trades_path)
    else:
        trades = pd.read_csv(trades_path)

    # Load predictions
    if preds_path.suffix == ".parquet":
        preds = pd.read_parquet(preds_path)
    else:
        preds = pd.read_csv(preds_path)

    # Basic sanity
    if "event_id" not in trades.columns:
        raise ValueError("trades file must have column 'event_id'")
    if "event_id" not in preds.columns or "p_hat" not in preds.columns:
        raise ValueError("preds file must have columns 'event_id' and 'p_hat'")

    # Ensure exit price column name
    if "implied_exit_price" not in trades.columns:
        if "exit_price" in trades.columns:
            trades = trades.rename(columns={"exit_price": "implied_exit_price"})
        else:
            raise ValueError(
                "trades file must have either 'implied_exit_price' or 'exit_price'"
            )

    # Attach risk profile
    trades["risk_profile"] = risk_profile

    # Attach p_hat
    preds_small = preds[["event_id", "p_hat"]].drop_duplicates("event_id")
    df = trades.merge(preds_small, on="event_id", how="left")

    missing = df["p_hat"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} rows have no p_hat (check event_id alignment).")

    desired_cols = [
        "event_id",
        "risk_profile",
        "timestamp",
        "symbol",
        "side",
        "entry_price",
        "implied_exit_price",
        "sl_price",
        "be_price",
        "tp_price",
        "y_swing_1p5",
        "r_final",
        "mfe_r",
        "mae_r",
        "exit_time",
        "hit_be",
        "hit_tp",
        "hit_sl",
        "p_hat",
    ]

    missing_cols = [c for c in desired_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in merged df: {missing_cols}")

    df = df[desired_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[Save] routed trade tape -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="Path to trades CSV/Parquet")
    parser.add_argument("--preds", required=True, help="Path to predictions CSV/Parquet")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--risk_profile",
        default="strategy",
        help="Risk profile label to tag in output (default: strategy)",
    )
    args = parser.parse_args()

    build_routed_trade_tape(
        trades_path=args.trades,
        preds_path=args.preds,
        out_path=args.out,
        risk_profile=args.risk_profile,
    )


if __name__ == "__main__":
    main()
