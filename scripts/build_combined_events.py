import argparse
from pathlib import Path
import sys
from typing import List, Mapping

import pandas as pd

# Allow running directly from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.integrity import drop_duplicates_with_log, to_utc
from snap_harvester.logging_utils import get_logger

REQUIRED_COLS = {"timestamp", "symbol", "side", "price", "atr"}
ALIAS_MAP: Mapping[str, str] = {
    "time": "timestamp",
    "ts": "timestamp",
    "direction": "side",
    "mid_price": "price",
}

SIDE_MAP = {
    "long": 1,
    "buy": 1,
    "short": -1,
    "sell": -1,
    "1": 1,
    "-1": -1,
}


def _normalize_side(series: pd.Series, path: Path) -> pd.Series:
    mapped = series.map(SIDE_MAP).fillna(series)
    side = pd.to_numeric(mapped, errors="coerce").astype("Int64")
    if side.isna().any():
        raise ValueError(f"{path}: failed to normalize side to +/-1")
    invalid = ~side.isin([1, -1])
    if invalid.any():
        bad = side[invalid].unique()
        raise ValueError(f"{path}: unexpected side values {bad}")
    return side.astype(int)


def _load_events(path: Path, symbol: str | None, logger) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    df = pd.read_csv(path)

    rename_cols = {c: ALIAS_MAP[c] for c in df.columns if c in ALIAS_MAP}
    df = df.rename(columns=rename_cols)

    if "symbol" not in df.columns:
        if symbol is None:
            raise ValueError(f"{path}: missing symbol column and no default provided")
        df["symbol"] = symbol
    elif symbol is not None:
        df["symbol"] = symbol

    if "price" not in df.columns:
        for candidate in ("close", "open"):
            if candidate in df.columns:
                df["price"] = df[candidate]
                break

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    df["timestamp"] = to_utc(df["timestamp"], name=f"{path}::timestamp")
    df["side"] = _normalize_side(df["side"], path)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["atr"] = pd.to_numeric(df["atr"], errors="coerce")

    if df[["price", "atr"]].isna().any().any():
        raise ValueError(f"{path}: price/atr contains nulls after coercion.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = drop_duplicates_with_log(df, subset=["timestamp", "symbol", "side"], logger=logger, label="events")
    return df


def combine_events(
    btc_path: Path,
    eth_path: Path,
    out_path: Path,
    logger,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    frames.append(_load_events(btc_path, symbol="BTCUSDT", logger=logger))
    frames.append(_load_events(eth_path, symbol="ETHUSDT", logger=logger))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine BTC + ETH events into one CSV.")
    parser.add_argument(
        "--btc",
        default="results/diamond_hunter_btc/events_annotated.csv",
        help="Path to BTC events CSV",
    )
    parser.add_argument(
        "--eth",
        default="results/diamond_hunter_eth/events_annotated.csv",
        help="Path to ETH events CSV",
    )
    parser.add_argument(
        "--out",
        default="results/diamond_hunter/events_annotated.csv",
        help="Output combined CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("build_combined_events")

    combined = combine_events(Path(args.btc), Path(args.eth), Path(args.out), logger=logger)

    counts = combined["symbol"].value_counts().to_dict()
    span = (combined["timestamp"].min(), combined["timestamp"].max())
    logger.info("Wrote combined events -> %s", args.out)
    logger.info("Rows per symbol: %s", counts)
    logger.info("Time span: %s to %s", span[0], span[1])


if __name__ == "__main__":
    main()
