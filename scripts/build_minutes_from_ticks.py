import argparse
from glob import glob
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

# Ensure repo root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.integrity import (
    detect_large_gaps,
    infer_timestamp_unit_from_values,
    to_utc,
    validate_minute_bars,
)
from snap_harvester.logging_utils import get_logger


ALIAS_GROUPS = {
    "timestamp": ["timestamp", "time", "ts"],
    "price": ["price"],
    "qty": ["qty", "quantity"],
    "is_buyer_maker": ["is_buyer_maker", "isBuyerMaker", "is_buyer_maker"],
}


def _detect_usecols(path: str) -> tuple[list[str], dict, dict]:
    """Return available columns, dtype map, and rename map for a tick file."""
    head = pd.read_csv(path, nrows=0)
    cols = [c.strip() for c in head.columns.tolist()]

    usecols: list[str] = []
    rename_map: dict[str, str] = {}
    dtype_map: dict[str, str] = {"price": "float64"}

    for canonical, aliases in ALIAS_GROUPS.items():
        found = next((c for c in aliases if c in cols), None)
        if canonical in ("timestamp", "price") and not found:
            raise ValueError(f"{path}: missing required columns [{canonical}] (aliases tried: {aliases})")
        if found:
            usecols.append(found)
            if found != canonical:
                rename_map[found] = canonical
            if canonical in ("qty", "quantity"):
                dtype_map[found] = "float64"
            elif canonical == "is_buyer_maker":
                dtype_map[found] = "string"

    return usecols, dtype_map, rename_map


def _aggregate_file(
    path: str,
    unit: str,
    logger,
    chunksize: int = 5_000_000,
) -> pd.DataFrame:
    """Stream and aggregate one tick CSV into minute bars."""
    usecols, dtype_map, rename_map = _detect_usecols(path)

    chunk_aggs = []
    dup_removed = 0
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_map,
        low_memory=False,
        chunksize=chunksize,
    ):
        if chunk.empty:
            continue

        if rename_map:
            chunk = chunk.rename(columns=rename_map)
        # Normalize column names for optional fields
        if "quantity" in chunk.columns and "qty" not in chunk.columns:
            chunk = chunk.rename(columns={"quantity": "qty"})

        chunk = chunk.dropna(subset=["timestamp", "price"])

        ts = chunk["timestamp"]
        if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
            ts_int = pd.to_numeric(ts, errors="coerce").astype("Int64")
            if ts_int.isna().any():
                raise ValueError(f"{path}: numeric timestamp parse failed for some rows")
            chunk["timestamp"] = to_utc(
                ts_int.astype("int64"),
                unit=unit,
                name=f"{path}::timestamp",
            )
        else:
            parsed = pd.to_datetime(ts, utc=True, errors="coerce")
            bad = parsed.isna()
            if bad.any():
                logger.warning(
                    "%s::timestamp: dropping %d rows with unparsable timestamps (examples: %s)",
                    path,
                    int(bad.sum()),
                    ts[bad].head(3).tolist(),
                )
                chunk = chunk.loc[~bad].copy()
                parsed = parsed.loc[~bad]
                if chunk.empty:
                    continue
            chunk["timestamp"] = parsed

        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        if "qty" in chunk.columns:
            chunk["qty"] = pd.to_numeric(chunk["qty"], errors="coerce")

        chunk = chunk.dropna(subset=["timestamp", "price"])
        chunk = chunk.sort_values("timestamp")

        dedup_subset = [c for c in ["timestamp", "price", "qty", "is_buyer_maker"] if c in chunk.columns]
        before = len(chunk)
        if dedup_subset:
            chunk = chunk.drop_duplicates(subset=dedup_subset)
            dup_removed += before - len(chunk)

        chunk["minute"] = chunk["timestamp"].dt.floor("min")
        agg = chunk.groupby("minute").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
        )
        chunk_aggs.append(agg.reset_index())

    if dup_removed:
        logger.info("Removed %d duplicate ticks in %s", dup_removed, Path(path).name)

    if not chunk_aggs:
        return pd.DataFrame(columns=["minute", "open", "high", "low", "close"])

    file_agg = pd.concat(chunk_aggs, ignore_index=True)
    file_agg = file_agg.sort_values("minute")
    # Re-aggregate to ensure continuity across chunks within the same file
    file_agg = file_agg.groupby("minute").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()
    return file_agg


def build_minutes_from_paths(
    paths: list[str],
    unit: str,
    price_tolerance: float,
    max_gap_seconds: float,
    logger,
) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No tick files matched the provided glob")

    frames = []
    for p in tqdm(paths, desc="Aggregating tick files"):
        try:
            frames.append(_aggregate_file(p, unit, logger=logger))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed while processing tick file {p}") from exc

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined

    combined = combined.sort_values("minute")
    # Re-aggregate to ensure correctness across file boundaries
    ohlc = combined.groupby("minute").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    ohlc = ohlc.dropna(subset=["open"]).reset_index()
    ohlc.rename(columns={"minute": "timestamp"}, inplace=True)

    validate_minute_bars(
        ohlc,
        time_col="timestamp",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        tolerance=price_tolerance,
    )

    gaps = detect_large_gaps(ohlc["timestamp"], max_gap_seconds=max_gap_seconds)
    if gaps:
        first = gaps[0]
        logger.warning(
            "Detected %d gaps > %.1f seconds (example: %s -> %s, gap=%.1fs)",
            len(gaps),
            max_gap_seconds,
            first[0],
            first[1],
            first[2],
        )

    return ohlc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1-minute OHLC bars from tick data.")
    parser.add_argument("--symbol", required=True, help="Symbol name, e.g., BTCUSDT")
    parser.add_argument(
        "--tick_glob",
        required=True,
        help="Glob for tick CSVs, e.g., data/ticks/BTCUSDT/*.csv",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output Parquet path, e.g., data/minutes/BTCUSDT_1min.parquet",
    )
    parser.add_argument(
        "--timestamp_unit",
        default="infer",
        choices=["infer", "s", "ms", "us", "ns"],
        help="Unit for tick timestamps (default: infer from data, expected ms).",
    )
    parser.add_argument(
        "--max_gap_seconds",
        type=float,
        default=180.0,
        help="Log gaps with no ticks longer than this many seconds (default: 180s).",
    )
    parser.add_argument(
        "--price_tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for OHLC integrity checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("build_minutes")

    tick_paths = sorted(glob(args.tick_glob))
    logger.info("Found %d tick files for %s", len(tick_paths), args.symbol)

    ts_unit = args.timestamp_unit
    if ts_unit == "infer":
        if not tick_paths:
            raise FileNotFoundError("No tick files matched the provided glob")
        sample = pd.read_csv(tick_paths[0], usecols=["timestamp"], nrows=1000)
        ts_unit = infer_timestamp_unit_from_values(sample["timestamp"])
        logger.info("Inferred timestamp unit '%s' from sample %s", ts_unit, tick_paths[0])

    ohlc = build_minutes_from_paths(
        tick_paths,
        unit=ts_unit,
        price_tolerance=args.price_tolerance,
        max_gap_seconds=args.max_gap_seconds,
        logger=logger,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ohlc.to_parquet(out_path, index=False)

    if not ohlc.empty:
        start, end = ohlc["timestamp"].iloc[0], ohlc["timestamp"].iloc[-1]
    else:
        start = end = None

    logger.info("Wrote %d minute bars for %s -> %s", len(ohlc), args.symbol, out_path)
    logger.info("Time span: %s to %s", start, end)


if __name__ == "__main__":
    main()
