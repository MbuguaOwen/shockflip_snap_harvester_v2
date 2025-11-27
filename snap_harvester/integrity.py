from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def infer_timestamp_unit_from_values(values: pd.Series) -> str:
    """
    Infer timestamp unit (s, ms, us, ns) from integer magnitudes.

    The heuristic is intentionally simple and biased toward ms, which is the
    expected unit for the provided tick data.
    """
    series = pd.Series(values).dropna()
    if series.empty:
        return "ms"

    max_val = series.astype(np.int64, copy=False).abs().max()
    if max_val >= 1e18:
        return "ns"
    if max_val >= 1e15:
        return "us"
    if max_val >= 1e12:
        return "ms"
    return "s"


def to_utc(series: pd.Series, unit: str | None = None, name: str = "timestamp") -> pd.Series:
    """Convert a timestamp-like series to UTC, failing on invalid values."""
    if unit:
        converted = pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    else:
        converted = pd.to_datetime(series, utc=True, errors="coerce")

    if converted.isna().any():
        bad = converted.isna().sum()
        raise ValueError(f"{name}: failed to convert {bad} timestamps to UTC; check units/format.")
    return converted


def drop_duplicates_with_log(
    df: pd.DataFrame,
    subset: List[str],
    logger: logging.Logger | None = None,
    label: str = "records",
) -> pd.DataFrame:
    before = len(df)
    deduped = df.drop_duplicates(subset=subset).copy()
    removed = before - len(deduped)
    if removed and logger:
        logger.info("Dropped %d duplicate %s based on %s", removed, label, subset)
    return deduped


def detect_large_gaps(
    timestamps: pd.Series,
    max_gap_seconds: float,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Return list of gaps (prev_ts, next_ts, gap_seconds) exceeding threshold."""
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values().reset_index(drop=True)
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, float]] = []
    if ts.empty:
        return gaps

    deltas = ts.diff().dt.total_seconds()
    gap_mask = deltas > max_gap_seconds
    gap_positions = np.where(gap_mask)[0]
    for pos in gap_positions:
        prev_ts = ts.iloc[pos - 1] if pos - 1 >= 0 else pd.NaT
        cur_ts = ts.iloc[pos]
        gaps.append((prev_ts, cur_ts, float(deltas.iloc[pos])))
    return gaps


def validate_minute_bars(
    bars: pd.DataFrame,
    time_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    tolerance: float = 1e-9,
) -> None:
    """Validate OHLC bars against simple invariants."""
    required = [time_col, open_col, high_col, low_col, close_col]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"Minute bars missing required columns: {missing}")

    if bars[time_col].isna().any():
        raise ValueError("Minute bars contain null timestamps after parsing.")

    if bars[time_col].duplicated().any():
        raise ValueError("Minute bars contain duplicate timestamps; ensure 1m uniqueness.")

    if not bars[time_col].is_monotonic_increasing:
        raise ValueError("Minute bars timestamps are not sorted ascending.")

    # Basic OHLC sanity
    highs = bars[high_col].astype(float)
    lows = bars[low_col].astype(float)
    opens = bars[open_col].astype(float)
    closes = bars[close_col].astype(float)

    bad_range = (highs + tolerance < lows) | lows.isna() | highs.isna()
    if bad_range.any():
        idx = bad_range.idxmax()
        raise ValueError(f"Minute bar high/low inconsistency at index {idx}")

    open_out = (opens < lows - tolerance) | (opens > highs + tolerance)
    close_out = (closes < lows - tolerance) | (closes > highs + tolerance)
    if open_out.any():
        idx = open_out.idxmax()
        raise ValueError(f"Open price lies outside high/low at index {idx}")
    if close_out.any():
        idx = close_out.idxmax()
        raise ValueError(f"Close price lies outside high/low at index {idx}")


def align_events_to_bars(
    events: pd.DataFrame,
    bars_by_symbol: Dict[str, pd.DataFrame],
    time_col: str,
    sym_col: str,
    price_col: str,
    bars_time_col: str,
    bars_low_col: str,
    bars_high_col: str,
    tolerance: float = 1e-6,
    max_snap_minutes: int = 1,
    drop_outside: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Ensure each event's entry price lives inside a valid minute bar.

    - If the exact event minute is missing or out of range, we try neighboring
      minutes up to `max_snap_minutes` away.
    - Adds two columns:
        * event_bar_time: timestamp of the bar used for simulation
        * event_bar_offset_min: integer minute offset applied (0 means no snap)
    """
    offsets: List[int] = [0]
    for k in range(1, max_snap_minutes + 1):
        offsets.extend([k, -k])

    indexed_bars: Dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        indexed = df.set_index(bars_time_col)
        if indexed.index.duplicated().any():
            raise ValueError(f"Bars for {sym} contain duplicate timestamps after indexing.")
        indexed_bars[sym] = indexed

    aligned_rows = []
    anomalies = []
    snapped = 0

    for idx, event in events.iterrows():
        sym = str(event[sym_col])
        if sym not in indexed_bars:
            anomalies.append((idx, sym, "missing_bars"))
            continue

        ts = pd.to_datetime(event[time_col], utc=True, errors="coerce")
        price = float(event[price_col]) if pd.notna(event[price_col]) else np.nan
        if pd.isna(ts) or not np.isfinite(price):
            anomalies.append((idx, sym, "invalid_timestamp_or_price"))
            continue

        bars_idx = indexed_bars[sym]
        matched_time = None
        matched_offset = None

        for offset in offsets:
            candidate = ts + pd.Timedelta(minutes=offset)
            try:
                bar = bars_idx.loc[candidate]
            except KeyError:
                continue

            low = float(bar[bars_low_col])
            high = float(bar[bars_high_col])
            if (low - tolerance) <= price <= (high + tolerance):
                matched_time = candidate
                matched_offset = offset
                break

        if matched_time is None:
            anomalies.append((idx, sym, "price_outside_bars"))
            continue

        if matched_offset:
            snapped += 1

        row = event.to_dict()
        row["event_bar_time"] = matched_time
        row["event_bar_offset_min"] = matched_offset
        aligned_rows.append(row)

    if anomalies and not drop_outside:
        sample = anomalies[:5]
        raise ValueError(
            f"{len(anomalies)} events could not be aligned to bars (examples: {sample}). "
            "Fix timestamps/price scales or re-run with a larger snap window."
        )

    if anomalies and logger:
        logger.warning("Dropping %d misaligned events (examples: %s)", len(anomalies), anomalies[:5])

    if logger:
        logger.info("Aligned events to bars; snapped=%d, kept=%d, dropped=%d", snapped, len(aligned_rows), len(anomalies))

    aligned_df = pd.DataFrame(aligned_rows)
    return aligned_df


def validate_risk_price_scale(
    events: pd.DataFrame,
    price_col: str,
    atr_col: str,
    risk_k_atr: float,
    max_risk_to_price_ratio: float = 10.0,
) -> None:
    """Fail if ATR / price scales are implausible given the configured risk_k_atr."""
    price = events[price_col].astype(float)
    atr = events[atr_col].astype(float)
    risk_dist = risk_k_atr * atr

    invalid = (atr <= 0) | (risk_dist <= 0) | (price <= 0) | (~np.isfinite(atr)) | (~np.isfinite(price))
    if invalid.any():
        idx = invalid.idxmax()
        raise ValueError(f"Non-positive or invalid ATR/price at row {idx}")

    ratio = risk_dist.abs() / price.abs().replace(0, np.nan)
    bad_ratio = ratio > max_risk_to_price_ratio
    if bad_ratio.any():
        idx = bad_ratio.idxmax()
        raise ValueError(
            f"Unrealistic ATR/price scale detected at row {idx} "
            f"(risk distance / price > {max_risk_to_price_ratio})."
        )
