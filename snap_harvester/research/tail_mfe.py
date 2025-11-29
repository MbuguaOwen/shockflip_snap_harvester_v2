from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.integrity import to_utc, validate_minute_bars
from snap_harvester.logging_utils import get_logger

__all__ = [
    "compute_tail_mfe_for_events",
    "load_minutes_bars_for_symbol",
]


def load_minutes_bars_for_symbol(
    symbol: str,
    data_cfg: dict,
    *,
    price_tolerance: float = 1e-6,
    logger: logging.Logger | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load 1m bars for a symbol using the config's bars_path_template.

    Parameters
    ----------
    symbol: str
        Symbol name used to format data_cfg["bars_path_template"].
    data_cfg: dict
        Config block under cfg["data"] containing bars_* keys.
    price_tolerance: float
        Tolerance forwarded to validate_minute_bars.
    logger: logging.Logger | None
        Optional logger for info/warn messages.
    start, end: pd.Timestamp | None
        Optional bounds for filtering the loaded bars.
    """
    template = data_cfg["bars_path_template"]
    time_col = data_cfg["bars_time_col"]
    open_col = data_cfg["bars_open_col"]
    high_col = data_cfg["bars_high_col"]
    low_col = data_cfg["bars_low_col"]
    close_col = data_cfg["bars_close_col"]

    path = Path(template.format(symbol=symbol))
    if not path.exists():
        raise FileNotFoundError(
            f"Bars file not found for symbol {symbol}: {path.resolve()} (check data.bars_path_template)"
        )

    if path.suffix == ".parquet":
        bars = pd.read_parquet(path)
    else:
        bars = pd.read_csv(path)

    bars[time_col] = to_utc(bars[time_col], name=f"{symbol} bars::{time_col}")
    bars = bars.sort_values(time_col).reset_index(drop=True)
    validate_minute_bars(
        bars,
        time_col=time_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        tolerance=price_tolerance,
    )

    if start is not None:
        bars = bars[bars[time_col] >= start].copy()
    if end is not None:
        bars = bars[bars[time_col] <= end].copy()

    if logger:
        logger.info(
            "Loaded %d bars for %s from %s (range: %s -> %s)",
            len(bars),
            symbol,
            path,
            bars[time_col].iloc[0] if not bars.empty else None,
            bars[time_col].iloc[-1] if not bars.empty else None,
        )
    return bars


def compute_tail_mfe_for_events(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    max_horizon_bars: int,
    time_col: str = "ts",
    event_time_col: str = "event_ts",
    entry_col: str = "entry_price",
    direction_col: str = "direction",
    atr_col: str = "atr",
    high_col: str = "high",
    low_col: str = "low",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute tail MFE/MAE and binary tail hits for each event using minute bars.

    Adds columns:
        mfe_r, mae_r, hit_5r, hit_10r, hit_20r, hit_40r
    """
    log = logger or get_logger("tail_mfe")
    if max_horizon_bars <= 0:
        raise ValueError("max_horizon_bars must be positive.")
    if bars.empty or events.empty:
        log.warning("Bars or events are empty; returning empty result.")
        return pd.DataFrame(
            columns=list(events.columns)
            + ["mfe_r", "mae_r", "hit_5r", "hit_10r", "hit_20r", "hit_40r"]
        )

    bars = bars.copy()
    bars[time_col] = pd.to_datetime(bars[time_col], utc=True, errors="coerce")
    if bars[time_col].isna().any():
        raise ValueError(f"{time_col}: failed to parse some bar timestamps.")
    bars = bars.sort_values(time_col).reset_index(drop=True)

    bar_times = bars[time_col].astype("int64").to_numpy()
    highs = bars[high_col].to_numpy(dtype=float)
    lows = bars[low_col].to_numpy(dtype=float)

    out_rows: list[dict] = []
    unmatched = 0
    skipped_empty = 0
    invalid_direction = 0

    for idx, event in events.iterrows():
        event_ts = pd.to_datetime(event[event_time_col], utc=True, errors="coerce")
        if pd.isna(event_ts):
            log.warning("Event %s has invalid timestamp; skipping.", idx)
            unmatched += 1
            continue

        event_ts_ns = int(event_ts.value)
        start_idx = np.searchsorted(bar_times, event_ts_ns, side="left")
        if start_idx >= len(bar_times):
            log.warning("Event at %s has no following bars; skipping.", event_ts)
            skipped_empty += 1
            continue

        if bar_times[start_idx] != event_ts_ns:
            log.warning(
                "Event at %s not aligned exactly; using next bar at %s",
                event_ts,
                pd.to_datetime(bar_times[start_idx], utc=True),
            )

        slice_end = min(start_idx + max_horizon_bars, len(bars))
        slice_high = highs[start_idx:slice_end]
        slice_low = lows[start_idx:slice_end]
        if slice_high.size == 0 or slice_low.size == 0:
            log.warning("Empty bar slice for event at %s; skipping.", event_ts)
            skipped_empty += 1
            continue

        direction = int(event[direction_col]) if pd.notna(event[direction_col]) else None
        if direction not in (1, -1):
            log.warning("Event at %s has invalid direction %r; skipping.", event_ts, event[direction_col])
            invalid_direction += 1
            continue

        entry_price = float(event[entry_col])
        r_unit = float(event[atr_col])
        if not np.isfinite(entry_price) or not np.isfinite(r_unit) or r_unit <= 0:
            log.warning("Event at %s has invalid entry/ATR; skipping.", event_ts)
            unmatched += 1
            continue

        if direction == 1:
            mfe_price = float(np.max(slice_high))
            mae_price = float(np.min(slice_low))
        else:
            mfe_price = float(np.min(slice_low))
            mae_price = float(np.max(slice_high))

        mfe_r = (mfe_price - entry_price) / r_unit * direction
        mae_r = (mae_price - entry_price) / r_unit * direction  # negative or zero is adverse

        row = event.to_dict()
        row["mfe_r"] = float(mfe_r)
        row["mae_r"] = float(mae_r)
        row["hit_5r"] = int(mfe_r >= 5.0)
        row["hit_10r"] = int(mfe_r >= 10.0)
        row["hit_20r"] = int(mfe_r >= 20.0)
        row["hit_40r"] = int(mfe_r >= 40.0)
        out_rows.append(row)

    if unmatched or skipped_empty or invalid_direction:
        log.info(
            "Tail MFE completed: kept=%d, unmatched_ts=%d, empty_slice=%d, invalid_dir=%d",
            len(out_rows),
            unmatched,
            skipped_empty,
            invalid_direction,
        )

    return pd.DataFrame(out_rows)


def _run_synthetic_example() -> None:
    ts0 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": [
                ts0,
                ts0 + pd.Timedelta(minutes=1),
                ts0 + pd.Timedelta(minutes=2),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 121.0, 105.0],
            "low": [99.0, 95.0, 104.0],
            "close": [100.0, 120.0, 104.0],
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "event_ts": [ts0],
            "entry_price": [100.0],
            "direction": [1],
            "atr": [1.0],
        }
    )
    out = compute_tail_mfe_for_events(
        bars=bars,
        events=events,
        max_horizon_bars=3,
        time_col="timestamp",
        event_time_col="event_ts",
    )
    assert not out.empty, "Synthetic run returned empty DataFrame"
    assert abs(out.loc[0, "mfe_r"] - 21.0) < 1e-9, "Unexpected MFE"
    assert out.loc[0, "hit_20r"] == 1, "20R hit should be flagged"
    assert out.loc[0, "hit_40r"] == 0, "40R hit should not be flagged"
    print("Synthetic Tail MFE example passed.")


if __name__ == "__main__":
    _run_synthetic_example()
