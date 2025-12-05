import os
import sys
import numpy as np
import pandas as pd
from typing import List, Generator, Tuple, Optional

# Fix path to allow imports from core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from snap_harvester.utils.ticks import get_tick_size, price_to_tick, tick_to_price

# ----------------------------------------------------------------------
# File Listing & Robust Parsing
# ----------------------------------------------------------------------

def list_tick_files(tick_dir: str) -> List[str]:
    """List CSV files under `tick_dir` sorted by name."""
    if not os.path.isdir(tick_dir):
        return []
    files = [
        os.path.join(tick_dir, f)
        for f in os.listdir(tick_dir)
        if f.lower().endswith(".csv")
    ]
    return sorted(files)


def _load_single_tick_csv(path: str) -> pd.DataFrame:
    """
    Load a single tick CSV and normalize columns safely.
    Handles various column name formats (Binance, generic, etc.).
    If the fast C-engine parser fails (e.g., malformed rows), fall back to a
    slower parser that skips bad lines instead of aborting the whole file.
    """
    # Optimized for aggTrades-derived ticks: timestamp,price,qty,is_buyer_maker
    usecols = ["timestamp", "price", "qty", "is_buyer_maker"]
    dtype = {
        "timestamp": "int64",
        "price": "float64",
        "qty": "float64",
        "is_buyer_maker": "int8",
    }

    try:
        chunks = pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype,
            low_memory=False,
            engine="c",
            memory_map=True,
            chunksize=5_000_000,
        )
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"Error reading {path} with optimized parser: {e}")
        try:
            chunks = pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip",
                usecols=usecols,
                dtype=dtype,
                chunksize=1_000_000,
            )
            df = pd.concat(chunks, ignore_index=True)
            print(f"[Warning] Retried {path} with on_bad_lines='skip'; shape={df.shape}")
        except Exception as e2:
            print(f"Error reading {path} with fallback parser: {e2}")
            return pd.DataFrame()

    if df.empty:
        return df

    # Timestamp normalization (auto-detect unit)
    ts_raw = df["timestamp"]
    ts_numeric = pd.to_numeric(ts_raw, errors="coerce").dropna()
    if ts_numeric.empty:
        return pd.DataFrame()

    max_ts = ts_numeric.max()
    unit = "ms"
    if 1e14 < max_ts < 1e17:
        unit = "us"
    elif max_ts >= 1e17:
        unit = "ns"

    df["ts"] = pd.to_datetime(ts_numeric.astype("int64"), unit=unit, utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])

    # Canonical columns
    df = df[["ts", "price", "qty", "is_buyer_maker"]].copy()

    # Bool normalization
    if df["is_buyer_maker"].dtype != bool:
        if df["is_buyer_maker"].dtype == object:
            s = df["is_buyer_maker"].astype(str).str.strip().str.lower()
            mapping = {"true": 1, "t": 1, "1": 1, "false": 0, "f": 0, "0": 0}
            s = s.map(mapping).fillna(0)
            df["is_buyer_maker"] = s.astype(int) != 0
        else:
            num = pd.to_numeric(df["is_buyer_maker"], errors="coerce").fillna(0)
            df["is_buyer_maker"] = num.astype(int) != 0

    # Drop duplicates and sort
    df = df.drop_duplicates(subset=["ts", "price", "qty", "is_buyer_maker"]).sort_values("ts")
    return df.reset_index(drop=True)


def resample_ticks_to_bars(
    ticks: pd.DataFrame,
    timeframe: str = "1min",
    symbol: Optional[str] = None,
    tick_size: Optional[float] = None,
) -> pd.DataFrame:
    """
    Aggregate ticks into OHLCV bars with buy/sell volume.
    """
    if ticks.empty:
        return pd.DataFrame()

    resolved_tick = tick_size
    if resolved_tick is None and symbol:
        resolved_tick = get_tick_size(symbol)

    df = ticks.copy()
    if resolved_tick is not None:
        # Vectorized price -> tick -> price snap
        price_arr = df["price"].to_numpy()
        tick_arr = np.rint(price_arr / resolved_tick).astype("int64")
        df["tick"] = tick_arr
        df["price"] = tick_arr.astype("float64") * resolved_tick

    df = df.set_index("ts")

    ohlc = df["price"].resample(timeframe).ohlc()
    vol = df["qty"].resample(timeframe).sum().rename("volume")

    # buy_qty = aggressor is buyer (is_buyer_maker=False)
    buy_qty = df.loc[~df["is_buyer_maker"], "qty"].resample(timeframe).sum().rename("buy_qty")
    # sell_qty = aggressor is seller (is_buyer_maker=True)
    sell_qty = df.loc[df["is_buyer_maker"], "qty"].resample(timeframe).sum().rename("sell_qty")

    bars = pd.concat([ohlc, vol, buy_qty, sell_qty], axis=1)
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars[["buy_qty", "sell_qty"]] = bars[["buy_qty", "sell_qty"]].fillna(0.0)
    bars = bars.sort_index()

    return bars.reset_index().rename(columns={"ts": "timestamp"})


# ----------------------------------------------------------------------
# Streaming Logic (Memory Safe)
# ----------------------------------------------------------------------

def stream_ticks_from_dir(tick_dir: str, chunk_days: int = 7) -> Generator[pd.DataFrame, None, None]:
    """
    Yields chunks of ticks (concatenated by date) to avoid memory overflow.
    """
    files = list_tick_files(tick_dir)
    if not files:
        return

    chunk_ticks = []
    current_date = None

    for file_path in files:
        df = _load_single_tick_csv(file_path)
        if df.empty:
            continue

        # Assign date group
        df['date'] = df['ts'].dt.floor('D')

        for date, group in df.groupby('date', observed=True):
            chunk_ticks.append(group)
            
            if current_date is None:
                current_date = date
            
            # Check accumulation size (approx by day count in chunk)
            # We just check if we have enough unique dates in the accumulator
            unique_days = pd.concat(chunk_ticks)['date'].nunique()
            
            if unique_days >= chunk_days:
                chunk_df = pd.concat(chunk_ticks, ignore_index=True)
                chunk_df = chunk_df.sort_values('ts').reset_index(drop=True)
                yield chunk_df
                chunk_ticks = []
                current_date = None

    # Final chunk
    if chunk_ticks:
        chunk_df = pd.concat(chunk_ticks, ignore_index=True)
        chunk_df = chunk_df.sort_values('ts').reset_index(drop=True)
        yield chunk_df


# ----------------------------------------------------------------------
# Convenience Loader
# ----------------------------------------------------------------------

def load_marketdata_as_bars(
    tick_dir: str,
    timeframe: str = "1min",
    chunk_days: int = 10,
    symbol: Optional[str] = None,
    tick_size: Optional[float] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Stream ticks from a directory and return a single bars DataFrame.
    Returns (bars_df, source_label).
    """
    bar_chunks = []
    for tick_chunk in stream_ticks_from_dir(tick_dir, chunk_days=chunk_days):
        if tick_chunk is None or tick_chunk.empty:
            continue
        bars = resample_ticks_to_bars(
            tick_chunk,
            timeframe=timeframe,
            symbol=symbol,
            tick_size=tick_size,
        )
        if bars.empty:
            continue
        bar_chunks.append(bars)

    if not bar_chunks:
        return pd.DataFrame(), "empty"

    full = pd.concat(bar_chunks).sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    return full, "stream"
