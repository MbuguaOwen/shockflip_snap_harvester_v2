import os
import sys
import pandas as pd
from typing import List, Generator, Tuple

# Fix path to allow imports from core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

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
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}

    # 1. Timestamp -> 'ts'
    if "ts" in df.columns:
        pass
    elif "timestamp" in cols:
        df = df.rename(columns={cols["timestamp"]: "ts"})
    elif "time" in cols:
        df = df.rename(columns={cols["time"]: "ts"})
    else:
        # Skip file if critical columns missing
        return pd.DataFrame()

    # 2. Price -> 'price'
    if "price" not in cols:
        for alt in ("p", "last_price", "close", "trade_price"):
            if alt in cols:
                df = df.rename(columns={cols[alt]: "price"})
                break
    if "price" not in df.columns:
        return pd.DataFrame()

    # 3. Qty -> 'qty'
    if "qty" not in cols:
        for alt in ("quantity", "size", "amount", "vol", "volume"):
            if alt in cols:
                df = df.rename(columns={cols[alt]: "qty"})
                break
    if "qty" not in df.columns:
        return pd.DataFrame()

    # 4. Buyer Maker -> 'is_buyer_maker'
    if "is_buyer_maker" not in df.columns:
        for alt in ("isbuyermaker", "buyer_is_maker", "is_buyer_mkt_maker", "is_buyer_maker"):
            if alt in cols:
                df = df.rename(columns={cols[alt]: "is_buyer_maker"})
                break
    if "is_buyer_maker" not in df.columns:
        return pd.DataFrame()

    # 5. Timestamp Normalization
    ts = df["ts"]
    is_numeric = False
    try:
        is_numeric = pd.api.types.is_numeric_dtype(ts)
    except:
        is_numeric = ts.dtype.kind in ("i", "u", "f")

    if is_numeric:
        # Detect time unit: ms (~1e12) vs us (~1e15) vs ns (~1e18)
        max_ts = pd.to_numeric(ts, errors="coerce").dropna().max()
        unit = "ms"
        if pd.notnull(max_ts):
            if max_ts > 1e14 and max_ts < 1e17:
                unit = "us"
            elif max_ts >= 1e17:
                unit = "ns"
        df["ts"] = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True, errors="coerce")
    else:
        df["ts"] = pd.to_datetime(ts, utc=True, errors="coerce")

    df = df.dropna(subset=["ts"])

    # 6. Bool Normalization
    if df["is_buyer_maker"].dtype != bool:
        if df["is_buyer_maker"].dtype == object:
            s = df["is_buyer_maker"].astype(str).str.strip().str.lower()
            mapping = {"true": 1, "t": 1, "1": 1, "false": 0, "f": 0, "0": 0}
            s = s.map(mapping).fillna(0)
            df["is_buyer_maker"] = s.astype(int) != 0
        else:
            num = pd.to_numeric(df["is_buyer_maker"], errors="coerce").fillna(0)
            df["is_buyer_maker"] = num.astype(int) != 0

    return df


def resample_ticks_to_bars(ticks: pd.DataFrame, timeframe: str = "1min") -> pd.DataFrame:
    """
    Aggregate ticks into OHLCV bars with buy/sell volume.
    """
    if ticks.empty:
        return pd.DataFrame()

    df = ticks.set_index("ts")

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

def load_marketdata_as_bars(tick_dir: str, timeframe: str = "1min", chunk_days: int = 10) -> Tuple[pd.DataFrame, str]:
    """
    Stream ticks from a directory and return a single bars DataFrame.
    Returns (bars_df, source_label).
    """
    bar_chunks = []
    for tick_chunk in stream_ticks_from_dir(tick_dir, chunk_days=chunk_days):
        if tick_chunk is None or tick_chunk.empty:
            continue
        bars = resample_ticks_to_bars(tick_chunk, timeframe=timeframe)
        if bars.empty:
            continue
        bar_chunks.append(bars)

    if not bar_chunks:
        return pd.DataFrame(), "empty"

    full = pd.concat(bar_chunks).sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    return full, "stream"
