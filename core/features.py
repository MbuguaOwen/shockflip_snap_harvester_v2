from typing import Tuple

import numpy as np
import pandas as pd


def compute_orderflow_features(bars: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Add Q+, Q-, delta, imbalance, and z-scores to bars.

    Assumes bars has:
    - buy_qty
    - sell_qty
    """
    df = bars.copy()

    df["q_plus"] = df["buy_qty"].astype(float)
    df["q_minus"] = df["sell_qty"].astype(float)
    df["delta"] = df["q_plus"] - df["q_minus"]
    df["imbalance"] = (df["q_plus"] - df["q_minus"]) / (
        df["q_plus"] + df["q_minus"] + eps
    )

    return df


def rolling_zscore(series: pd.Series, window: int, eps: float = 1e-9) -> pd.Series:
    """Causal rolling z-score (includes current bar in window)."""
    roll_mean = series.rolling(window, min_periods=window).mean()
    roll_std = series.rolling(window, min_periods=window).std(ddof=0)
    z = (series - roll_mean) / (roll_std + eps)
    return z


def compute_atr(bars: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Compute ATR(window) using classic True Range definition."""
    df = bars.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr"] = tr.rolling(window, min_periods=window).mean()

    return df


def compute_donchian(bars: pd.DataFrame, window: int = 120, eps: float = 1e-9) -> pd.DataFrame:
    """Compute Donchian channel and location.

    - donchian_high: rolling max of high
    - donchian_low: rolling min of low
    - donchian_loc: (close - low) / (high - low + eps) in [0,1]
    - at_upper_extreme: bar makes new window high (high >= donchian_high)
    - at_lower_extreme: bar makes new window low (low <= donchian_low)
    """
    df = bars.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    donchian_high = high.rolling(window, min_periods=window).max()
    donchian_low = low.rolling(window, min_periods=window).min()

    df["donchian_high"] = donchian_high
    df["donchian_low"] = donchian_low

    rng = donchian_high - donchian_low
    df["donchian_loc"] = (close - donchian_low) / (rng + eps)

    df["at_upper_extreme"] = high >= donchian_high
    df["at_lower_extreme"] = low <= donchian_low

    return df


def add_core_features(
    bars: pd.DataFrame,
    z_window: int = 240,
    atr_window: int = 60,
    donchian_window: int = 120,
) -> pd.DataFrame:
    """Convenience: compute orderflow, imbalance z, ATR, Donchian."""
    df = compute_orderflow_features(bars)
    df["imbalance_z"] = rolling_zscore(df["imbalance"], window=z_window)
    df = compute_atr(df, window=atr_window)
    df = compute_donchian(df, window=donchian_window)
    return df


# --- H1: Prior order-flow sign -------------------------------------------------


def compute_prior_flow_sign(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    H1: Prior order-flow sign before the ShockFlip.

    We use a rolling sum of signed volume (delta) over a lookback window and
    collapse it to a sign in {-1, 0, +1}.

    Columns produced:
      - prior_flow_sum   : rolling sum of signed flow
      - prior_flow_sign  : sign(prior_flow_sum) in {-1, 0, +1}
    """
    out = df.copy()

    if "delta" in out.columns:
        flow = out["delta"].astype(float)
    elif "q_plus" in out.columns and "q_minus" in out.columns:
        flow = out["q_plus"].astype(float) - out["q_minus"].astype(float)
    else:
        raise KeyError(
            "compute_prior_flow_sign expected 'delta' or ('q_plus','q_minus') "
            "in the DataFrame."
        )

    roll = flow.rolling(window=window, min_periods=1).sum()
    out["prior_flow_sum"] = roll
    out["prior_flow_sign"] = np.sign(roll).astype(int)
    return out


# --- H2: Price vs flow divergence ---------------------------------------------


def compute_price_flow_divergence(
    df: pd.DataFrame,
    window: int = 60,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    H2: Price/flow divergence over a window.

    Intuition:
      - price_chg_window  : log-price change over 'window'
      - flow_chg_window   : rolling sum of signed flow over 'window'
      - price_flow_div    : z(price_chg_window) - z(flow_chg_window)

    Positive values ~ price "over-performs" flow.
    Negative values ~ flow "over-performs" price.
    """
    out = df.copy()

    if price_col not in out.columns:
        raise KeyError(
            f"compute_price_flow_divergence expected '{price_col}' column."
        )

    if "delta" in out.columns:
        flow = out["delta"].astype(float)
    elif "q_plus" in out.columns and "q_minus" in out.columns:
        flow = out["q_plus"].astype(float) - out["q_minus"].astype(float)
    else:
        raise KeyError(
            "compute_price_flow_divergence expected 'delta' or ('q_plus','q_minus')."
        )

    # Log price change over 'window'
    price = np.log(out[price_col].astype(float).clip(lower=1e-12))
    price_chg = price.diff(window)

    # Flow "change" over window = rolling sum
    flow_chg = flow.rolling(window=window, min_periods=1).sum()

    def zscore(s: pd.Series) -> pd.Series:
        m = s.mean()
        v = s.std(ddof=0)
        if not np.isfinite(v) or v == 0.0:
            # Degenerate case -> zero out to avoid NaNs everywhere
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - m) / v

    price_z = zscore(price_chg)
    flow_z = zscore(flow_chg)

    out["price_chg_window"] = price_chg
    out["flow_chg_window"] = flow_chg
    out["price_flow_div"] = price_z - flow_z
    return out


# --- H3: ATR percentile (vol regime) ------------------------------------------


def compute_atr_percentile(df: pd.DataFrame, window: int = 5000) -> pd.DataFrame:
    """
    H3: ATR percentile.

    For now we use a simple cross-sectional percentile over the whole sample:
      atr_pct in (0, 1].

    If you want strictly rolling percentiles later, we can swap this to a
    rolling-window rank; the column name can stay the same.
    """
    out = df.copy()

    if "atr" not in out.columns:
        raise KeyError("compute_atr_percentile expected 'atr' column to exist.")

    atr = out["atr"].astype(float)

    # Simple percentile over the entire sample [0,1].
    out["atr_pct"] = atr.rank(pct=True).astype(float)
    return out


def add_hypothesis_features(
    df: pd.DataFrame,
    prior_flow_window: int = 60,
    div_window: int = 60,
    atr_pct_window: int = 5000,
) -> pd.DataFrame:
    """
    Convenience helper to apply the H1–H3 feature transforms on top of the
    core features.

    This DOES NOT change ShockFlip detection. It only adds research columns.
    """
    out = compute_prior_flow_sign(df, window=prior_flow_window)
    out = compute_price_flow_divergence(out, window=div_window)
    out = compute_atr_percentile(out, window=atr_pct_window)
    return out
