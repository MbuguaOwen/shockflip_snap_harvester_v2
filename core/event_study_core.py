from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from .progress import get_progress


@dataclass
class EventStudyConfig:
    horizons_long: Sequence[int]
    horizons_short: Sequence[int]
    baseline_n_random: int = 2000


def _compute_horizon_returns(
    bars: pd.DataFrame,
    idx: int,
    side: int,
    horizons: Sequence[int],
) -> Dict[int, float]:
    """Compute horizon returns for a single event.

    Returns dict: horizon -> return.
    """
    res: Dict[int, float] = {}
    entry_price = float(bars["close"].iloc[idx])
    n = len(bars)

    for h in horizons:
        j = idx + h
        if j >= n:
            res[h] = np.nan
            continue
        exit_price = float(bars["close"].iloc[j])
        ret = side * (exit_price - entry_price) / entry_price
        res[h] = float(ret)

    return res


def _compute_baseline_returns(
    bars: pd.DataFrame,
    horizons: Sequence[int],
    n_samples: int,
    seed: int = 123,
) -> Dict[int, np.ndarray]:
    """Unconditional baseline: random timestamps across series.

    For each horizon H:
    - Long baseline: ret = (P[t+H] - P[t]) / P[t]
    - Short baseline: ret = -(P[t+H] - P[t]) / P[t]
    """
    rng = np.random.default_rng(seed)
    n = len(bars)

    baseline: Dict[int, np.ndarray] = {}

    for h in horizons:
        max_start = n - h - 1
        if max_start <= 0:
            baseline[h] = np.array([])
            continue
        idxs = rng.integers(0, max_start, size=n_samples)
        entry = bars["close"].to_numpy()[idxs]
        exit_long = bars["close"].to_numpy()[idxs + h]
        ret_long = (exit_long - entry) / entry
        baseline[h] = ret_long

    return baseline


def run_event_study(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cfg: EventStudyConfig,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run event study on ShockFlip signals.

    Returns:
    - events_df: per-event returns by horizon
    - summary_df: per-horizon stats (mean, t, baseline mean, lift)
    """
    df = features.reset_index(drop=True)
    bars = bars.reset_index(drop=True)

    signal = df.get("shockflip_signal")
    if signal is None:
        raise ValueError("shockflip_signal not found in features")

    events: List[Dict] = []

    long_h = list(cfg.horizons_long)
    short_h = list(cfg.horizons_short)

    all_horizons = sorted(set(long_h + short_h))

    p = get_progress(progress, total=len(df), desc="Event study")
    for i in range(len(df)):
        p.update(1)
        side = int(signal.iloc[i])
        if side == 0:
            continue

        # Per-event z at the same index; default to NaN series if missing
        z_series = df["shockflip_z"] if "shockflip_z" in df.columns else pd.Series(np.nan, index=df.index)

        base = {
            "idx": i,
            "timestamp": df["timestamp"].iloc[i],
            "side": side,
            "shockflip_z": float(z_series.iloc[i]),
        }

        # H1–H3: add pre-context features if present
        for col in ["prior_flow_sign", "price_flow_div", "atr_pct"]:
            if col in df.columns:
                base[col] = df[col].iloc[i]

        if side == 1:
            ret_map = _compute_horizon_returns(bars, i, side=1, horizons=long_h)
            for h, r in ret_map.items():
                base[f"ret_h{h}"] = r
        else:
            ret_map = _compute_horizon_returns(bars, i, side=-1, horizons=short_h)
            for h, r in ret_map.items():
                base[f"ret_h{h}"] = r

        events.append(base)

    p.close()
    events_df = pd.DataFrame(events)

    # Handle zero-event case gracefully: return empty summary with columns
    if events_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "horizon",
                "side",
                "n",
                "mean_event",
                "t_event",
                "mean_baseline",
                "lift",
            ]
        )
        return events_df, summary_df

    # Baseline stats
    baseline = _compute_baseline_returns(
        bars,
        horizons=all_horizons,
        n_samples=cfg.baseline_n_random,
    )

    rows = []
    for h in all_horizons:
        if h in long_h:
            mask = events_df["side"] == 1
        else:
            mask = events_df["side"] == -1

        col = f"ret_h{h}"
        if col not in events_df.columns:
            continue

        r = events_df.loc[mask, col].dropna()
        if r.empty:
            continue

        mean_evt = float(r.mean())
        std_evt = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        t_evt = mean_evt / (std_evt / np.sqrt(len(r))) if len(r) > 1 and std_evt > 0 else float("nan")

        base_ret = baseline.get(h, np.array([]))
        mean_base = float(base_ret.mean()) if base_ret.size > 0 else float("nan")
        lift = mean_evt - mean_base if np.isfinite(mean_base) else float("nan")

        rows.append(
            dict(
                horizon=h,
                side="long" if h in long_h else "short",
                n=len(r),
                mean_event=mean_evt,
                t_event=t_evt,
                mean_baseline=mean_base,
                lift=lift,
            )
        )

    summary_df = pd.DataFrame(rows)

    return events_df, summary_df
