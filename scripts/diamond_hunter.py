#!/usr/bin/env python3
"""
Diamond Hunter v2.4
Stream-safe ShockFlip event annotator for microstructure physics studies.
- Streams ticks -> 1m bars
- Detects ShockFlip events
- Writes a rich events_annotated.csv with ATR-path stats (MFE) and context features
- Prints rel_vol decile thresholds (Diamond gate) as before
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from core.data_loader import stream_ticks_from_dir, resample_ticks_to_bars
from core.features import add_core_features, add_hypothesis_features
from core.shockflip_detector import ShockFlipConfig, detect_shockflip_signals
from snap_harvester.utils.ticks import get_tick_size

EPS = 1e-9
DEFAULT_HORIZONS = [6, 10, 20, 30, 60, 120, 240]
SNAP_THRESHOLDS_ATR = [0.5, 0.75, 1.0]
# Barrier label specs: list of (horizon, tp_r, sl_r)
BARRIER_SPECS = [(30, 3.0, 0.5)]
MIN_BARS_DEFAULT = 240
Z_WINDOW_DEFAULT = 240
DONCHIAN_WINDOW_DEFAULT = 120


def _compute_rel_vol(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Relative volume: bar volume / mean volume over window."""
    vol = df["buy_qty"].astype(float) + df["sell_qty"].astype(float)
    rolling_sum = vol.rolling(window=window, min_periods=1).sum()
    avg_vol = rolling_sum / float(window)
    return vol / (avg_vol + EPS)


def _compute_trend_context(df: pd.DataFrame, fast_span: int = 50, slow_span: int = 200) -> pd.DataFrame:
    """Add simple trend descriptors (fast/slow EMAs + slope proxy)."""
    out = df.copy()
    close = out["close"].astype(float)
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    out["trend_fast"] = fast
    out["trend_slow"] = slow
    out["trend_dir"] = np.sign(fast - slow).astype(int)

    # Simple slope proxy over ~4h (240 bars): log-price drift / bars
    drift = np.log(close.clip(lower=1e-12)) - np.log(close.shift(240).clip(lower=1e-12))
    out["trend_slope"] = drift / 240.0
    return out


def _barrier_col_name(h: int, tp_r: float, sl_r: float) -> str:
    return f"barrier_y_H{h}_R{tp_r}_sl{sl_r}".replace(".", "p")


def _compute_barrier_label(
    bars: pd.DataFrame,
    idx: int,
    side: int,
    atr: float,
    horizon: int,
    tp_r: float,
    sl_r: float,
) -> int:
    """1 if TP is hit before SL within horizon, else 0. TP wins ties."""
    n = len(bars)
    if idx >= n - 1 or not np.isfinite(atr) or atr <= 0:
        return 0

    entry_price = float(bars.iloc[idx]["close"])
    tp = entry_price + side * (tp_r * atr)
    sl = entry_price - side * (sl_r * atr)

    max_idx = min(idx + horizon, n - 1)
    for j in range(idx + 1, max_idx + 1):
        bar = bars.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])
        if side == 1:
            tp_hit = high >= tp
            sl_hit = low <= sl
        else:
            tp_hit = low <= tp
            sl_hit = high >= sl

        if tp_hit and sl_hit:
            return 1  # TP wins tie
        if tp_hit:
            return 1
        if sl_hit:
            return 0
    return 0


def _compute_divergence_score(df: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """Crude divergence score: flow vs short-term price move."""
    price = df["close"].astype(float)
    flow_z = df.get("imbalance_z", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    ret = price.pct_change(lookback).fillna(0.0)
    std = price.rolling(lookback * 5, min_periods=1).std().replace(0.0, np.nan)
    score = -flow_z * (ret / (std + EPS))
    return score.clip(lower=0.0).fillna(0.0)


def compute_mfe_multi(
    bars: pd.DataFrame,
    idx: int,
    side: int,
    atr_col: str,
    horizons: list[int],
) -> dict[str, float]:
    """
    Compute max favourable excursion over multiple horizons in ATR units.
    Looks strictly forward (idx, idx+H].
    """
    entry_bar = bars.iloc[idx]
    entry_price = float(entry_bar["close"])
    atr0 = float(entry_bar[atr_col])
    if not np.isfinite(atr0) or atr0 <= 0:
        return {f"mfe{H}_atr": 0.0 for H in horizons}

    out: dict[str, float] = {}
    n_bars = len(bars)

    for H in horizons:
        end_idx = min(idx + 1 + H, n_bars)
        window = bars.iloc[idx + 1 : end_idx]

        if window.empty:
            mfe_atr = 0.0
        else:
            if side > 0:
                fav_moves = window["high"] - entry_price
            else:
                fav_moves = entry_price - window["low"]
            max_fav = float(max(fav_moves.max(), 0.0))
            mfe_atr = max_fav / atr0

        out[f"mfe{H}_atr"] = mfe_atr

    return out


def build_features(bars: pd.DataFrame, sf_cfg: ShockFlipConfig) -> pd.DataFrame:
    """Compute all context features needed for event annotation."""
    feats = add_core_features(
        bars,
        z_window=sf_cfg.z_window,
        atr_window=60,
        donchian_window=sf_cfg.location_filter["donchian_window"],
    )
    feats = add_hypothesis_features(feats, prior_flow_window=60, div_window=60, atr_pct_window=5000)
    feats["rel_vol"] = _compute_rel_vol(feats, window=60)
    feats["shock_intensity_z"] = np.abs(feats.get("imbalance_z", 0.0))
    feats["range_atr"] = (feats["high"] - feats["low"]) / (feats["atr"] + EPS)
    feats["shock_intensity_z"] = np.maximum(feats["shock_intensity_z"], feats["range_atr"])
    feats["div_score"] = _compute_divergence_score(feats, lookback=6)
    feats = _compute_trend_context(feats, fast_span=50, slow_span=200)
    feats["stall_flag"] = (
        feats["close"].pct_change(5).abs() < 0.0015
    ) & (np.abs(feats.get("imbalance_z", 0.0)) > 1.0)
    feats["stall_flag"] = feats["stall_flag"].astype(int)
    feats = detect_shockflip_signals(feats, sf_cfg)
    return feats


def get_chunk_events(bars: pd.DataFrame, cfg_overrides: dict, symbol: str, horizons: list[int]) -> list[dict]:
    """Extract annotated ShockFlip events from a bars chunk."""
    sf_cfg = ShockFlipConfig(
        source="imbalance",
        z_window=cfg_overrides["z_window"],
        z_band=cfg_overrides["z_band"],
        jump_band=cfg_overrides["jump_band"],
        persistence_bars=cfg_overrides["persistence"],
        persistence_ratio=0.5,
        dynamic_thresholds={"enabled": False},
        location_filter={"donchian_window": cfg_overrides["donchian_window"], "require_extreme": True},
    )

    feats = build_features(bars, sf_cfg)
    events: list[dict] = []

    for idx, row in feats.iterrows():
        side = int(row.get("shockflip_signal", 0))
        if side == 0:
            continue

        atr = float(row.get("atr", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            continue

        mfe_vals = compute_mfe_multi(feats, idx, side, atr_col="atr", horizons=horizons)

        trend_dir = int(row.get("trend_dir", 0))
        trend_aligned = 1 if trend_dir != 0 and np.sign(side) == np.sign(trend_dir) else 0

        event = {
            "timestamp": row["timestamp"],
            "idx": idx,
            "symbol": symbol,
            "side": side,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "atr": atr,
            "atr_pct": row.get("atr_pct"),
            "rel_vol": float(row.get("rel_vol", np.nan)),
            "imbalance": row.get("imbalance"),
            "imbalance_z": row.get("imbalance_z"),
            "shockflip_z": row.get("imbalance_z"),
            "shock_intensity_z": row.get("shock_intensity_z"),
            "range_atr": float(row.get("range_atr", np.nan)),
            "donchian_loc": row.get("donchian_loc"),
            "price_flow_div": row.get("price_flow_div"),
            "prior_flow_sign": row.get("prior_flow_sign"),
            "div_score": float(row.get("div_score", np.nan)),
            "trend_dir": trend_dir,
            "trend_aligned": trend_aligned,
            "trend_slope": float(row.get("trend_slope", np.nan)),
            "stall_flag": int(row.get("stall_flag", 0)),
        }
        event.update(mfe_vals)
        for H in horizons:
            mfe_col = f"mfe{H}_atr"
            mfe_val = mfe_vals.get(mfe_col, 0.0)
            for thr in SNAP_THRESHOLDS_ATR:
                label_name = f"did_snap_H{H}_K{thr}".replace(".", "_")
                event[label_name] = int(mfe_val >= thr)
        # Barrier labels (TP before SL)
        for (bh, btp, bsl) in BARRIER_SPECS:
            col = _barrier_col_name(bh, btp, bsl)
            event[col] = _compute_barrier_label(feats, idx, side, atr, horizon=bh, tp_r=btp, sl_r=bsl)
        events.append(event)
    return events


def analyze_diamonds(all_events_df: pd.DataFrame, out_dir: str, min_n: int = 10) -> None:
    """Print rel_vol thresholds and save annotated events."""
    if all_events_df.empty:
        print("[Diamond Hunter] No events found.")
        return

    ev = all_events_df.copy()
    ev["did_snap"] = ev.get("did_snap_0_5", ev.get("did_snap", 0))

    rel_vol_90 = ev["rel_vol"].quantile(0.90)
    rel_vol_80 = ev["rel_vol"].quantile(0.80)

    for col in ["shock_intensity_z", "div_score", "rel_vol"]:
        try:
            ev[f"{col}_decile"] = pd.qcut(ev[col].rank(method="first"), 10, labels=False, duplicates="drop")
        except Exception:
            ev[f"{col}_decile"] = 0

    rows = []
    base_rate = ev["did_snap"].mean() * 100
    rows.append({"bucket": "BASELINE", "n": len(ev), "snap_rate": round(base_rate, 2), "lift": 0.0})

    for feat in ["shock_intensity_z", "div_score", "rel_vol"]:
        for dec in [8, 9]:
            sel = ev[ev[f"{feat}_decile"] >= dec]
            if len(sel) < min_n:
                continue
            rate = sel["did_snap"].mean() * 100
            rows.append(
                {
                    "bucket": f"{feat} >= Decile {dec}",
                    "n": len(sel),
                    "snap_rate": round(rate, 2),
                    "lift": round(rate - base_rate, 2),
                }
            )

    results = pd.DataFrame(rows).sort_values("snap_rate", ascending=False)

    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "diamond_candidates.csv"), index=False)
    ev.to_csv(os.path.join(out_dir, "events_annotated.csv"), index=False)

    print("\n--- DIAMOND CANDIDATES ---")
    print(results.to_string(index=False))

    print(f"\n\n>>> THE MAGIC NUMBERS <<<")
    print(f"Decile 9 Threshold (Rel Vol): {rel_vol_90:.4f}")
    print(f"Decile 8 Threshold (Rel Vol): {rel_vol_80:.4f}")
    print(f"Use {rel_vol_90:.4f} in your config to filter for the 'Diamond' trades.")


def main():
    p = argparse.ArgumentParser(description="Diamond Hunter v2.4 – ShockFlip event annotator.")
    p.add_argument("--tick_dir", required=True, help="Tick data directory (expects CSVs).")
    p.add_argument("--out", required=True, help="Output directory (events_annotated.csv, diamond_candidates.csv).")
    p.add_argument("--z_band", type=float, default=2.0)
    p.add_argument(
        "--z_window",
        type=int,
        default=Z_WINDOW_DEFAULT,
        help=f"Z-score window for shockflip detection (default: {Z_WINDOW_DEFAULT}).",
    )
    p.add_argument("--jump_band", type=float, default=2.5)
    p.add_argument("--persistence", type=int, default=4)
    p.add_argument(
        "--donchian_window",
        type=int,
        default=DONCHIAN_WINDOW_DEFAULT,
        help=f"Donchian window for location filter (default: {DONCHIAN_WINDOW_DEFAULT}).",
    )
    p.add_argument(
        "--min_bars",
        type=int,
        default=MIN_BARS_DEFAULT,
        help=f"Minimum number of 1m bars required to run detection (default: {MIN_BARS_DEFAULT}).",
    )
    p.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated list of horizons (bars) for MFE computation, e.g. '6,10,20,30,60'",
    )
    args = p.parse_args()

    if args.horizons:
        try:
            horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"Invalid --horizons {args.horizons!r}, must be comma-separated ints")
    else:
        horizons = DEFAULT_HORIZONS
    if not horizons:
        horizons = DEFAULT_HORIZONS

    cfg_overrides = {
        "z_band": args.z_band,
        "z_window": args.z_window,
        "jump_band": args.jump_band,
        "persistence": args.persistence,
        "donchian_window": args.donchian_window,
    }
    all_events: list[dict] = []
    symbol = os.path.basename(os.path.abspath(args.tick_dir)).upper()
    try:
        tick_size = get_tick_size(symbol)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Diamond Hunter v2.4 running on {args.tick_dir}...")
    for tick_chunk in stream_ticks_from_dir(args.tick_dir, chunk_days=10):
        bars = resample_ticks_to_bars(
            tick_chunk,
            timeframe="1min",
            symbol=symbol,
            tick_size=tick_size,
        )
        if len(bars) < max(args.min_bars, 1):
            continue
        events = get_chunk_events(bars, cfg_overrides, symbol=symbol, horizons=horizons)
        all_events.extend(events)

    if all_events:
        analyze_diamonds(pd.DataFrame(all_events), args.out)
    else:
        print("No events found.")


if __name__ == "__main__":
    main()
