from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from snap_harvester.integrity import (
    align_events_to_bars,
    drop_duplicates_with_log,
    to_utc,
    validate_minute_bars,
    validate_risk_price_scale,
)
from snap_harvester.logging_utils import get_logger


def _load_events(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw_paths = data_cfg["events_path"]
    paths = list(raw_paths) if isinstance(raw_paths, (list, tuple)) else [raw_paths]

    frames = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Events file not found: {path.resolve()}")
        frames.append(pd.read_csv(path))

    if not frames:
        raise ValueError("No event files provided in config.data.events_path")

    df = pd.concat(frames, ignore_index=True)

    time_col = data_cfg["events_time_col"]
    sym_col = data_cfg["events_symbol_col"]
    side_col = data_cfg["events_side_col"]
    price_col = data_cfg["events_price_col"]
    atr_col = data_cfg["events_atr_col"]

    required = [time_col, sym_col, side_col, price_col, atr_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Events file missing required columns: {missing}")

    df[time_col] = to_utc(df[time_col], name="events.timestamp")
    df[side_col] = pd.to_numeric(df[side_col], errors="coerce").astype("Int64")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[atr_col] = pd.to_numeric(df[atr_col], errors="coerce")

    invalid_side = ~df[side_col].isin([1, -1])
    if invalid_side.any():
        bad = df.loc[invalid_side, side_col].unique()
        raise ValueError(f"Events contain non +/-1 side values: {bad}")

    if df[[price_col, atr_col]].isna().any().any():
        raise ValueError("Events contain null price/atr after coercion.")

    symbols = data_cfg.get("symbols")
    if symbols:
        df = df[df[sym_col].isin(symbols)].copy()

    df = drop_duplicates_with_log(df, subset=[sym_col, time_col, side_col], logger=logger, label="events")
    df = df.sort_values([sym_col, time_col]).reset_index(drop=True)
    return df


def _load_bars_for_symbol(symbol: str, cfg: dict, price_tolerance: float) -> pd.DataFrame:
    data_cfg = cfg["data"]
    template = data_cfg["bars_path_template"]
    path = Path(template.format(symbol=symbol))

    if not path.exists():
        raise FileNotFoundError(
            f"Bars file not found for symbol {symbol}: {path.resolve()} (check data.bars_path_template)"
        )

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    time_col = data_cfg["bars_time_col"]
    open_col = data_cfg["bars_open_col"]
    high_col = data_cfg["bars_high_col"]
    low_col = data_cfg["bars_low_col"]
    close_col = data_cfg["bars_close_col"]

    df[time_col] = to_utc(df[time_col], name=f"{symbol} bars::timestamp")
    df = df.sort_values(time_col).reset_index(drop=True)
    validate_minute_bars(
        df,
        time_col=time_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        tolerance=price_tolerance,
    )
    return df


def _simulate_snap_trade(
    event: pd.Series,
    bars: pd.DataFrame,
    cfg: dict,
    time_col: str,
    bars_time_col: str,
    side_col: str,
    price_col: str,
    atr_col: str,
    label_col: str,
    r_col: str,
    mfe_col: str,
    mae_col: str,
) -> Dict[str, Any]:
    """Simulate one Snap Harvester trade path on 1m OHLCV."""
    risk_cfg = cfg["risk"]
    risk_k_atr = float(risk_cfg["risk_k_atr"])
    be_k_atr = float(risk_cfg.get("be_k_atr", 0.0))
    tp_k_atr = float(risk_cfg.get("tp_k_atr", 0.0))
    sl_r_mult = float(risk_cfg.get("sl_r_multiple", 1.0))
    tp_r_mult = risk_cfg.get("tp_r_multiple")
    be_r_mult = risk_cfg.get("be_r_multiple")
    horizon = int(risk_cfg["horizon_bars"])
    target_r = risk_cfg.get("target_r_multiple")

    side = int(event[side_col])
    if side not in (1, -1):
        raise ValueError(f"side must be +1 or -1, got {side!r}")

    dir_sign = 1 if side > 0 else -1

    event_time = event.get("event_bar_time", event[time_col])
    t0 = pd.to_datetime(event_time, utc=True, errors="coerce")
    entry = float(event[price_col])
    atr = float(event[atr_col])
    if atr <= 0:
        raise ValueError("ATR must be positive")

    risk_dist = risk_k_atr * atr
    sl_dist = risk_dist * sl_r_mult
    if be_r_mult is not None:
        be_dist = risk_dist * float(be_r_mult)
    else:
        be_dist = be_k_atr * atr
    if tp_r_mult is not None:
        tp_dist = risk_dist * float(tp_r_mult)
    else:
        tp_dist = tp_k_atr * atr

    sl_price = entry - dir_sign * sl_dist
    be_price = entry + dir_sign * be_dist
    tp_price = entry + dir_sign * tp_dist

    if target_r is None:
        target_r = tp_dist / max(risk_dist, 1e-8)
    else:
        target_r = float(target_r)

    # Slice forward bars
    bars = bars.sort_values(bars_time_col)
    start_idx = bars[bars_time_col].searchsorted(t0, side="left")
    slice_df = bars.iloc[start_idx : start_idx + horizon].copy()
    if slice_df.empty:
        return {
            label_col: 0,
            r_col: 0.0,
            mfe_col: 0.0,
            mae_col: 0.0,
            "exit_time": pd.NaT,
            "hit_be": False,
            "hit_tp": False,
            "hit_sl": False,
        }

    high_col = cfg["data"]["bars_high_col"]
    low_col = cfg["data"]["bars_low_col"]
    close_col = cfg["data"]["bars_close_col"]

    state = "open"
    hit_be = False
    hit_tp = False
    hit_sl = False

    mfe_r = 0.0
    mae_r = 0.0
    r_final = 0.0
    exit_time = slice_df[bars_time_col].iloc[-1]

    for _, bar in slice_df.iterrows():
        high = float(bar[high_col])
        low = float(bar[low_col])
        close = float(bar[close_col])
        t = bar[bars_time_col]

        if dir_sign == 1:
            fav = (high - entry) / max(risk_dist, 1e-8)
            adv = (low - entry) / max(risk_dist, 1e-8)
            cross_tp = high >= tp_price
            cross_be = high >= be_price
            cross_sl = low <= sl_price
            cross_be_stop = low <= entry  # when BE is already locked
        else:
            fav = (entry - low) / max(risk_dist, 1e-8)
            adv = (entry - high) / max(risk_dist, 1e-8)
            cross_tp = low <= tp_price
            cross_be = low <= be_price
            cross_sl = high >= sl_price
            cross_be_stop = high >= entry

        mfe_r = max(mfe_r, fav)
        mae_r = min(mae_r, adv)

        if state == "open":
            if cross_tp:
                hit_tp = True
                r_final = (tp_price - entry) * dir_sign / max(risk_dist, 1e-8)
                exit_time = t
                state = "closed"
                break
            if cross_sl:
                hit_sl = True
                r_final = (sl_price - entry) * dir_sign / max(risk_dist, 1e-8)
                exit_time = t
                state = "closed"
                break
            if cross_be:
                hit_be = True
                state = "be_locked"
        elif state == "be_locked":
            if cross_tp:
                hit_tp = True
                r_final = (tp_price - entry) * dir_sign / max(risk_dist, 1e-8)
                exit_time = t
                state = "closed"
                break
            if cross_be_stop:
                hit_sl = True
                r_final = 0.0  # stopped at breakeven
                exit_time = t
                state = "closed"
                break

    if state != "closed":
        # Exit at horizon close, respecting BE (no losses after BE)
        close = float(slice_df[close_col].iloc[-1])
        if dir_sign == 1:
            pnl_r = (close - entry) / max(risk_dist, 1e-8)
        else:
            pnl_r = (entry - close) / max(risk_dist, 1e-8)
        if hit_be and pnl_r < 0.0:
            pnl_r = 0.0
        r_final = pnl_r
        exit_time = slice_df[bars_time_col].iloc[-1]

    y_swing = int(r_final >= target_r)

    return {
        label_col: y_swing,
        r_col: float(r_final),
        mfe_col: float(mfe_r),
        mae_col: float(mae_r),
        "exit_time": exit_time,
        "hit_be": bool(hit_be),
        "hit_tp": bool(hit_tp),
        "hit_sl": bool(hit_sl),
    }


def build_meta_dataset(cfg: dict, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Build a meta dataset with path-based labels for Snap Harvester."""
    logger = logger or get_logger("meta_builder")
    integrity_cfg = cfg.get("integrity", {})

    price_tolerance = float(integrity_cfg.get("price_tolerance", 1e-6))
    max_snap_minutes = int(integrity_cfg.get("event_snap_minutes", 1))
    drop_outside = bool(integrity_cfg.get("drop_misaligned_events", False))
    max_risk_to_price_ratio = float(integrity_cfg.get("max_risk_to_price_ratio", 10.0))

    events = _load_events(cfg, logger=logger)
    data_cfg = cfg["data"]
    meta_cfg = cfg.get("meta", {})

    time_col = data_cfg["events_time_col"]
    sym_col = data_cfg["events_symbol_col"]
    side_col = data_cfg["events_side_col"]
    price_col = data_cfg["events_price_col"]
    atr_col = data_cfg["events_atr_col"]
    bars_time_col = data_cfg["bars_time_col"]
    bars_high_col = data_cfg["bars_high_col"]
    bars_low_col = data_cfg["bars_low_col"]

    label_col = meta_cfg.get("label_col", "y_swing_1p5")
    r_col = meta_cfg.get("r_multiple_col", "r_final")
    mfe_col = meta_cfg.get("mfe_col", "mfe_r")
    mae_col = meta_cfg.get("mae_col", "mae_r")

    symbols = sorted(events[sym_col].unique())
    bars_by_symbol: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        bars_by_symbol[sym] = _load_bars_for_symbol(sym, cfg, price_tolerance=price_tolerance)

    events_aligned = align_events_to_bars(
        events=events,
        bars_by_symbol=bars_by_symbol,
        time_col=time_col,
        sym_col=sym_col,
        price_col=price_col,
        bars_time_col=bars_time_col,
        bars_low_col=bars_low_col,
        bars_high_col=bars_high_col,
        tolerance=price_tolerance,
        max_snap_minutes=max_snap_minutes,
        drop_outside=drop_outside,
        logger=logger,
    )
    events_aligned = events_aligned.sort_values([sym_col, time_col]).reset_index(drop=True)

    # Ensure scales make sense before simulation
    validate_risk_price_scale(
        events_aligned,
        price_col=price_col,
        atr_col=atr_col,
        risk_k_atr=float(cfg["risk"]["risk_k_atr"]),
        max_risk_to_price_ratio=max_risk_to_price_ratio,
    )

    risk_profile = cfg.get("active_risk_profile", None)
    if risk_profile is None and "risk_profile" in cfg:
        risk_profile = cfg["risk_profile"]

    out_rows = []
    for idx, event in tqdm(
        events_aligned.iterrows(),
        total=len(events_aligned),
        desc="Building meta dataset",
    ):
        sym = event[sym_col]
        bars = bars_by_symbol[sym]

        info = _simulate_snap_trade(
            event=event,
            bars=bars,
            cfg=cfg,
            time_col=time_col,
            bars_time_col=bars_time_col,
            side_col=side_col,
            price_col=price_col,
            atr_col=atr_col,
            label_col=label_col,
            r_col=r_col,
            mfe_col=mfe_col,
            mae_col=mae_col,
        )
        row = event.to_dict()
        row["event_id"] = f"{sym}-{pd.Timestamp(event[time_col]).isoformat()}-{idx}"
        if risk_profile is not None:
            row["risk_profile"] = risk_profile
        row.update(info)
        out_rows.append(row)

    meta_df = pd.DataFrame(out_rows)
    if r_col in meta_df.columns:
        meta_df["y_win_R4_SL2p5"] = (meta_df[r_col] > 0).astype(int)
    return meta_df
