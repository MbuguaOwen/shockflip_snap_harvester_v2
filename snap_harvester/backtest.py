from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def _aggregate_stats(df: pd.DataFrame, r_col: str, sym_col: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {"overall": {}, "by_symbol": {}}

    if df.empty:
        res["overall"] = {
            "n_trades": 0,
            "hit_rate": 0.0,
            "avg_R": 0.0,
            "std_R": 0.0,
            "sum_R": 0.0,
        }
        return res

    r = df[r_col].astype(float).values
    res["overall"] = {
        "n_trades": int(r.size),
        "hit_rate": float((r > 0).mean()),
        "avg_R": float(r.mean()),
        "std_R": float(r.std(ddof=0)),
        "sum_R": float(r.sum()),
    }

    for sym, d_sym in df.groupby(sym_col):
        r_s = d_sym[r_col].astype(float).values
        res["by_symbol"][str(sym)] = {
            "n_trades": int(r_s.size),
            "hit_rate": float((r_s > 0).mean()) if r_s.size > 0 else 0.0,
            "avg_R": float(r_s.mean()) if r_s.size > 0 else 0.0,
            "std_R": float(r_s.std(ddof=0)) if r_s.size > 0 else 0.0,
            "sum_R": float(r_s.sum()) if r_s.size > 0 else 0.0,
        }

    return res


def run_backtest(df_oos: pd.DataFrame, cfg: dict) -> Dict[str, Any]:
    """Evaluate Snap Harvester on an OOS slice."""
    meta_cfg = cfg["meta"]
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})

    r_col = meta_cfg["r_multiple_col"]
    sym_col = data_cfg["events_symbol_col"]
    p_col = "p_hat"
    thresholds = eval_cfg.get("prob_thresholds", [0.5])

    out: Dict[str, Any] = {}

    # Naive: no ML filtering, all ShockFlips
    out["naive"] = _aggregate_stats(df_oos, r_col=r_col, sym_col=sym_col)

    # ML-routed variants
    ml_res: Dict[str, Any] = {}
    if p_col in df_oos.columns:
        for thr in thresholds:
            d_thr = df_oos[df_oos[p_col] >= float(thr)]
            ml_res[str(thr)] = _aggregate_stats(d_thr, r_col=r_col, sym_col=sym_col)
    out["ml_routing"] = ml_res

    return out
