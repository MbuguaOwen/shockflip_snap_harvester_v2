from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from .barriers import FeesConfig, RiskConfig, build_barriers, compute_trade_pnl, enforce_tp_sl_invariants
from .features import add_core_features, add_hypothesis_features
from .meta_runner import MetaRunnerConfig, MetaRunnerRouter, add_time_features
from .shockflip_detector import ShockFlipConfig, detect_shockflip_signals
from .progress import get_progress

# -------------------------------------------------------------------------
# CONFIGURATION MODELS
# -------------------------------------------------------------------------

@dataclass
class FiltersConfig:
    min_relative_volume: Optional[float] = None
    min_divergence: Optional[float] = None
    vol_regime_low: Optional[float] = None
    vol_regime_high: Optional[float] = None
    require_counter_trend: bool = False
    require_positive_divergence: bool = False
    snap_filters: Optional[Dict[str, Any]] = None
    
    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class BacktestConfig:
    symbol: str
    tick_dir: str
    timeframe: str
    fees: FeesConfig
    slippage_bp: float
    risk: RiskConfig
    shockflip: ShockFlipConfig
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    meta_runner: Optional[MetaRunnerConfig] = None
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    
    # Management Config (The Zombie Kit)
    mfe_breakeven_r: Optional[float] = None  # Lock BE after this many R
    time_stop_bars: Optional[int] = None     # Kill trade if not profitable by this bar
    time_stop_r: Optional[float] = None      # Profit threshold required to survive time stop
    
    _debug: bool = False
    _progress: bool = True

# -------------------------------------------------------------------------
# FEATURE PREPARATION & FILTERING
# -------------------------------------------------------------------------

def _compute_relative_volume(bars: pd.DataFrame, window: int = 60) -> pd.Series:
    vol = bars["buy_qty"] + bars["sell_qty"]
    rolling_sum = vol.rolling(window=window, min_periods=1).sum()
    avg_vol = rolling_sum / float(window)
    return vol / (avg_vol + 1e-9)


def _add_trend_context(df: pd.DataFrame, fast_span: int = 50, slow_span: int = 200) -> pd.DataFrame:
    """Add fast/slow EMA trend direction and slope proxy."""
    out = df.copy()
    close = out["close"].astype(float)
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    out["trend_fast"] = fast
    out["trend_slow"] = slow
    out["trend_dir"] = np.sign(fast - slow).astype(int)

    drift = np.log(close.clip(lower=1e-12)) - np.log(close.shift(240).clip(lower=1e-12))
    out["trend_slope"] = drift / 240.0
    return out


def _compute_divergence_score(df: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """Crude divergence score: flow vs short-term price move."""
    price = df["close"].astype(float)
    flow_z = df.get("imbalance_z", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    ret = price.pct_change(lookback).fillna(0.0)
    std = price.rolling(lookback * 5, min_periods=1).std().replace(0.0, np.nan)
    score = -flow_z * (ret / (std + 1e-9))
    return score.clip(lower=0.0).fillna(0.0)

def apply_entry_filters(df: pd.DataFrame, cfg: FiltersConfig) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if cfg.min_relative_volume is not None:
        if "rel_vol" in df.columns:
            mask &= (df["rel_vol"] >= float(cfg.min_relative_volume))
    if cfg.min_divergence is not None:
        if "price_flow_div" in df.columns:
            mask &= (df["price_flow_div"].abs() >= float(cfg.min_divergence))
    return mask


def snap_filter_pass(row: pd.Series, cfg: FiltersConfig) -> bool:
    """
    Optional snap harvester gate: enforce counter-trend snaps when enabled.
    Returns True to allow the trade.
    """
    snap_cfg = (cfg.snap_filters or {}) if cfg else {}
    use_ctr = bool(snap_cfg.get("use_counter_trend", False))
    if not use_ctr:
        return True

    ta = row.get("trend_aligned")
    if ta is None or (isinstance(ta, float) and np.isnan(ta)):
        return False
    try:
        return int(ta) == 0
    except Exception:
        return False

def prepare_features_for_backtest(bars: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    loc_filter = cfg.shockflip.location_filter
    d_window = loc_filter['donchian_window'] if isinstance(loc_filter, dict) else loc_filter.donchian_window

    df = add_core_features(
        bars,
        z_window=cfg.shockflip.z_window,
        atr_window=cfg.risk.atr_window,
        donchian_window=d_window,
    )
    df = add_hypothesis_features(df, prior_flow_window=60, div_window=60, atr_pct_window=5000)
    df["rel_vol"] = _compute_relative_volume(df, window=60)
    df = _add_trend_context(df, fast_span=50, slow_span=200)
    atr = df.get("atr", np.nan)
    df["range_atr"] = (df["high"] - df["low"]) / (atr + 1e-9)
    df["shock_intensity_z"] = np.abs(df.get("imbalance_z", 0.0))
    df["shock_intensity_z"] = np.maximum(df["shock_intensity_z"], df["range_atr"])
    df["div_score"] = _compute_divergence_score(df, lookback=6)
    df["stall_flag"] = (
        (df["close"].pct_change(5).abs() < 0.0015)
        & (np.abs(df.get("imbalance_z", 0.0)) > 1.0)
    ).astype(int)
    df = add_time_features(df)
    df = detect_shockflip_signals(df, cfg.shockflip)
    return df

# -------------------------------------------------------------------------
# SIMULATION ENGINE
# -------------------------------------------------------------------------

def _simulate_trades(features: pd.DataFrame, cfg: BacktestConfig, progress: bool = True) -> pd.DataFrame:
    df = features.reset_index(drop=True).copy()
    filter_mask = apply_entry_filters(df, cfg.filters)
    
    trades: List[Dict] = []
    cooldown = 0
    
    sl_mult_long = cfg.risk.long.sl_mult
    sl_mult_short = cfg.risk.short.sl_mult
    tp_mult_long = cfg.risk.long.tp_mult
    tp_mult_short = cfg.risk.short.tp_mult
    cooldown_bars = cfg.risk.cooldown_bars
    meta_router = MetaRunnerRouter(cfg.meta_runner) if (cfg.meta_runner and cfg.meta_runner.enabled) else None
    
    # Management Settings
    be_threshold_r = cfg.mfe_breakeven_r
    time_stop_bars = cfg.time_stop_bars
    time_stop_r = cfg.time_stop_r if cfg.time_stop_r is not None else 0.5
    tol_div = 1e-6
    
    p = get_progress(progress, total=len(df), desc="Simulate trades")
    
    for i in range(len(df)):
        p.update(1)
        if cooldown > 0:
            cooldown -= 1
            continue

        row = df.iloc[i]
        raw_signal = int(row.get("shockflip_signal", 0))
        if raw_signal == 0: continue
        if not filter_mask.iloc[i]: continue
        if not snap_filter_pass(row, cfg.filters):
            continue

        # Optional microstructure gates for the harvester
        if cfg.filters.require_positive_divergence:
            if "price_flow_div" not in row or not np.isfinite(row["price_flow_div"]):
                continue
            if float(row["price_flow_div"]) <= tol_div:
                continue

        if cfg.filters.require_counter_trend:
            trend_dir = float(row.get("trend_dir", 0.0))
            if trend_dir != 0.0 and np.sign(raw_signal * trend_dir) >= 0:
                continue

        atr = float(row.get("atr", float("nan")))
        if not np.isfinite(atr) or atr <= 0: continue

        entry_ts = row["timestamp"]
        entry_price = float(row["close"])
        side = raw_signal
        
        base_sl_mult = sl_mult_long if side == 1 else sl_mult_short
        base_tp_mult = tp_mult_long if side == 1 else tp_mult_short
        eff_sl_mult = base_sl_mult
        eff_tp_mult = base_tp_mult
        p_runner = None
        runner_bucket = None

        if meta_router is not None:
            try:
                route = meta_router.route(row, side=side)
                p_runner = route.get("p_runner")
                runner_bucket = route.get("bucket")

                if meta_router.cfg.skip_low_bucket and runner_bucket == "low":
                    continue  # gate out low-confidence trades

                if meta_router.cfg.use_tp_sl_routing:
                    eff_tp_mult = float(route.get("tp_mult", eff_tp_mult))
                    eff_sl_mult = float(route.get("sl_mult", eff_sl_mult))
            except Exception as exc:  # pragma: no cover - defensive
                if cfg._debug:
                    print(f"[MetaRunner] routing failed at idx={i}: {exc}")

        risk_per_unit = atr * eff_sl_mult
        
        tp, sl, _, _ = build_barriers(
            side,
            entry_price,
            atr,
            cfg.risk,
            entry_price,
            entry_price,
            tp_mult_override=eff_tp_mult,
            sl_mult_override=eff_sl_mult,
        )
        if tp is None or sl is None: continue

        exit_idx, exit_ts, exit_price, result = None, None, None, None
        best_fav = 0.0
        worst_adv = 0.0
        be_active = False

        for j in range(i + 1, len(df)):
            bar = df.iloc[j]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            
            # 1. Update Path Stats
            fav = side * (high - entry_price)
            adv = side * (low - entry_price)
            if fav > best_fav: best_fav = fav
            if adv < worst_adv: worst_adv = adv
            
            # 2. Breakeven Lock (The Free Shot)
            if not be_active and be_threshold_r is not None:
                 if best_fav >= be_threshold_r * risk_per_unit:
                     be_active = True
            
            # 3. Zombie Kill (The Time Stop)
            bars_held = j - i
            if time_stop_bars is not None and bars_held >= time_stop_bars:
                # If we haven't hit the survival threshold (0.5R) by bar 10...
                if best_fav < time_stop_r * risk_per_unit:
                    # KILL IT
                    exit_price = close
                    exit_ts = bar["timestamp"]
                    exit_idx = j
                    result = "ZOMBIE"
                    break

            # 4. Check Exit Barriers
            sl_eff = sl
            if be_active:
                sl_eff = max(sl_eff, entry_price) if side == 1 else min(sl_eff, entry_price)
            
            hit_sl_eff = (low <= sl_eff) if side == 1 else (high >= sl_eff)
            hit_tp = (high >= tp) if side == 1 else (low <= tp)
            
            if hit_sl_eff:
                exit_price = sl_eff
                exit_ts = bar["timestamp"]
                exit_idx = j
                # Label BE if stopped at entry
                is_be = np.isclose(sl_eff, entry_price, atol=1e-8)
                result = "BE" if is_be else "SL"
                break
                
            if hit_tp:
                result = "TP"
                exit_price = tp
                exit_ts = bar["timestamp"]
                exit_idx = j
                break
        
        # End of Data Close
        if exit_idx is None:
            bar = df.iloc[-1]
            exit_idx = len(df) - 1
            exit_ts = bar["timestamp"]
            exit_price = float(bar["close"])
            result = "TP" if side * (exit_price - entry_price) > 0 else "SL"

        pnl = compute_trade_pnl(side, entry_price, exit_price, cfg.fees, cfg.slippage_bp)
        
        trades.append(dict(
            symbol=cfg.symbol,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            result=result,
            pnl=pnl,
            mfe_r=best_fav / risk_per_unit if risk_per_unit > 0 else 0,
            mae_r=abs(worst_adv / risk_per_unit) if risk_per_unit > 0 else 0,
            holding_period=exit_idx - i,
            p_runner=p_runner,
            runner_bucket=runner_bucket,
            tp_mult=eff_tp_mult,
            sl_mult=eff_sl_mult,
        ))
        cooldown = cooldown_bars

    p.close()
    return pd.DataFrame(trades)

def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty: return {"n": 0, "win_rate": 0.0, "pf": 0.0, "total_pnl": 0.0}
    pnl = trades["pnl"]
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    pf = float(pos / -neg) if neg < 0 else (100.0 if pos > 0 else 0.0)
    return {"n": len(trades), "win_rate": float((pnl > 0).mean()), "pf": pf, "total_pnl": float(pnl.sum())}

def run_backtest_from_bars(bars, cfg):
    feats = prepare_features_for_backtest(bars, cfg)
    trades = _simulate_trades(feats, cfg, progress=cfg._progress)
    stats = summarize_trades(trades)
    return trades, stats
