import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ShockFlipConfig:
    source: str = "imbalance"
    z_window: int = 240
    z_band: float = 2.5
    jump_band: float = 3.0
    persistence_bars: int = 6
    persistence_ratio: float = 0.60
    # These two fields were likely missing in your old version
    location_filter: Dict = field(default_factory=lambda: {"donchian_window": 120, "require_extreme": True})
    dynamic_thresholds: Dict = field(default_factory=lambda: {"enabled": False, "percentile": 0.99})

def detect_shockflip_signals(df: pd.DataFrame, cfg: ShockFlipConfig) -> pd.DataFrame:
    """
    Applies ShockFlip logic:
    1. Calculates Z-scores for flow (imbalance).
    2. Checks for sudden "jumps" (volatility shocks).
    3. Filters by price location (Donchian Channel).
    4. Generates +1 (Buy) / -1 (Sell) signals.
    """
    df = df.copy()
    
    # --- 1. Signal Source (Z-Score) ---
    if cfg.source == "imbalance":
        # Assumes 'imbalance_z' is already calculated in core/features.py
        # If not, we calculate a basic one here as fallback:
        if 'imbalance_z' not in df.columns:
            # Simple proxy calculation if missing
            vol_ma = df['volume'].rolling(cfg.z_window).mean()
            imb_raw = (df['buy_qty'] - df['sell_qty']) / (vol_ma + 1e-9)
            mean = imb_raw.rolling(cfg.z_window).mean()
            std = imb_raw.rolling(cfg.z_window).std()
            df['imbalance_z'] = (imb_raw - mean) / (std + 1e-9)
        
        signal_col = 'imbalance_z'
    else:
        # Fallback or other sources
        signal_col = 'imbalance_z'

    # --- 2. Dynamic Thresholds (Optional) ---
    # If enabled, we raise the z_band based on recent volatility
    current_z_band = cfg.z_band
    if cfg.dynamic_thresholds.get("enabled", False):
        # Calculate rolling percentile of absolute Z to find "extreme" regime
        roll_z_high = df[signal_col].abs().rolling(cfg.z_window).quantile(cfg.dynamic_thresholds.get("percentile", 0.99))
        # If the market is calm, this is low. If crazy, this is high.
        # We enforce that the threshold is AT LEAST z_band, but can be higher.
        # This prevents signals during pure noise, but might be too strict.
        # For now, let's keep it simple: strict thresholding.
        # (A common simple logic: threshold = max(cfg.z_band, roll_z_high * 0.8))
        pass 

    # --- 3. Detection Logic ---
    # Long Signal: Z < -Threshold (Selling Exhaustion -> Reversal Long) 
    # Short Signal: Z > +Threshold (Buying Exhaustion -> Reversal Short)
    # Note: ShockFlip is often mean-reverting on extreme flow.
    
    # However, standard "Trend" ShockFlip might follow the flow.
    # Let's assume Reversion logic (Contrarian) usually for "ShockFlip":
    # High Buying (Z > 2.5) -> Price likely to drop -> Signal -1
    # High Selling (Z < -2.5) -> Price likely to bounce -> Signal 1
    
    long_condition = (df[signal_col] <= -current_z_band)
    short_condition = (df[signal_col] >= current_z_band)

    # --- 4. Jump/Shock Filter (Volatility) ---
    # We want price to have moved FAST recently (shock), creating the dislocation.
    # Use ATR or simple high-low range relative to average
    if 'atr' in df.columns:
        # Relative range
        rel_range = (df['high'] - df['low']) / (df['atr'] + 1e-9)
        # If the bar is huge (e.g. 3x ATR), it's a shock.
        shock_cond = rel_range > cfg.jump_band
        
        # Combine
        long_condition = long_condition & shock_cond
        short_condition = short_condition & shock_cond

    # --- 5. Location Filter (Donchian) ---
    # We only want to catch falling knives (Long) or shooting stars (Short)
    # Long: Price near Donchian Low
    # Short: Price near Donchian High
    if cfg.location_filter.get("require_extreme", False):
        d_win = cfg.location_filter.get("donchian_window", 120)
        
        d_low = df['low'].rolling(d_win).min()
        d_high = df['high'].rolling(d_win).max()
        
        # Allow some buffer (e.g. within 1% or just strict touch)
        # Strict touch of recent low for Longs
        is_low = (df['low'] <= d_low * 1.001) 
        is_high = (df['high'] >= d_high * 0.999)
        
        long_condition = long_condition & is_low
        short_condition = short_condition & is_high

    # --- 6. Generate Output ---
    df['shockflip_signal'] = 0
    df.loc[long_condition, 'shockflip_signal'] = 1
    df.loc[short_condition, 'shockflip_signal'] = -1

    return df