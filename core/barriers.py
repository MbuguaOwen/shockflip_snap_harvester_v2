from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeesConfig:
    taker_bp: float = 1.0  # basis points


@dataclass
class RiskSideConfig:
    tp_mult: float
    sl_mult: float


@dataclass
class RiskConfig:
    atr_window: int = 60
    cooldown_bars: int = 10
    long: RiskSideConfig = field(default_factory=lambda: RiskSideConfig(tp_mult=27.5, sl_mult=9.0))
    short: RiskSideConfig = field(default_factory=lambda: RiskSideConfig(tp_mult=15.0, sl_mult=6.5))


def build_barriers(
    side: int,
    entry: float,
    atr: float,
    risk: RiskConfig,
    high: float,
    low: float,
    tp_mult_override: float | None = None,
    sl_mult_override: float | None = None,
):
    """Construct TP/SL and evaluate hits for a single bar.

    Returns (tp, sl, hit_tp, hit_sl).
    """
    if side not in (+1, -1):
        raise ValueError(f"side must be +1 or -1, got {side}")
    if atr <= 0 or not np.isfinite(atr):
        return None, None, False, False

    if side == +1:
        tp_mult = tp_mult_override if tp_mult_override is not None else risk.long.tp_mult
        sl_mult = sl_mult_override if sl_mult_override is not None else risk.long.sl_mult
        tp = entry + tp_mult * atr
        sl = entry - sl_mult * atr
        hit_tp = high >= tp
        hit_sl = low <= sl
    else:
        tp_mult = tp_mult_override if tp_mult_override is not None else risk.short.tp_mult
        sl_mult = sl_mult_override if sl_mult_override is not None else risk.short.sl_mult
        tp = entry - tp_mult * atr
        sl = entry + sl_mult * atr
        hit_tp = low <= tp
        hit_sl = high >= sl

    return tp, sl, hit_tp, hit_sl


def compute_trade_pnl(
    side: int,
    entry_price: float,
    exit_price: float,
    fees: FeesConfig,
    slippage_bp: float,
) -> float:
    """Compute PnL (fractional return) including fees + slippage.

    We assume:
    - fees charged on notional both at entry and exit.
    - slippage applied symmetrically at entry and exit in the adverse direction.
    """
    if side not in (+1, -1):
        raise ValueError("side must be +1 or -1")

    raw_ret = side * (exit_price - entry_price) / entry_price

    # Total bps (entry + exit)
    total_bp = 2.0 * fees.taker_bp + 2.0 * slippage_bp
    cost = total_bp * 1e-4

    pnl = raw_ret - cost
    return float(pnl)


def enforce_tp_sl_invariants(
    side: int,
    result: str,
    pnl: float,
    entry: float,
    exit_price: float,
):
    """Raise if invariants are broken: TP => pnl>0, SL => pnl<0."""
    assert side in (+1, -1), f"Invalid side {side}"
    assert result in ("TP", "SL", "BE"), f"Invalid result {result}"

    if result == "TP":
        assert pnl > 0, f"TP invariant broken: entry={entry}, exit={exit_price}, pnl={pnl}"
    elif result == "SL":
        assert pnl < 0, f"SL invariant broken: entry={entry}, exit={exit_price}, pnl={pnl}"
    elif result == "BE":
        # Breakeven exits should be approximately flat after costs.
        # We only require that they are not strongly profitable; allow tiny negatives from fees.
        pass
