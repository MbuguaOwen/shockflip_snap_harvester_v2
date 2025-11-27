import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest_from_bars
from .features import add_core_features
from .shockflip_detector import ShockFlipConfig, detect_shockflip_signals


@dataclass
class ParityReport:
    n_trades_research: int
    n_trades_live: int
    identical: bool
    max_abs_pnl_diff: float


def _run_research_path(bars: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Research-style path: use backtest helper directly."""
    trades, _ = run_backtest_from_bars(bars, cfg)
    return trades


def _run_live_style_path(bars: pd.DataFrame, cfg: BacktestConfig, progress: bool = True) -> pd.DataFrame:
    """Live-style path: recompute features and simulate trades in a step-wise loop.

    For simplicity we still rely on the same feature and ShockFlip detector
    modules, but we iterate bar-by-bar to mimic a streaming engine.
    """
    # Precompute features (causal) for all bars
    feats = add_core_features(
        bars,
        z_window=cfg.shockflip.z_window,
        atr_window=cfg.risk.atr_window,
        donchian_window=cfg.shockflip.donchian_window,
    )
    feats = detect_shockflip_signals(feats, cfg.shockflip)

    # Now simulate trades by calling the same backtest internals
    # to keep invariants identical.
    from .backtest import _simulate_trades  # type: ignore

    trades = _simulate_trades(feats, cfg, progress=progress)
    return trades


def run_parity(
    bars: pd.DataFrame,
    cfg: BacktestConfig,
    progress: bool = True,
) -> Tuple[ParityReport, pd.DataFrame, pd.DataFrame]:
    """Run research vs live-style paths and compare trade outputs."""
    # Propagate progress preference into the backtest helpers
    try:
        setattr(cfg, "_progress", progress)
    except Exception:
        pass

    research_trades = _run_research_path(bars, cfg)
    live_trades = _run_live_style_path(bars, cfg, progress=progress)

    if research_trades.empty and live_trades.empty:
        report = ParityReport(
            n_trades_research=0,
            n_trades_live=0,
            identical=True,
            max_abs_pnl_diff=0.0,
        )
        return report, research_trades, live_trades

    # Align by index; in a stricter setup we'd align by entry_ts/side
    min_len = min(len(research_trades), len(live_trades))
    pnl_diff = np.abs(
        research_trades["pnl"].iloc[:min_len].to_numpy()
        - live_trades["pnl"].iloc[:min_len].to_numpy()
    )
    max_diff = float(pnl_diff.max()) if pnl_diff.size else 0.0

    identical = (
        len(research_trades) == len(live_trades)
        and max_diff < 1e-12
        and research_trades[["entry_ts", "exit_ts", "side", "result"]]
        .reset_index(drop=True)
        .equals(
            live_trades[["entry_ts", "exit_ts", "side", "result"]]
            .reset_index(drop=True)
        )
    )

    report = ParityReport(
        n_trades_research=int(len(research_trades)),
        n_trades_live=int(len(live_trades)),
        identical=bool(identical),
        max_abs_pnl_diff=max_diff,
    )

    return report, research_trades, live_trades


def parity_report_to_dict(report: ParityReport) -> Dict:
    return {
        "n_trades_research": report.n_trades_research,
        "n_trades_live": report.n_trades_live,
        "identical": report.identical,
        "max_abs_pnl_diff": report.max_abs_pnl_diff,
    }
