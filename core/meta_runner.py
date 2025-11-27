import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd


@dataclass
class MetaRunnerConfig:
    enabled: bool = False
    model_path: str = ""
    p_low: float = 0.35
    p_high: float = 0.60
    routing: Dict[str, Dict[str, float]] = None  # type: ignore[assignment]
    skip_low_bucket: bool = False          # If true, drop trades in "low" bucket
    use_tp_sl_routing: bool = True         # If false, only gate; do not override TP/SL


def load_meta_runner_cfg(raw: dict) -> MetaRunnerConfig:
    routing_defaults = {
        "low": {"tp_mult": 1.0, "sl_mult": 0.5},
        "mid": {"tp_mult": 2.0, "sl_mult": 0.5},
        "high": {"tp_mult": 3.0, "sl_mult": 0.5},
    }
    routing_cfg = raw.get("routing") or routing_defaults
    return MetaRunnerConfig(
        enabled=bool(raw.get("enabled", False)),
        model_path=str(raw.get("model_path", "results/meta_runner_barrier_30_3R/runner_meta_model_hgb.pkl")),
        p_low=float(raw.get("p_low", 0.35)),
        p_high=float(raw.get("p_high", 0.60)),
        routing=routing_cfg,
        skip_low_bucket=bool(raw.get("skip_low_bucket", False)),
        use_tp_sl_routing=bool(raw.get("use_tp_sl_routing", True)),
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day/session features to a DataFrame with a timestamp column."""
    out = df.copy()
    if "timestamp" not in out.columns:
        return out

    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)

    dow = ts.dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    minute = ts.dt.minute
    out["is_first5"] = (minute < 5).astype(int)
    out["is_last5"] = (minute >= 55).astype(int)

    h = ts.dt.hour
    out["sess_asia"] = ((h >= 0) & (h < 8)).astype(int)
    out["sess_london"] = ((h >= 8) & (h < 16)).astype(int)
    out["sess_ny"] = ((h >= 13) & (h < 21)).astype(int)
    out["sess_ln_ny_overlap"] = ((h >= 13) & (h < 16)).astype(int)

    return out


class MetaRunnerRouter:
    """
    Wraps a trained sklearn Pipeline (preprocessor + classifier) to score ShockFlip events.
    Converts t0 features into p_runner and TP/SL R-multipliers based on config routing.
    """

    def __init__(self, cfg: Optional[MetaRunnerConfig]):
        self.cfg = cfg or MetaRunnerConfig(enabled=False, routing={})
        self.model = None
        self.required_columns: Sequence[str] = []

        if not self.cfg.enabled:
            return

        try:
            self.model = joblib.load(self.cfg.model_path)
            pre = None
            if hasattr(self.model, "named_steps"):
                pre = self.model.named_steps.get("pre")
            if pre is not None and hasattr(pre, "transformers_"):
                cols: list[str] = []
                for name, trans, c in pre.transformers_:
                    if name == "remainder":
                        continue
                    if isinstance(c, list):
                        cols.extend(c)
                    elif isinstance(c, str):
                        cols.append(c)
                self.required_columns = cols
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[MetaRunner] Warning: failed to load model at {self.cfg.model_path}: {exc}")
            self.cfg.enabled = False
            self.model = None

    def _prepare_row(self, feat_row: pd.Series, side: Optional[int]) -> pd.DataFrame:
        """Prepare a single-row DataFrame with required columns for the model."""
        df = pd.DataFrame([feat_row.to_dict()])

        # Derived categorical alignment feature
        if "trend_aligned" not in df.columns and side is not None:
            val = df.get("trend_dir", 0.0)
            if hasattr(val, "iloc"):
                val = val.iloc[0]
            trend_dir = float(val)
            df["trend_aligned"] = int(trend_dir != 0.0 and np.sign(side * trend_dir) > 0)

        # Time features
        df = add_time_features(df)

        # Ensure stall_flag is int
        if "stall_flag" in df.columns:
            df["stall_flag"] = df["stall_flag"].fillna(0).astype(int)

        # Add any missing required columns expected by the preprocessor
        for col in self.required_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def score_event(self, feat_row: pd.Series, side: Optional[int] = None) -> float:
        if (self.model is None) or (not self.cfg.enabled):
            return 0.5

        X = self._prepare_row(feat_row, side=side)
        try:
            proba = self.model.predict_proba(X)[:, 1]
            return float(proba[0])
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[MetaRunner] Warning: predict_proba failed, defaulting to 0.5 ({exc})")
            return 0.5

    def route(self, feat_row: pd.Series, side: Optional[int] = None) -> Dict[str, Any]:
        """Return tp_mult/sl_mult along with p_runner and bucket."""
        p = self.score_event(feat_row, side=side)
        if p < self.cfg.p_low:
            bucket = "low"
        elif p < self.cfg.p_high:
            bucket = "mid"
        else:
            bucket = "high"

        routing_cfg = (self.cfg.routing or {}).get(bucket, {})
        tp_mult = float(routing_cfg.get("tp_mult", 2.0))
        sl_mult = float(routing_cfg.get("sl_mult", 0.5))

        return {
            "p_runner": p,
            "bucket": bucket,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
        }
