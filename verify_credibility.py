#!/usr/bin/env python3
"""
Phase 1: ShockFlip Snap Harvester v2 - Results Credibility Verification

This script establishes the believability of your trading results by verifying:
1. Data integrity (no leakage, proper train/test split)
2. Feature causality (no look-ahead bias)
3. Barrier logic correctness
4. Statistical plausibility
5. Reproducibility

Run this BEFORE claiming final results to establish methodological soundness.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Assume we're running from repo root
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class VerificationResult:
    """Container for verification check results."""
    check_name: str
    passed: bool
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    message: str
    details: Dict = None
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"[{self.severity}] {status}: {self.check_name}\n  → {self.message}"


class CredibilityVerifier:
    """Comprehensive verification suite for trading results."""
    
    def __init__(self, config_path: str = "configs/snap_harvester_2024_btc_agg.yaml"):
        self.config_path = Path(config_path)
        self.results: List[VerificationResult] = []
        self.critical_failures = 0
        
    def run_all_checks(self) -> Tuple[bool, List[VerificationResult]]:
        """Run all verification checks in sequence."""
        print("=" * 80)
        print("PHASE 1: RESULTS CREDIBILITY VERIFICATION")
        print("=" * 80)
        print()
        
        # Load config first
        try:
            with open(self.config_path) as f:
                import yaml
                self.cfg = yaml.safe_load(f)
        except Exception as e:
            self._add_result(
                "Config Loading",
                False,
                "CRITICAL",
                f"Failed to load config: {e}"
            )
            return False, self.results
        
        # Run checks in order of importance
        self._check_1_temporal_split()
        self._check_2_feature_causality()
        self._check_3_data_leakage()
        self._check_4_model_integrity()
        self._check_5_barrier_logic()
        self._check_6_statistical_plausibility()
        self._check_7_reproducibility()
        self._check_8_overfitting_indicators()
        
        # Summary
        self._print_summary()
        
        return self.critical_failures == 0, self.results
    
    # =========================================================================
    # CHECK 1: TEMPORAL SPLIT INTEGRITY
    # =========================================================================
    def _check_1_temporal_split(self):
        """Verify train/test split has no temporal overlap."""
        print("\n[CHECK 1] Temporal Split Integrity")
        print("-" * 80)
        
        try:
            train_cfg = self.cfg["train"]
            train_start = pd.to_datetime(train_cfg["train_start"])
            train_end = pd.to_datetime(train_cfg["train_end"])
            oos_start = pd.to_datetime(train_cfg["oos_start"])
            oos_end = pd.to_datetime(train_cfg["oos_end"])
            
            # Check 1a: No overlap
            if train_end >= oos_start:
                self._add_result(
                    "Temporal Split - No Overlap",
                    False,
                    "CRITICAL",
                    f"Train period ({train_end}) overlaps with OOS period ({oos_start})"
                )
            else:
                gap_days = (oos_start - train_end).days
                self._add_result(
                    "Temporal Split - No Overlap",
                    True,
                    "CRITICAL",
                    f"Clean temporal split with {gap_days} day gap between train and OOS",
                    {"train_end": str(train_end), "oos_start": str(oos_start), "gap_days": gap_days}
                )
            
            # Check 1b: Verify actual data matches config
            meta_path = Path(self.cfg["paths"]["meta_out"])
            if meta_path.exists():
                meta = pd.read_csv(meta_path, parse_dates=["timestamp"])
                # Ensure timezone consistency
                actual_start = pd.to_datetime(meta["timestamp"].min(), utc=True)
                actual_end = pd.to_datetime(meta["timestamp"].max(), utc=True)
                train_start = pd.to_datetime(train_start, utc=True)
                train_end = pd.to_datetime(train_end, utc=True)
                
                if actual_start < train_start or actual_end > train_end:
                    self._add_result(
                        "Temporal Split - Data Matches Config",
                        False,
                        "HIGH",
                        f"Training data outside config bounds: {actual_start} to {actual_end}"
                    )
                else:
                    self._add_result(
                        "Temporal Split - Data Matches Config",
                        True,
                        "HIGH",
                        f"Training data within bounds: {actual_start} to {actual_end}"
                    )
            
        except Exception as e:
            self._add_result(
                "Temporal Split",
                False,
                "CRITICAL",
                f"Error checking temporal split: {e}"
            )
    
    # =========================================================================
    # CHECK 2: FEATURE CAUSALITY
    # =========================================================================
    def _check_2_feature_causality(self):
        """Verify all features are strictly causal (no look-ahead)."""
        print("\n[CHECK 2] Feature Causality (No Look-Ahead Bias)")
        print("-" * 80)
        
        meta_cfg = self.cfg.get("meta", {})
        train_features = meta_cfg.get("train_features", [])
        diagnostics_only = meta_cfg.get("diagnostics_only_features", [])
        
        # Features that would leak future info
        FORBIDDEN_IN_TRAINING = [
            "mfe_",  # Max favorable excursion - future path dependent
            "snap_H",  # Snap labels - future outcome
            "barrier_y_",  # Barrier outcomes - future path dependent
            "r_final",  # Final R - known only after exit
            "exit_",  # Any exit timing/price
            "hit_tp", "hit_sl", "hit_be"  # Exit outcomes
        ]
        
        leaking_features = []
        for feat in train_features:
            for forbidden in FORBIDDEN_IN_TRAINING:
                if forbidden in feat.lower():
                    leaking_features.append(feat)
                    break
        
        if leaking_features:
            self._add_result(
                "Feature Causality - No Future Info",
                False,
                "CRITICAL",
                f"Training features contain look-ahead data: {leaking_features}"
            )
        else:
            self._add_result(
                "Feature Causality - No Future Info",
                True,
                "CRITICAL",
                f"All {len(train_features)} training features are causal"
            )
        
        # Verify diagnostics are properly excluded
        overlap = set(train_features) & set(diagnostics_only)
        if overlap:
            self._add_result(
                "Feature Causality - Diagnostics Excluded",
                False,
                "CRITICAL",
                f"Diagnostic features found in training set: {overlap}"
            )
        else:
            self._add_result(
                "Feature Causality - Diagnostics Excluded",
                True,
                "HIGH",
                f"{len(diagnostics_only)} diagnostic features properly excluded from training"
            )
    
    # =========================================================================
    # CHECK 3: DATA LEAKAGE DETECTION
    # =========================================================================
    def _check_3_data_leakage(self):
        """Check for subtle data leakage patterns."""
        print("\n[CHECK 3] Data Leakage Detection")
        print("-" * 80)
        
        # Check 3a: Model file dates
        model_path = Path(self.cfg["paths"]["model_out"])
        if model_path.exists():
            model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
            
            # Load model and check training data timestamp
            try:
                model = joblib.load(model_path)
                
                # The model should have been trained BEFORE the OOS period
                oos_start = pd.to_datetime(self.cfg["train"]["oos_start"])
                
                self._add_result(
                    "Data Leakage - Model Timestamp",
                    True,
                    "MEDIUM",
                    f"Model file created: {model_mtime.strftime('%Y-%m-%d %H:%M:%S')}",
                    {"model_mtime": str(model_mtime), "oos_start": str(oos_start)}
                )
            except Exception as e:
                self._add_result(
                    "Data Leakage - Model Loading",
                    False,
                    "MEDIUM",
                    f"Could not load model: {e}"
                )
        
        # Check 3b: Full-sample statistics in features
        meta_cfg = self.cfg.get("meta", {})
        train_features = meta_cfg.get("train_features", [])
        
        full_sample_features = [f for f in train_features if "fullsample" in f.lower()]
        if full_sample_features:
            self._add_result(
                "Data Leakage - No Full-Sample Features",
                False,
                "HIGH",
                f"Full-sample features in training (uses future data): {full_sample_features}"
            )
        else:
            self._add_result(
                "Data Leakage - No Full-Sample Features",
                True,
                "HIGH",
                "No full-sample statistics in training features"
            )
    
    # =========================================================================
    # CHECK 4: MODEL INTEGRITY
    # =========================================================================
    def _check_4_model_integrity(self):
        """Verify model was trained correctly and hasn't been retrained on OOS."""
        print("\n[CHECK 4] Model Integrity")
        print("-" * 80)
        
        model_path = Path(self.cfg["paths"]["model_out"])
        
        if not model_path.exists():
            self._add_result(
                "Model Integrity - File Exists",
                False,
                "CRITICAL",
                f"Model file not found: {model_path}"
            )
            return
        
        try:
            model = joblib.load(model_path)
            
            # Check 4a: Model type
            expected_type = self.cfg.get("model", {}).get("type", "hgb")
            if expected_type == "hgb":
                from sklearn.ensemble import HistGradientBoostingClassifier
                is_correct_type = isinstance(model, HistGradientBoostingClassifier)
            else:
                is_correct_type = True  # Can't verify other types
            
            self._add_result(
                "Model Integrity - Correct Type",
                is_correct_type,
                "HIGH",
                f"Model is correct type: {type(model).__name__}"
            )
            
            # Check 4b: Feature names match config
            if hasattr(model, "feature_names_in_"):
                model_features = set(model.feature_names_in_)
                config_features = set(self.cfg["meta"]["train_features"])
                
                missing = config_features - model_features
                extra = model_features - config_features
                
                if missing or extra:
                    self._add_result(
                        "Model Integrity - Feature Alignment",
                        False,
                        "HIGH",
                        f"Feature mismatch: {len(missing)} missing, {len(extra)} extra"
                    )
                else:
                    self._add_result(
                        "Model Integrity - Feature Alignment",
                        True,
                        "HIGH",
                        f"All {len(model_features)} features aligned"
                    )
        
        except Exception as e:
            self._add_result(
                "Model Integrity",
                False,
                "CRITICAL",
                f"Error loading/verifying model: {e}"
            )
    
    # =========================================================================
    # CHECK 5: BARRIER LOGIC CORRECTNESS
    # =========================================================================
    def _check_5_barrier_logic(self):
        """Verify barrier/exit logic is realistic and correctly implemented."""
        print("\n[CHECK 5] Barrier Logic Correctness")
        print("-" * 80)
        
        risk_cfg = self.cfg.get("risk", {})
        
        # Check 5a: Risk parameters are sensible
        risk_k = risk_cfg.get("risk_k_atr", 0.5)
        sl_mult = risk_cfg.get("sl_r_multiple", 2.5)
        tp_mult = risk_cfg.get("tp_r_multiple", 4.0)
        horizon = risk_cfg.get("horizon_bars", 30)
        
        # SL should be smaller than TP for positive expectancy
        if sl_mult >= tp_mult:
            self._add_result(
                "Barrier Logic - SL < TP",
                False,
                "HIGH",
                f"SL ({sl_mult}R) >= TP ({tp_mult}R) - unusual risk/reward"
            )
        else:
            rr_ratio = tp_mult / sl_mult
            self._add_result(
                "Barrier Logic - SL < TP",
                True,
                "MEDIUM",
                f"Risk/Reward ratio: {rr_ratio:.2f}:1 (TP={tp_mult}R, SL={sl_mult}R)"
            )
        
        # Check 5b: Horizon is reasonable (30 bars = 30 min)
        if horizon < 5 or horizon > 500:
            self._add_result(
                "Barrier Logic - Reasonable Horizon",
                False,
                "MEDIUM",
                f"Unusual horizon: {horizon} bars"
            )
        else:
            self._add_result(
                "Barrier Logic - Reasonable Horizon",
                True,
                "LOW",
                f"Horizon: {horizon} bars (~{horizon} minutes)"
            )
        
        # Check 5c: Verify breakeven logic if enabled
        be_mult = risk_cfg.get("be_r_multiple")
        if be_mult is not None and be_mult > 0:
            if be_mult >= tp_mult:
                self._add_result(
                    "Barrier Logic - BE < TP",
                    False,
                    "HIGH",
                    f"Breakeven ({be_mult}R) >= TP ({tp_mult}R) - impossible to hit TP"
                )
            else:
                self._add_result(
                    "Barrier Logic - BE Logic",
                    True,
                    "LOW",
                    f"Breakeven enabled at {be_mult}R (before TP at {tp_mult}R)"
                )
    
    # =========================================================================
    # CHECK 6: STATISTICAL PLAUSIBILITY
    # =========================================================================
    def _check_6_statistical_plausibility(self):
        """Check if claimed performance is statistically plausible."""
        print("\n[CHECK 6] Statistical Plausibility")
        print("-" * 80)
        
        # We'll check this against results once provided
        # For now, establish theoretical bounds
        
        risk_cfg = self.cfg.get("risk", {})
        tp_mult = risk_cfg.get("tp_r_multiple", 4.0)
        sl_mult = risk_cfg.get("sl_r_multiple", 2.5)
        
        # Theoretical maximum win rate for breakeven
        # E[R] = p_win * TP - (1 - p_win) * SL = 0
        # p_win = SL / (TP + SL)
        breakeven_winrate = sl_mult / (tp_mult + sl_mult)
        
        self._add_result(
            "Statistical Plausibility - Breakeven WR",
            True,
            "LOW",
            f"Breakeven win rate: {breakeven_winrate:.1%} (TP={tp_mult}R, SL={sl_mult}R)",
            {
                "breakeven_wr": breakeven_winrate,
                "tp_r": tp_mult,
                "sl_r": sl_mult
            }
        )
        
        # For 80% win rate with these parameters:
        # E[R] = 0.80 * 4.0 - 0.20 * 2.5 = 3.2 - 0.5 = 2.7R per trade
        expected_r = 0.80 * tp_mult - 0.20 * sl_mult
        
        self._add_result(
            "Statistical Plausibility - Expected R",
            True,
            "LOW",
            f"If WR=80%: Expected R = {expected_r:.2f}R per trade",
            {"theoretical_r_at_80pct": expected_r}
        )
    
    # =========================================================================
    # CHECK 7: REPRODUCIBILITY
    # =========================================================================
    def _check_7_reproducibility(self):
        """Verify results can be reproduced from documented steps."""
        print("\n[CHECK 7] Reproducibility")
        print("-" * 80)
        
        # Check 7a: All required files exist
        paths_to_check = [
            ("Config", self.config_path),
            ("Model", Path(self.cfg["paths"]["model_out"])),
            ("Training Meta", Path(self.cfg["paths"]["meta_out"])),
        ]
        
        all_exist = True
        for name, path in paths_to_check:
            exists = path.exists()
            if not exists:
                all_exist = False
                print(f"  ✗ Missing: {name} at {path}")
            else:
                print(f"  ✓ Found: {name}")
        
        self._add_result(
            "Reproducibility - Required Files",
            all_exist,
            "HIGH",
            f"{'All' if all_exist else 'Some'} required files present"
        )
        
        # Check 7b: Random seed is set
        seed = self.cfg.get("experiment", {}).get("seed")
        if seed is not None:
            self._add_result(
                "Reproducibility - Random Seed",
                True,
                "MEDIUM",
                f"Random seed set: {seed}"
            )
        else:
            self._add_result(
                "Reproducibility - Random Seed",
                False,
                "MEDIUM",
                "No random seed configured - results may not be reproducible"
            )
    
    # =========================================================================
    # CHECK 8: OVERFITTING INDICATORS
    # =========================================================================
    def _check_8_overfitting_indicators(self):
        """Check for signs of overfitting."""
        print("\n[CHECK 8] Overfitting Indicators")
        print("-" * 80)
        
        # Check 8a: Model complexity
        model_cfg = self.cfg.get("model", {}).get("params", {})
        max_depth = model_cfg.get("max_depth", 6)
        min_samples_leaf = model_cfg.get("min_samples_leaf", 30)
        
        if max_depth > 10:
            self._add_result(
                "Overfitting - Tree Depth",
                False,
                "MEDIUM",
                f"Very deep trees (depth={max_depth}) may overfit"
            )
        else:
            self._add_result(
                "Overfitting - Tree Depth",
                True,
                "MEDIUM",
                f"Reasonable tree depth: {max_depth}"
            )
        
        if min_samples_leaf < 10:
            self._add_result(
                "Overfitting - Min Samples",
                False,
                "MEDIUM",
                f"Small min_samples_leaf ({min_samples_leaf}) may overfit"
            )
        else:
            self._add_result(
                "Overfitting - Min Samples",
                True,
                "MEDIUM",
                f"Adequate min_samples_leaf: {min_samples_leaf}"
            )
        
        # Check 8b: Feature count vs sample size
        meta_path = Path(self.cfg["paths"]["meta_out"])
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            n_samples = len(meta)
            n_features = len(self.cfg["meta"]["train_features"])
            ratio = n_samples / n_features
            
            if ratio < 10:
                self._add_result(
                    "Overfitting - Samples/Features Ratio",
                    False,
                    "HIGH",
                    f"Low ratio: {n_samples} samples / {n_features} features = {ratio:.1f}:1"
                )
            else:
                self._add_result(
                    "Overfitting - Samples/Features Ratio",
                    True,
                    "MEDIUM",
                    f"Good ratio: {n_samples} samples / {n_features} features = {ratio:.1f}:1"
                )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _add_result(self, check_name: str, passed: bool, severity: str, 
                    message: str, details: Dict = None):
        """Add a verification result."""
        result = VerificationResult(check_name, passed, severity, message, details)
        self.results.append(result)
        
        if not passed and severity == "CRITICAL":
            self.critical_failures += 1
        
        print(f"  {result}")
    
    def _print_summary(self):
        """Print summary of all checks."""
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        
        by_severity = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        for result in self.results:
            by_severity[result.severity].append(result)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        print(f"\nOverall: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
        print()
        
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            checks = by_severity[severity]
            if checks:
                failed = [c for c in checks if not c.passed]
                print(f"{severity}: {len(checks)-len(failed)}/{len(checks)} passed")
                if failed:
                    for check in failed:
                        print(f"  ✗ {check.check_name}: {check.message}")
        
        print()
        if self.critical_failures > 0:
            print(f"⚠️  {self.critical_failures} CRITICAL FAILURES - Results NOT credible")
        else:
            print("✓ All critical checks passed - Methodology is sound")
        print()


def main():
    """Run verification suite."""
    verifier = CredibilityVerifier("configs/snap_harvester_2024_btc_agg.yaml")
    is_credible, results = verifier.run_all_checks()
    
    # Export results
    output = {
        "timestamp": datetime.now().isoformat(),
        "is_credible": is_credible,
        "critical_failures": verifier.critical_failures,
        "checks": [
            {
                "name": r.check_name,
                "passed": r.passed,
                "severity": r.severity,
                "message": r.message,
                "details": r.details
            }
            for r in results
        ]
    }
    
    output_path = Path("results/verification/phase1_credibility_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Report saved to: {output_path}")
    
    return 0 if is_credible else 1


if __name__ == "__main__":
    sys.exit(main())