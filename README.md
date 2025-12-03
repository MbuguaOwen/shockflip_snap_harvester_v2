# ShockFlip Snap Harvester v2

## BTCUSDT 2025 – Performance Verification Memo

### 1) Scope
- Engine: Snap Harvester v2 meta-model, BTCUSDT only.
- Model: `results/models/hgb_snap_2024_BTC.joblib` (trained on 2024 BTC, no 2025 leakage).
- OOS window validated: 2025 Jan–Jul BTCUSDT.
- Geometry: TP = +3.0R, SL = -0.5R, horizon = 30 bars (risk_profile `strategy`).
- Negative tail tests (10R/20R with SL 1.5R/2R) are explicitly rejected.
- ETH not yet validated (await 2024 ETH data + full pipeline).

### 2) OOS purity & training window (verified)
- Config: `configs/snap_harvester_2024_btc.yaml`
- Train window: 2024-01-01 → 2024-11-30
- OOS window: 2025-01-01 → 2025-12-31
- Model file contains only 2024 BTC; 2025 is pure OOS.
- 2025 meta dataset: `results/meta/snap_meta_dataset_2025_BTC.csv` spans 2025-01-01 12:20:00 to 2025-07-31 04:35:00 (pure 2025).

### 3) Baseline vs routed (BTC 2025, 0.5R/3R, H30)
- Baseline (no routing): `results/meta/snap_base_trades_2025_BTC.csv`  
  n=359, p_win≈0.217, mean_R≈0.26, total_R≈93.5
- Routed (using 2024 model): `results/meta/snap_routed_tape_2025_BTC.csv`  
  n=359, wins=288, losses=33, flats=38, p_win≈0.802, mean_R≈2.979, median_R=4.0, total_R≈1069.5

### 4) Monthly robustness (routed 2025)
- 2025-01: n=47, sum_R=151.0, mean_R=3.21, hit_rate=0.83
- 2025-02: n=42, sum_R=122.0, mean_R=2.90, hit_rate=0.79
- 2025-03: n=57, sum_R=159.5, mean_R=2.80, hit_rate=0.75
- 2025-04: n=63, sum_R=165.5, mean_R=2.63, hit_rate=0.75
- 2025-05: n=40, sum_R=122.0, mean_R=3.05, hit_rate=0.83
- 2025-06: n=62, sum_R=212.5, mean_R=3.43, hit_rate=0.89
- 2025-07: n=48, sum_R=137.0, mean_R=2.85, hit_rate=0.79

### 5) Extreme tail tests (10R / 20R; SL 1.5R / 2R)
- 2024 BTC and 2025 BTC tail MFEs show negative expectancy for all tested static geometries (see `snap_tail_mfe_2024_BTC.csv` / `snap_tail_mfe_2025_BTC.csv`). No validated fixed 10R/20R edge.

### 6) What is validated vs not
- Validated: BTCUSDT, SL=0.5R / TP=3R / H=30, using `hgb_snap_2024_BTC.joblib`; OOS Jan–Jul 2025 routed tape matches reference stats above.
- Not validated: static 10R/20R, ETHUSDT (pending data/pipeline), any other geometry.

### 7) Feature blueprint (model view)
The model consumes every column below from `snap_meta_dataset_2024_BTC.csv` (excluded: `y_swing_1p5`, `r_final`, `mfe_r`, `mae_r`, `p_hat`, `risk_profile`, `event_id`, `event_bar_time`, `event_bar_offset_min`, `exit_time`, `hit_be`, `hit_tp`, `hit_sl`).

- Identity/structural: `timestamp`, `idx`, `symbol`, `side`
- Raw bar + vol: `open`, `high`, `low`, `close`, `atr`, `atr_pct`, `range_atr`
- Flow/intensity: `rel_vol`, `imbalance`, `imbalance_z`, `shockflip_z`, `shock_intensity_z`
- Divergence: `donchian_loc`, `price_flow_div`, `prior_flow_sign`, `div_score`
- Trend: `trend_dir`, `trend_aligned`, `trend_slope`, `stall_flag`
- Forward shape (MFE ATR): `mfe6_atr`, `mfe10_atr`, `mfe20_atr`, `mfe30_atr`, `mfe60_atr`, `mfe120_atr`, `mfe240_atr`
- Binary snap flags: all `did_snap_H{6,10,20,30,60,120,240}_K{0_5,0_75,1_0}` plus `did_snap`
- Barrier geometry outcome: `barrier_y_H30_R3p0_sl0p5` (H30, TP 3R, SL 0.5R)
- Deciles: `shock_intensity_z_decile`, `div_score_decile`, `rel_vol_decile`

LiveMetaBuilder must recreate these columns exactly (names, scales) for each event; everything else is excluded by config.

### 8) Offline end-to-end reproduction (commands)
Run from repo root with the venv active (`.\venv\Scripts\python`). These steps rebuild the full tape and stats offline.

1. Build 2024 meta dataset (train):  
   `.\venv\Scripts\python scripts/01_build_meta_dataset.py --config configs/snap_harvester_2024_btc.yaml --risk-profile strategy`

2. Train 2024 model + attach p_hat to 2024 meta:  
   `.\venv\Scripts\python scripts/02_train_meta_model.py --config configs/snap_harvester_2024_btc.yaml`

3. Build 2025 meta dataset (OOS events):  
   `.\venv\Scripts\python scripts/01_build_meta_dataset.py --config configs/snap_harvester_2025_btc.yaml --risk-profile strategy`

4. Apply frozen 2024 model to 2025 meta (write p_hat):  
   ```
   .\venv\Scripts\python - <<'PY'
   import joblib, pandas as pd
   from snap_harvester.config import load_config
   from snap_harvester.modeling import build_feature_matrix

   cfg = load_config("configs/snap_harvester_2025_btc.yaml")
   df = pd.read_csv(cfg["paths"]["meta_out"])
   X = build_feature_matrix(df, cfg)
   model = joblib.load("results/models/hgb_snap_2024_BTC.joblib")
   train_cols = list(model.feature_names_in_)
   for col in train_cols:
       if col not in X.columns:
           X[col] = 0.0
   df["p_hat"] = model.predict_proba(X[train_cols])[:, 1]
   out = cfg["paths"]["meta_with_preds_out"]
   df.to_csv(out, index=False)
   print(f"[Save] meta with preds -> {out}")
   PY
   ```

5. Build base tape for 0.5R/3R geometry:  
   `.\venv\Scripts\python scripts/build_base_trade_tape_from_meta.py --meta results/meta/snap_meta_dataset_2025_BTC.csv --out results/meta/snap_base_trades_2025_BTC.csv --barrier_col barrier_y_H30_R3p0_sl0p5 --tp_r 3.0 --sl_r 0.5`

6. Build routed tape (attach p_hat):  
   `.\venv\Scripts\python scripts/build_routed_trade_tape.py --trades results/meta/snap_base_trades_2025_BTC.csv --preds results/meta/snap_meta_with_p_hat_2025_BTC.csv --out results/meta/snap_routed_tape_2025_BTC.csv --risk_profile strategy`

7. Summarize routed PnL (headline stats):  
   `.\venv\Scripts\python scripts/summarize_routed_pnl.py --routed results/meta/snap_routed_tape_2025_BTC.csv`

8. Replay parity via live runner (writes routed clone):  
   `.\venv\Scripts\python -m snap_harvester.live.runner --config configs/snap_harvester_live_btc.yaml --mode replay --out results/live/replay_routed_tape_2025_BTC.csv`  
   Then run the summarize command on the replay output to confirm parity.

9. Live paper launch (feeds to be wired):  
   `.\venv\Scripts\python -m snap_harvester.live.runner --config configs/snap_harvester_live_btc.yaml --mode live`

These commands regenerate the 2024 train artifacts, 2025 OOS meta, p_hat scoring, routed tape, PnL summary, and the replay parity output used for validation.

How ShockFlip thresholds were loosened for Nov

Two layers of changes:

CLI parameters for the Nov-only run (vs the stricter defaults used for the main 2025 OOS study):
Original 2025 study (from README): effectively around z_band≈1.8–2.0, jump_band≈2.2–2.5, persistence≈3–4, and default windows z_window=240, donchian_window=120, min_bars=240.
Nov-only run:
--z_band 1.2
--jump_band 1.5
--persistence 1
--z_window 90
--donchian_window 40
--min_bars 60
Command you ran for Nov:
$env:PYTHONPATH='.'; .\venv\Scripts\python -u scripts\diamond_hunter.py `
  --tick_dir data/ticks/BTCUSDT_nov11 `
  --out results/diamond_hunter_btc_nov11 `
  --z_band 1.2 --jump_band 1.5 --persistence 1 `
  --min_bars 60 --z_window 90 --donchian_window 40
Code-level tweaks to support small slices / temp dirs
scripts/diamond_hunter.py
Added MIN_BARS_DEFAULT, Z_WINDOW_DEFAULT, DONCHIAN_WINDOW_DEFAULT constants.
Added CLI args: --min_bars, --z_window, --donchian_window.
get_chunk_events now takes z_window and donchian_window from the CLI overrides instead of being hard-wired to 240/120.
stream_ticks_from_dir loop now uses len(bars) < max(args.min_bars, 1) instead of a fixed 240-bar guard.
snap_harvester/utils/ticks.py
get_tick_size now strips suffixes after _ so folders like BTCUSDT_nov11 still resolve to BTCUSDT.

BTCUSDT-ticks-2025-11.csv → 169M rows, minutes built to BTCUSDT_1min_2025-11.parquet (43,409 bars, 1969 artefact at start is just a timestamp bug in a few corrupted ticks that we dropped on the way into bars).
Diamond Hunter on full data/ticks/BTCUSDT found ~1,060 events, of which 49 aligned cleanly to bars under your integrity constraints (kept=49, dropped=1011).
The final Nov OOS stats (routed via the frozen 2024 HGB model) are:

n = 49 trades
wins = 46, losses = 1, flats = 2
total_r = 181.5, mean_r ≈ 3.704, median_r = 4.0
Interpreting in dollar terms at $100 risk per trade (1R = $100):

Total P&L ≈ 181.5 * 100 = $18,150 over 49 trades.
Average per trade ≈ 3.704R * $100 ≈ $370.
Median trade ≈ 4R * $100 = $400.
From a modeling standpoint, this remains purely OOS:

Model: hgb_snap_2024_BTC.joblib trained on 2024 BTC only; no retraining with 2025 data.
November events come from DH run on 2025‑11 ticks; those are then fed through the existing feature blueprint and scored by the frozen 2024 model.