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
