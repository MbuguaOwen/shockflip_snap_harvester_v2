# ShockFlip Research Vault Project

This bundle contains the **current ShockFlip research stack** including:

- Core engine (`core/`):
  - Orderflow and feature pipeline
  - ShockFlip detector
  - Backtest + management logic (breakeven + zombie exits)
  - Event-study engine
  - Parity replay harness
- Research scripts (`scripts/`):
  - `divergence_map.py` – stream-safe divergence heatmap over flow vs price
  - `diamond_hunter.py` – Diamond subset discovery and rel_vol threshold printer
  - `run_backtest.py` – ShockFlip backtest with management
  - `run_event_study.py` – event-study driver for ShockFlip
  - `run_parity_replay.py` – research vs live-style parity check
  - `run_shockflip_sweep.py` – micro-grid parameter sweep
  - `compute_pf_by_side.py` – PF breakdown by side
  - `analyze_v13_filters.py` – H1–H7 hypothesis analysis helper
  - `plot_trades.py` – single-trade visualization helper
- Configs (`configs/`):
  - ShockFlip strategy configs including the Diamond research config.
- Vault (`vault/`):
  - `shockflip_blueprint.md` – the Success Blueprint / physics notebook.

## Usage

From the project root (directory that contains `core/`, `scripts/`, `configs/`):

```bash
# Example: run Divergence Map (BTCUSDT)
python scripts/divergence_map.py --tick_dir data/ticks/BTCUSDT --out results/divergence_map

# Example: run Diamond Hunter v2.2
python scripts/diamond_hunter.py --tick_dir data/ticks/BTCUSDT --out results/diamond_hunter \
    --z_band 1.8 --jump_band 2.2 --persistence 3

# Example: backtest with current ShockFlip config
python scripts/run_backtest.py --config configs/strategies_shockflip_only.yaml

# Example: event study
python scripts/run_event_study.py --config configs/strategies_shockflip_only.yaml

# Example: parity replay
python scripts/run_parity_replay.py --config configs/strategies_shockflip_only.yaml
```

Make sure your tick data is under `data/ticks/SYMBOL/` and your Python path includes the project root.


1. .\venv\Scripts\python.exe scripts/divergence_map.py `
  --tick_dir data/ticks/BTCUSDT `
  --out results/divergence_map


2. .\venv\Scripts\python.exe scripts/diamond_hunter.py `
  --tick_dir data/ticks/BTCUSDT `
  --out results/diamond_hunter `
  --z_band 1.8 `
  --jump_band 2.2 `
  --persistence 3

3. .\venv\Scripts\python.exe scripts/run_backtest.py `
  --config configs/strategies_shockflip_only.yaml `
  --out results/backtests/shockflip_only.csv

4. .\venv\Scripts\python.exe scripts/run_event_study.py `
 --config configs/strategies_shockflip_only.yaml `
  --events_out results/event_study/events.csv `
  --summary_out results/event_study/summary.csv

## Snap Harvester v2 – Validation Snapshot

What is actually validated
- Data integrity + pipeline wiring: BTC+ETH ticks → minutes are clean (Jan–Sep). Timestamps normalized to UTC ms (including ETH 2025-08/09 which were originally in microseconds). Duplicates removed. 1m OHLCV built with price tolerance and gap logging (1970→2025 gap warning is expected from junk ticks). Outputs: `data/minutes/BTCUSDT_1min.parquet` (Jan–Jul), `data/minutes/ETHUSDT_1min.parquet` (Jan–Sep).
- Events → bars alignment: `align_events_to_bars()` enforces price ∈ [low, high]. For ETH Aug–Sep, 46 bad events are logged and dropped (`drop_misaligned_events=true`). Final meta: 736 events (379 BTC + 357 ETH), all aligned.
- Combined events consistency: `build_combined_events.py` normalizes symbols/sides, converts timestamps to UTC, dedups on (symbol, timestamp, side), and checks required columns. Output: `results/diamond_hunter/events_annotated.csv` (BTC + ETH).
- Risk template + labels: Strategy profile = `snap_base_2p5x4` (SL 2.5R, TP 4R, BE 1R). Same parameters used for label construction (`y_swing_1p5`) and backtest geometry. No SL/TP mismatch.
- Model training + OOS (Jan–Jul): Time split train Jan–May, OOS Jun–Jul. Train size 482, OOS 221. Features avoid future leakage. OOS AUC ≈ 0.60 (modest, not magic). Jun–Jul backtest under 2.5R/4R: naive win ~86.9%, avg R ~3.36R, total ~+743R; router improves hit rate/avg R as threshold rises.
- Shadow ETH Aug–Sep (out-of-sample-of-out-of-sample): ETH ticks fixed (µs→ms), minutes to 2025-09-30, 46 misaligned ETH events dropped. Shadow eval 2025-08-01→2025-09-30: naive N=23, win≈69.6%, avg R≈+2.46R, total≈+56.5R; router p̂≥0.5 N=13, win≈76.9%, avg R≈+2.69R, total≈+35R. Fresh data not used in geometry or original OOS.

What is not validated yet
- No multi-year or full regime coverage (only 7 months + 2-month ETH shadow).
- Template choice (2.5R/4R) picked using Jan–Jul geometry; could be tuned to that year.
- Live frictions (fees/spread/slippage/latency) not fully stress-tested.
- No rolling walk-forward; fixed train Jan–May, validate Jun–Jul, shadow Aug–Sep.
- No cross-exchange/venue robustness checks (Binance-style only).

Blunt summary
- Pipeline is clean (no leakage, no silent corruption). The 2.5R/4R ShockFlip swing template shows real, out-of-sample profits on BTC/ETH 2025, but long-term/regime robustness is still unproven.


GLOBAL COMMAND: For converting trades to ticks
ForEach ($m in 1..12) {
  $mm = $m.ToString("00")
  $in  = "raw/BTCUSDT-trades-2024-$mm.csv"
  $out = "data/ticks/BTCUSDT/BTCUSDT-ticks-2024-$mm.csv"
  if (-not (Test-Path $in)) { Write-Host "skip $in (missing)"; continue }
  .\venv\Scripts\python.exe scripts/convert_binance_ticks.py `
    --in_path $in `
    --out_path $out `
    --mode trades
}
