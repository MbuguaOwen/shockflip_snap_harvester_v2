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
