"""
Run Diamond Hunter for BTCUSDT and ETHUSDT in one go.

This assumes you already have `scripts/diamond_hunter.py` (the original pipeline)
available in this repository. The helper simply orchestrates two runs with
sensible defaults and writes per-symbol outputs:

- results/diamond_hunter_btc/
- results/diamond_hunter_eth/

Example:
    python scripts/run_diamond_hunter_batch.py

Override params if needed:
    python scripts/run_diamond_hunter_batch.py \\
        --btc-tick-dir data/ticks/BTCUSDT \\
        --eth-tick-dir data/ticks/ETHUSDT \\
        --z-band 1.8 --jump-band 2.2 --persistence 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def build_cmd(
    script_path: Path,
    tick_dir: Path,
    out_dir: Path,
    z_band: float,
    jump_band: float,
    persistence: int,
    extra: List[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--tick_dir",
        str(tick_dir),
        "--out",
        str(out_dir),
        "--z_band",
        str(z_band),
        "--jump_band",
        str(jump_band),
        "--persistence",
        str(persistence),
    ]
    cmd.extend(extra)
    return cmd


def run_one(symbol: str, tick_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    script_path = Path(args.dh_script)
    if not script_path.exists():
        raise FileNotFoundError(f"diamond_hunter script not found at {script_path}.")

    if not tick_dir.exists():
        raise FileNotFoundError(f"Tick directory for {symbol} not found: {tick_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    extra = args.extra if args.extra else []

    cmd = build_cmd(
        script_path=script_path,
        tick_dir=tick_dir,
        out_dir=out_dir,
        z_band=args.z_band,
        jump_band=args.jump_band,
        persistence=args.persistence,
        extra=extra,
    )
    print(f"[{symbol}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[{symbol}] Done. Outputs under: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for Diamond Hunter (BTC + ETH).")
    parser.add_argument("--dh-script", default="scripts/diamond_hunter.py", help="Path to diamond_hunter.py")
    parser.add_argument("--btc-tick-dir", default="data/ticks/BTCUSDT", help="BTC tick directory")
    parser.add_argument("--eth-tick-dir", default="data/ticks/ETHUSDT", help="ETH tick directory")
    parser.add_argument("--btc-out", default="results/diamond_hunter_btc", help="BTC output directory")
    parser.add_argument("--eth-out", default="results/diamond_hunter_eth", help="ETH output directory")
    parser.add_argument("--z-band", type=float, default=1.8, help="z_band parameter for Diamond Hunter")
    parser.add_argument("--jump-band", type=float, default=2.2, help="jump_band parameter for Diamond Hunter")
    parser.add_argument("--persistence", type=int, default=3, help="persistence parameter for Diamond Hunter")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Any extra args passed through to diamond_hunter.py (e.g., --config ...).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_one("BTCUSDT", Path(args.btc_tick_dir), Path(args.btc_out), args)
    run_one("ETHUSDT", Path(args.eth_tick_dir), Path(args.eth_out), args)


if __name__ == "__main__":
    main()
