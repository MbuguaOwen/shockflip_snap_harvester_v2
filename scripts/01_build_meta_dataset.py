import argparse
from pathlib import Path
import sys

# Allow running the script directly from the repo root by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.config import load_config
from snap_harvester.logging_utils import get_logger
from snap_harvester.meta_builder import build_meta_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Snap Harvester meta dataset.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/snap_harvester_example.yaml).",
    )
    parser.add_argument(
        "--risk-profile",
        help="Optional risk preset name from config.risk_presets (e.g. geometry or strategy).",
    )
    args = parser.parse_args()

    logger = get_logger("build_meta_dataset")
    cfg = load_config(args.config)

    # Select risk preset if provided
    risk_presets = cfg.get("risk_presets")
    if risk_presets:
        chosen = args.risk_profile or cfg.get("active_risk_profile") or next(iter(risk_presets.keys()))
        if chosen not in risk_presets:
            raise ValueError(f"risk profile '{chosen}' not found in config.risk_presets")
        cfg["risk"] = risk_presets[chosen]
        cfg["active_risk_profile"] = chosen
        logger.info("Using risk profile '%s'", chosen)

    meta_df = build_meta_dataset(cfg, logger=logger)

    out_path = Path(cfg["paths"]["meta_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(out_path, index=False)
    logger.info("Saved meta dataset -> %s (n=%d)", out_path, len(meta_df))


if __name__ == "__main__":
    main()
