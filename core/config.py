import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config and resolve minimal `include` semantics for `data`.

    If the config has:

    data:
      include: configs/data.yaml

    we replace `config['data']` with the contents of that include file.
    Paths are resolved relative to the main config file.
    """
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")

    base_dir = os.path.dirname(path)

    # Minimal include support for data section
    data = cfg.get("data")
    if isinstance(data, dict) and "include" in data:
        inc_rel = data["include"]
        inc_path = os.path.join(base_dir, inc_rel)
        with open(inc_path, "r", encoding="utf-8") as f:
            included = yaml.safe_load(f)
        if included is None:
            included = {}
        if not isinstance(included, dict):
            raise ValueError("Included data config must be a mapping")

        overrides = {k: v for k, v in data.items() if k != "include"}
        merged = {**included, **overrides}
        cfg["data"] = merged

    return cfg
