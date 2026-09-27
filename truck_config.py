"""Per-truck configuration lookup (tank size, etc).

Single source of truth in ~/BVD_PCN/optimizer/trucks.json. Falls back to the
`_default` entry if a truck name isn't listed. All scripts (optimize.py,
fleet.py, audit.py, watch_fleet.py) should call `tank_l(name)` rather than
assuming 1000 L.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

CONFIG_FILE = Path.home() / "BVD_PCN" / "optimizer" / "trucks.json"
DEFAULT_TANK_L = 1000.0


_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if not CONFIG_FILE.exists():
        _cache = {}
        return _cache
    try:
        _cache = json.loads(CONFIG_FILE.read_text())
    except json.JSONDecodeError:
        _cache = {}
    return _cache


def tank_l(truck_name: Optional[str]) -> float:
    """Return the truck's tank capacity in litres. Falls back to default if
    the truck isn't listed or the name is empty/None."""
    cfg = _load()
    name = str(truck_name) if truck_name is not None else ""
    entry = cfg.get(name)
    if entry and "tank_l" in entry:
        return float(entry["tank_l"])
    default = cfg.get("_default", {})
    if "tank_l" in default:
        return float(default["tank_l"])
    return DEFAULT_TANK_L


def all_known() -> dict:
    """All explicitly-configured trucks, dict of name -> {tank_l, ...}."""
    return {k: v for k, v in _load().items() if not k.startswith("_")}
