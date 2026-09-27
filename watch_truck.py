#!/usr/bin/env python3
"""
Poll FleetHunt for one truck and emit a line whenever its state changes
in a way that signals it has started moving. Stops once `state == "moving"`
or `speed > 0`.

Each stdout line is a Monitor event:
  HH:MM:SS  <tag>  state=...  speed=...  ign=...  fuel=... %  last_fix=...

Usage:
    python3 watch_truck.py <truck-name-or-id> [--interval 60]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Import local fleet client
sys.path.insert(0, str(Path(__file__).parent))
import fleet  # noqa: E402


def snapshot(name_or_id: str) -> dict:
    token = fleet.load_token()
    devices = fleet.fetch_devices(token)
    d = fleet.find_device(devices, name_or_id) or {}
    sensors = d.get("sensors") or {}
    fuel = sensors.get("fuel_percentage") or sensors.get("fuel")
    try:
        fuel = float(fuel) if fuel is not None else None
    except (TypeError, ValueError):
        fuel = None
    return {
        "state": (d.get("state") or "").lower(),
        "speed": int(d.get("speed") or 0),
        "ign":   int(d.get("ignition") or 0),
        "fuel":  fuel,
        "fix":   d.get("dt_tracker") or "",
        "lat":   d.get("latitude"),
        "lon":   d.get("longitude"),
        "name":  d.get("name") or "?",
    }


def format_line(tag: str, s: dict) -> str:
    fuel_s = f"{s['fuel']:.0f}%" if s["fuel"] is not None else "?%"
    return (f"{datetime.now().strftime('%H:%M:%S')}  {tag:<9}  "
            f"state={s['state']:<8} speed={s['speed']:>3}  ign={s['ign']}  "
            f"fuel={fuel_s:<4}  last_fix={s['fix']}")


def is_moving(s: dict) -> bool:
    return s["state"] == "moving" or s["speed"] > 0


def changed(prev: dict, cur: dict) -> bool:
    if prev is None:
        return True
    if cur["state"] != prev["state"]:
        return True
    if cur["speed"] > 0 and prev["speed"] == 0:
        return True
    if cur["ign"] == 1 and prev["ign"] == 0:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("truck", help="Truck name, id, VIN, or plate")
    ap.add_argument("--interval", type=int, default=60,
                    help="Poll interval in seconds (default 60)")
    args = ap.parse_args()

    prev: dict | None = None
    while True:
        try:
            cur = snapshot(args.truck)
        except SystemExit:
            raise
        except Exception as e:
            print(f"{datetime.now().strftime('%H:%M:%S')}  error      {e}",
                  flush=True)
            time.sleep(args.interval)
            continue

        if prev is None:
            print(format_line("initial", cur), flush=True)
        elif changed(prev, cur):
            tag = "MOVING" if is_moving(cur) else "change"
            print(format_line(tag, cur), flush=True)

        if is_moving(cur):
            print(f"{datetime.now().strftime('%H:%M:%S')}  Truck {cur['name']!r} "
                  f"is moving — re-run optimize.py to refresh the plan.",
                  flush=True)
            return 0

        prev = cur
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
