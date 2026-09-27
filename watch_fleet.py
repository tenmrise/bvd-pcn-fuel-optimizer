#!/usr/bin/env python3
"""
Poll FleetHunt for the whole fleet and emit a Monitor event per truck per
state change. One /api/devices call covers all trucks (vs one call per truck
for watch_truck.py), so it scales to the entire fleet at constant API cost.

Events fired:
  - movement:  state idle/stopped <-> moving, or speed 0 <-> >0
  - ignition:  OFF -> ON or ON -> OFF
  - refuel:    fuel jump up >= 5 percentage points (next poll)
  - drop:      fuel drop >= 5 percentage points (sensor blip or theft)

Usage:
  python3 watch_fleet.py                          # all trucks with fuel sensor
  python3 watch_fleet.py --truck 3006 --truck 3003   # subset only
  python3 watch_fleet.py --interval 30            # poll every 30s
  python3 watch_fleet.py --once                   # one-shot status, no loop
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import fleet  # noqa: E402


@dataclass
class State:
    state: str
    speed: int
    ign: int
    fuel: Optional[float]   # %
    fix: str


def snap(d: dict) -> State:
    sensors = d.get("sensors") or {}
    raw = sensors.get("fuel_percentage") or sensors.get("fuel")
    try:
        fuel = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        fuel = None
    return State(
        state=(d.get("state") or "").lower(),
        speed=int(d.get("speed") or 0),
        ign=int(d.get("ignition") or 0),
        fuel=fuel,
        fix=d.get("dt_tracker") or "",
    )


def diff_events(prev: State, cur: State) -> list[str]:
    """Return a list of human-readable change strings for one truck."""
    out = []
    # movement
    prev_moving = prev.state == "moving" or prev.speed > 0
    cur_moving  = cur.state  == "moving" or cur.speed  > 0
    if cur_moving and not prev_moving:
        out.append(f"STARTED   state={prev.state}->{cur.state}  speed={cur.speed}")
    elif prev_moving and not cur_moving:
        out.append(f"STOPPED   state={prev.state}->{cur.state}  speed={cur.speed}")
    elif prev.state != cur.state:
        out.append(f"state     {prev.state}->{cur.state}")
    # ignition
    if cur.ign and not prev.ign:
        out.append("IGN-ON")
    elif prev.ign and not cur.ign:
        out.append("IGN-OFF")
    # fuel — only treat as a REAL refuel if the jump is big AND the truck
    # is/was stationary. Otherwise sensor oscillation while driving fires
    # the threshold every poll (3003 oscillated 36-50% several times in 45
    # minutes on 2026-05-18, polluting the audit log).
    if prev.fuel is not None and cur.fuel is not None:
        dp = cur.fuel - prev.fuel
        stationary = (prev.speed == 0 and cur.speed == 0)
        if dp >= 15.0 and stationary:
            out.append(f"REFUELED  +{dp:.1f}% (was {prev.fuel:.0f}%, now {cur.fuel:.0f}%)")
        elif dp <= -15.0 and stationary:
            # large drop while parked is unusual — log but quietly
            out.append(f"FUEL-DROP {dp:.1f}% (was {prev.fuel:.0f}%, now {cur.fuel:.0f}%)")
        # quiet sensor oscillations during driving are filtered out entirely
    return out


def line(name: str, evt: str, cur: State) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    fuel_s = f"{cur.fuel:.0f}%" if cur.fuel is not None else "?%"
    return (f"{ts}  {name:<8}  {evt:<46} "
            f"fuel={fuel_s:<4}  last_fix={cur.fix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truck", action="append", default=[],
                    help="Limit to one or more trucks (repeatable). "
                         "Match by name, id, VIN, or plate. "
                         "If omitted, watches every truck that reports a fuel sensor.")
    ap.add_argument("--interval", type=int, default=60,
                    help="Poll interval seconds (default 60)")
    ap.add_argument("--once", action="store_true",
                    help="Print one snapshot of fleet state, then exit")
    ap.add_argument("--include-no-fuel", action="store_true",
                    help="Include trailer cameras / units with no fuel sensor")
    ap.add_argument("--no-audit", action="store_true",
                    help="Skip writing refuel events to the audit log")
    ap.add_argument("--tank", type=float, default=1000.0,
                    help="Tank size in L used for litres-added math (default 1000)")
    args = ap.parse_args()

    token = fleet.load_token()
    prev: dict[str, State] = {}
    first_pass = True

    while True:
        try:
            devices = fleet.fetch_devices(token)
        except Exception as e:
            # Redact the API token from any URL the exception may quote
            msg = str(e).replace(token, "***REDACTED***")
            print(f"{datetime.now().strftime('%H:%M:%S')}  ERROR fetching: {msg}",
                  flush=True)
            if args.once:
                return 2
            time.sleep(args.interval)
            continue

        # Filter
        roster: dict[str, State] = {}
        device_by_name: dict[str, dict] = {}
        for d in devices:
            name = d.get("name") or str(d.get("id"))
            if args.truck and not any(
                str(d.get(f, "")).lower() == t.lower() or
                (d.get(f) and str(d.get(f)) == t)
                for t in args.truck
                for f in ("name", "id", "vin", "license_plate_no")
            ):
                continue
            cur = snap(d)
            if cur.fuel is None and not args.include_no_fuel:
                continue
            roster[name] = cur
            device_by_name[name] = d

        for name, cur in roster.items():
            if first_pass:
                print(line(name, f"initial state={cur.state} speed={cur.speed} "
                                 f"ign={cur.ign}", cur), flush=True)
            else:
                pv = prev.get(name)
                if pv is None:
                    print(line(name, "NEW (first sighting)", cur), flush=True)
                else:
                    for evt in diff_events(pv, cur):
                        print(line(name, evt, cur), flush=True)
                        # Audit hook: when a REFUELED event fires, record it.
                        if evt.startswith("REFUELED") and not args.no_audit:
                            try:
                                import audit
                                import truck_config
                                # per-truck tank, falls back to --tank then default
                                per_truck = truck_config.tank_l(name)
                                rec = audit.audit_refuel_event(
                                    device_by_name[name],
                                    pv.fuel, cur.fuel,
                                    tank_l=per_truck or args.tank,
                                )
                                tag = ("matches plan"
                                       if (rec.get("comparison") or {}).get("matches_planned_site")
                                       else "off-plan" if rec.get("active_trip") else "no-trip")
                                cost = rec.get("estimated_cost_paid")
                                cost_s = f"~${cost:.2f}" if cost is not None else "$?"
                                bvd = (rec.get("nearest_bvd") or {}).get("name", "?")
                                print(line(name,
                                           f"  audit: {tag}  bvd={bvd}  paid={cost_s}",
                                           cur), flush=True)
                            except Exception as e:
                                print(line(name, f"  audit FAILED: {e}", cur),
                                      flush=True)
            prev[name] = cur

        first_pass = False
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
