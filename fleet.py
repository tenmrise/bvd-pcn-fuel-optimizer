#!/usr/bin/env python3
"""
FleetHunt API client.

Uses GET /api/devices (the rich endpoint) — returns ~80 fields per truck
including sensors.fuel_percentage. GET /api/fleet (documented) only gives
location, so we don't use it.

  Use:
    python3 fleet.py                       # list fleet with fuel% + location
    python3 fleet.py --truck 3006          # show one truck (full info)
    python3 fleet.py --truck 3006 --as-from   # print only "City, PR"
    python3 fleet.py --truck 3006 --start-fuel-for-tank 1000
                                              # print only starting fuel litres
    python3 fleet.py --truck 3006 --json   # raw API response for one truck

Token resolution order:
  1. env var FLEETHUNT_API_KEY
  2. ~/BVD_PCN/optimizer/.fleethunt-token (chmod 600)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent))
import truck_config  # noqa: E402

FLEETHUNT_BASE = "https://app.fleethunt.ca/api"
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
# OSM usage policy asks for a contact; set BVD_CONTACT_EMAIL to include one.
_CONTACT = os.environ.get("BVD_CONTACT_EMAIL")
USER_AGENT = f"BVD-PCN-FuelOpt/1.0 ({_CONTACT})" if _CONTACT else "BVD-PCN-FuelOpt/1.0"
TOKEN_FILE = Path.home() / "BVD_PCN" / "optimizer" / ".fleethunt-token"

PROV_MAP = {
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB",
    "New Brunswick": "NB", "Nouveau-Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS", "Nouvelle-Écosse": "NS",
    "Ontario": "ON", "Prince Edward Island": "PE",
    "Île-du-Prince-Édouard": "PE",
    "Quebec": "QC", "Québec": "QC",
    "Saskatchewan": "SK", "Yukon": "YT",
    "Northwest Territories": "NT", "Nunavut": "NU",
}


# --------------------------------------------------------------------------


def load_token() -> str:
    env = os.environ.get("FLEETHUNT_API_KEY")
    if env:
        return env.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    raise SystemExit(
        f"No FleetHunt token.\n"
        f"  Set env var FLEETHUNT_API_KEY or write the key to {TOKEN_FILE} (chmod 600)."
    )


def fetch_devices(token: str) -> list[dict]:
    """All fleet devices with full sensor payload."""
    r = requests.get(
        f"{FLEETHUNT_BASE}/devices",
        params={"api_token": token},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != 1:
        raise RuntimeError(f"FleetHunt status {data.get('status')}: {data}")
    return data.get("devices", [])


def find_device(devices: list[dict], identifier: str) -> Optional[dict]:
    """Match by id (numeric), name, vin, or license_plate_no (case-insensitive)."""
    ident = identifier.strip().lower()
    for d in devices:
        for field in ("id", "name", "vin", "license_plate_no"):
            v = d.get(field)
            if v is not None and str(v).strip().lower() == ident:
                return d
    return None


def fuel_pct(device: dict) -> Optional[float]:
    """Extract fuel percentage from device.sensors.fuel_percentage."""
    sensors = device.get("sensors") or {}
    v = sensors.get("fuel_percentage")
    if v is None:
        v = sensors.get("fuel")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def starting_fuel_l(device: dict, tank_capacity_l: float) -> Optional[float]:
    """Convert reported fuel % into litres for a given tank capacity."""
    pct = fuel_pct(device)
    if pct is None:
        return None
    return pct / 100.0 * tank_capacity_l


# --------------------------------------------------------------------------
# Reverse geocoding (lat/lon -> "City, PR")
# --------------------------------------------------------------------------


_last_nominatim = 0.0


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Resolve a coordinate to 'City, PR' string. Rate-limited (1 req/sec)."""
    global _last_nominatim
    elapsed = time.time() - _last_nominatim
    if elapsed < 1.05:
        time.sleep(1.05 - elapsed)
    r = requests.get(
        NOMINATIM,
        params={"lat": lat, "lon": lon, "format": "json", "zoom": 10,
                "accept-language": "en"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    _last_nominatim = time.time()
    r.raise_for_status()
    data = r.json()
    addr = data.get("address", {})
    city = (addr.get("city") or addr.get("town") or addr.get("village")
            or addr.get("hamlet") or addr.get("municipality")
            or addr.get("county"))
    prov = addr.get("state")
    if not city or not prov:
        return None
    return f"{city}, {PROV_MAP.get(prov, prov)}"


# --------------------------------------------------------------------------


def get_truck_context(identifier: str, tank_capacity_l: Optional[float] = None) -> dict:
    """Return the bundle optimize.py wants when given --truck.

    Output:
      {
        "device":       full device dict from FleetHunt
        "from_city":    "City, PR" suitable for --from
        "fuel_pct":     float | None
        "start_fuel_l": float | None  (= fuel_pct / 100 * tank_capacity_l)
      }
    """
    token = load_token()
    devices = fetch_devices(token)
    d = find_device(devices, identifier)
    if not d:
        roster = ", ".join(
            f"{x.get('name')!r}(id {x.get('id')})" for x in devices
        )
        raise SystemExit(f"Truck {identifier!r} not in fleet. Available: {roster}")
    loc = reverse_geocode(d["latitude"], d["longitude"])
    if not loc:
        raise SystemExit(
            f"Could not reverse-geocode truck position "
            f"({d['latitude']}, {d['longitude']})"
        )
    pct = fuel_pct(d)
    # Per-truck tank from trucks.json (fallback to default 1000 L) if caller
    # didn't pass an explicit tank size
    effective_tank = (tank_capacity_l if tank_capacity_l is not None
                      else truck_config.tank_l(d.get("name")))
    return {
        "device": d,
        "from_city": loc,
        "fuel_pct": pct,
        "tank_l": effective_tank,
        "start_fuel_l": (pct / 100.0 * effective_tank) if pct is not None else None,
    }


# --------------------------------------------------------------------------


def _print_one(d: dict, loc: str, tank_l: float) -> None:
    pct = fuel_pct(d)
    fuel_l = (pct / 100.0 * tank_l) if pct is not None else None
    print(f"Truck {d.get('name')!r}  (id {d.get('id')}, VIN {d.get('vin')})")
    print(f"  Plate:      {d.get('license_plate_no', '-')}")
    print(f"  Location:   {loc}")
    print(f"              ({d['latitude']:.5f}, {d['longitude']:.5f})")
    print(f"  Fuel:       {pct:.1f}%  ->  {fuel_l:.0f} L at {tank_l:.0f} L tank"
          if pct is not None else "  Fuel:       (not reported)")
    print(f"  Speed:      {d.get('speed', 0)} {d.get('unit_of_speed', 'km/h').lower()}"
          f"   Heading {d.get('angle', 0)}°")
    print(f"  Ignition:   {'ON' if d.get('ignition') else 'OFF'}"
          f"   State: {d.get('state', '-')}"
          f"   ({d.get('status', '').strip()})")
    print(f"  Odometer:   {int(d.get('odometer') or 0):,} km")
    print(f"  Last fix:   {d.get('dt_tracker', '-')}  (server: {d.get('dt_server', '-')})")
    if d.get("mil"):
        print(f"  ! Check-engine MIL is ON")
    dtc = d.get("dtc") or []
    if dtc:
        print(f"  ! Active DTCs: {len(dtc)}")
        for code in dtc[:5]:
            seg = code.get("segment") if isinstance(code, dict) else str(code)
            print(f"      {seg}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--truck", help="Filter to one truck by name, id, VIN, or plate")
    ap.add_argument("--as-from", action="store_true",
                    help="Print only the 'City, PR' string (shell-friendly)")
    ap.add_argument("--start-fuel-for-tank", type=float, metavar="L",
                    help="Print only the starting fuel litres "
                         "(= fuel_percentage * this tank size / 100)")
    ap.add_argument("--tank", type=float, default=1000.0,
                    help="Tank size used for fuel %% -> L conversion (default 1000)")
    ap.add_argument("--json", action="store_true",
                    help="Emit raw API JSON instead of formatted output")
    args = ap.parse_args()

    token = load_token()
    devices = fetch_devices(token)

    if args.truck:
        d = find_device(devices, args.truck)
        if not d:
            print(f"Truck {args.truck!r} not found. Available:", file=sys.stderr)
            for x in devices:
                print(f"  {x.get('name')!r}  (id {x.get('id')})  VIN {x.get('vin')}",
                      file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(d, indent=2, default=str))
            return 0
        if args.start_fuel_for_tank is not None:
            l = starting_fuel_l(d, args.start_fuel_for_tank)
            print(f"{l:.0f}" if l is not None else "0")
            return 0
        if args.as_from:
            loc = reverse_geocode(d["latitude"], d["longitude"])
            print(loc or "?")
            return 0
        loc = reverse_geocode(d["latitude"], d["longitude"]) or "?"
        _print_one(d, loc, truck_config.tank_l(d.get("name")))
        return 0

    if args.json:
        print(json.dumps(devices, indent=2, default=str))
        return 0

    print(f"Fleet: {len(devices)} truck(s)  (per-truck tank sizes from trucks.json)\n")
    fmt = "{:<7} {:<8} {:>6} {:>9} {:>6} {:>11} {:>5} {:>8} {:<8} {:<19}  {}"
    print(fmt.format("ID", "Name", "Fuel%", "Fuel L", "TankL", "Odo km", "Speed", "State", "Ign",
                     "Last Fix", "Location"))
    print("-" * 152)
    for d in devices:
        pct = fuel_pct(d)
        truck_tank = truck_config.tank_l(d.get("name"))
        fuel_l = (pct / 100.0 * truck_tank) if pct is not None else None
        loc = reverse_geocode(d["latitude"], d["longitude"]) or "?"
        print(fmt.format(
            d.get("id"),
            (d.get("name") or "")[:8],
            f"{pct:.0f}" if pct is not None else "-",
            f"{fuel_l:.0f}" if fuel_l is not None else "-",
            f"{truck_tank:.0f}",
            f"{int(d.get('odometer') or 0):,}",
            d.get("speed", 0),
            (d.get("state") or "-")[:8],
            "ON" if d.get("ignition") else "OFF",
            (d.get("dt_tracker") or "")[:19],
            loc,
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
