#!/usr/bin/env python3
"""
Trip registry for the BVD fuel optimizer.

Each truck can have one "active trip" — a registered (from, to) pair so the
watcher knows what plan to score real-time refuel events against. The file
lives at ~/BVD_PCN/optimizer/trips.json. Every successful `optimize.py
--truck X --to Y` call auto-writes/updates the entry; this CLI is for
manual inspection or ending a trip early.

  Use:
    python3 trips.py list                       # show all active trips
    python3 trips.py show 3006                  # detail for one truck
    python3 trips.py end 3006                   # mark trip complete
    python3 trips.py clear                      # wipe registry
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

TRIPS_FILE = Path.home() / "BVD_PCN" / "optimizer" / "trips.json"


def load_trips() -> dict[str, dict]:
    """Read the registry. Returns {} if missing."""
    if not TRIPS_FILE.exists():
        return {}
    try:
        return json.loads(TRIPS_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def save_trips(trips: dict[str, dict]) -> None:
    TRIPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRIPS_FILE.write_text(json.dumps(trips, indent=2, sort_keys=True))


def upsert_trip(truck: str, trip: dict) -> None:
    """Register or update an active trip for `truck`. The dict shape is:
        {
          "from": "Brampton, ON",
          "to": "Winnipeg, MB",
          "start_date": "2026-05-17",
          "start_fuel": 120,
          "extra_args": ["--carry-forward", "--one-way"],
          "registered_at": "<iso-timestamp>",
          "updated_at":    "<iso-timestamp>",
        }
    """
    trips = load_trips()
    now = dt.datetime.now().isoformat(timespec="seconds")
    existing = trips.get(truck)
    if existing:
        trip = {**existing, **trip, "updated_at": now}
    else:
        trip = {**trip, "registered_at": now, "updated_at": now}
    trips[truck] = trip
    save_trips(trips)


def end_trip(truck: str) -> bool:
    trips = load_trips()
    if truck not in trips:
        return False
    del trips[truck]
    save_trips(trips)
    return True


def get_trip(truck: str) -> Optional[dict]:
    return load_trips().get(truck)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="list all active trips")
    sub.add_parser("clear", help="remove every trip from the registry")
    p_show = sub.add_parser("show", help="show detail for one truck")
    p_show.add_argument("truck")
    p_end = sub.add_parser("end", help="mark a truck's trip complete")
    p_end.add_argument("truck")
    args = ap.parse_args()

    if args.cmd == "list":
        trips = load_trips()
        if not trips:
            print("(no active trips)")
            return 0
        fmt = "{:<10} {:<22} -> {:<22}  start {:<10} fuel0={:>4}  updated {}"
        for name, t in sorted(trips.items()):
            print(fmt.format(
                name,
                t.get("from", "-")[:22],
                t.get("to", "-")[:22],
                t.get("start_date", "-"),
                t.get("start_fuel", "-"),
                t.get("updated_at", "-"),
            ))
        return 0
    if args.cmd == "show":
        trip = get_trip(args.truck)
        if not trip:
            print(f"no active trip for truck {args.truck!r}", file=sys.stderr)
            return 1
        print(json.dumps(trip, indent=2))
        return 0
    if args.cmd == "end":
        if end_trip(args.truck):
            print(f"ended trip for truck {args.truck!r}")
            return 0
        print(f"no active trip for truck {args.truck!r}", file=sys.stderr)
        return 1
    if args.cmd == "clear":
        save_trips({})
        print(f"registry wiped ({TRIPS_FILE})")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
