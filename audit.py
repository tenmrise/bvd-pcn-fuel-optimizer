#!/usr/bin/env python3
"""
Refuel audit logger.

When the fleet watcher detects a REFUELED event for a truck that has an
active trip registered (see trips.py), this module is called to:

  1. Reverse-geocode the truck's GPS to a "City, PR" string
  2. Find the nearest BVD station and its current-PDF price
  3. Estimate how much was paid (litres added * nearest-BVD price)
  4. Look at the trip's last optimizer plan and decide whether this fuel
     stop matches the planned next fill (or not)
  5. Append one JSONL record to ~/BVD_PCN/optimizer/refuel_audit.jsonl

Records can be analyzed later by reading the JSONL file (jq, pandas, etc.).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

AUDIT_LOG = Path.home() / "BVD_PCN" / "optimizer" / "refuel_audit.jsonl"
MAX_BVD_DISTANCE_KM = 15.0   # if nearest BVD is further than this, treat as off-network


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def find_nearest_bvd(lat: float, lon: float) -> Optional[dict]:
    """Match GPS to closest BVD station from today's PDF."""
    import optimize as opt
    today = dt.date.today().isoformat()
    pdf = opt.find_pdf_for_date(today)
    if pdf is None:
        return None
    stations = opt.parse_pdf(pdf)
    # Quick geocoding pass — most stations are cached
    nearest = None
    best = float("inf")
    for s in stations:
        coord = opt.geocode(f"{s.city}, {s.prov}, Canada")
        if not coord:
            continue
        d = haversine_km((lat, lon), coord)
        if d < best:
            best = d
            nearest = s
    if not nearest:
        return None
    return {
        "site": nearest.site,
        "name": nearest.name,
        "city": nearest.city,
        "prov": nearest.prov,
        "price": nearest.price,                    # economic (decision) $/L
        "price_cash": getattr(nearest, "price_cash", nearest.price),  # tax-in $/L
        "distance_km": round(best, 2),
        "pdf_date": pdf.stem.split("_")[0],
    }


def audit_refuel_event(
    truck: dict,
    before_pct: float,
    after_pct: float,
    *,
    tank_l: float = 1000.0,
) -> dict:
    """Build and append one audit record for a REFUELED event.

    `truck` is the full device record from FleetHunt /api/devices.
    `before_pct` and `after_pct` are the fuel %% snapshots just before and
    after the jump as detected by the watcher.

    Returns the record (also written to AUDIT_LOG as JSONL).
    """
    import trips as trips_mod
    import fleet as fleet_mod
    import optimize as opt

    name = str(truck.get("name") or truck.get("id"))
    lat, lon = float(truck["latitude"]), float(truck["longitude"])
    litres_added = round((after_pct - before_pct) / 100.0 * tank_l, 1)

    # Where did it happen?
    try:
        location = opt.reverse_geocode(lat, lon) or "?"
    except Exception:
        location = "?"

    # Closest BVD + estimated cost
    nearest = find_nearest_bvd(lat, lon)
    estimated_cost = None
    on_network = False
    if nearest is not None:
        on_network = nearest["distance_km"] <= MAX_BVD_DISTANCE_KM
        if on_network:
            # Cash estimate — what actually hits the card is the tax-in price
            estimated_cost = round(litres_added * nearest["price_cash"], 2)

    # Active trip context + optimizer plan from when the trip was registered
    trip = trips_mod.get_trip(name)
    comparison: dict = {}
    if trip:
        planned_fills = trip.get("last_plan_fills") or []
        # Find planned next fill ≥ truck's GPS (rough: smallest km > 0 after origin)
        # We don't have current km on chain, but we can match by site if nearest
        # BVD matches one of the planned sites.
        if nearest and on_network:
            match = next((f for f in planned_fills if f.get("site") == nearest["site"]), None)
            if match:
                comparison["matches_planned_site"] = True
                comparison["planned_litres_at_site"] = match.get("litres")
                comparison["planned_price_at_site"] = match.get("price")
                if match.get("price"):
                    delta_per_l = nearest["price"] - match["price"]
                    comparison["delta_per_liter"] = round(delta_per_l, 4)
                    comparison["delta_total"] = round(delta_per_l * litres_added, 2)
            else:
                comparison["matches_planned_site"] = False
                # Find cheapest planned fill site for reference
                cheapest = min(planned_fills, key=lambda f: f.get("price", 999),
                               default=None)
                if cheapest:
                    comparison["cheapest_planned_site"] = cheapest.get("site")
                    comparison["cheapest_planned_price"] = cheapest.get("price")
                    if nearest.get("price"):
                        delta = nearest["price"] - cheapest["price"]
                        comparison["overpay_vs_cheapest_planned_per_liter"] = round(delta, 4)
                        comparison["overpay_total"] = round(delta * litres_added, 2)

    record = {
        "event_time": dt.datetime.now().isoformat(timespec="seconds"),
        "truck": {
            "id": truck.get("id"),
            "name": name,
            "vin": truck.get("vin"),
            "lat": lat,
            "lon": lon,
            "location": location,
            "odometer": truck.get("odometer"),
            "device_time": truck.get("dt_tracker"),
        },
        "refuel": {
            "fuel_pct_before": before_pct,
            "fuel_pct_after": after_pct,
            "litres_added_estimated": litres_added,
            "tank_assumed_L": tank_l,
        },
        "nearest_bvd": nearest,
        "estimated_cost_paid": estimated_cost,
        "on_bvd_network": on_network,
        "active_trip": trip,
        "comparison": comparison if trip else None,
    }
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def main() -> int:
    """CLI: tail the audit log."""
    import argparse
    ap = argparse.ArgumentParser(description="Inspect refuel audit log")
    ap.add_argument("--tail", type=int, default=10,
                    help="Show last N records (default 10)")
    ap.add_argument("--truck", help="Filter to one truck name")
    args = ap.parse_args()
    if not AUDIT_LOG.exists():
        print(f"(no audit log yet at {AUDIT_LOG})")
        return 0
    records = [json.loads(l) for l in AUDIT_LOG.read_text().splitlines() if l.strip()]
    if args.truck:
        records = [r for r in records if r["truck"]["name"] == args.truck]
    for r in records[-args.tail:]:
        print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
