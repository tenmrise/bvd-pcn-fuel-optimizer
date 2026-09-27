#!/usr/bin/env python3
"""
BVD fuel route optimizer.

Computes the minimum-cost fueling plan for a truck trip (one-way or round
trip) given the BVD daily price PDFs. Handles tank capacity, fuel reserve,
date-aware pricing (each stop priced from the PDF effective on the day the
truck arrives), multi-pass stations on round trips (e.g. fill twice at
Cornwall, ON), and an end-of-trip fuel "carry-forward" credit so the LP
overfills at cheap stops when the destination is in expensive-fuel territory.

Usage:
    python optimize.py --from "Mississauga, ON" --to "Kentville, NS"
    python optimize.py --truck 3006 --to "Winnipeg, MB" --one-way
    python optimize.py --truck 3003 --to "Montreal, QC" --carry-forward \\
        --via "Cornwall, ON|Trois Rivieres, QC" --one-way

Defaults (fleet):
    --tank 1000          litres
    --consumption 0.4    L/km (40 L/100km, 2.5 km/L)
    --reserve 200        L hard floor at every stop
    --min-fill-liters 80 don't pull off the highway for a smaller fill
    --avg-speed 80       km/h for the date-of-arrival clock math
    --start-date today; --start-time = truck's last GPS fix or "now"

Live FleetHunt integration (--truck NAME):
    auto-fills --from (live GPS) and --start-fuel (live fuel %)
    on every run; no caching of truck data.

Highway override:
    --via "City, PR|..."  hand-picked waypoint cities
    --highway 11          fetched from OSM (Overpass) once per run, then
                          translated to nearest cities — output prints a
                          ready-to-paste --via for next time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pdfplumber
import requests
from scipy.optimize import linprog

PRICE_DIR = Path.home() / "BVD_PCN"
SCRIPT_DIR = PRICE_DIR / "optimizer"
CACHE_DIR = SCRIPT_DIR / ".cache"
# OSM usage policy asks for a contact; set BVD_CONTACT_EMAIL to include one.
_CONTACT = os.environ.get("BVD_CONTACT_EMAIL")
USER_AGENT = f"BVD-PCN-FuelOpt/1.0 ({_CONTACT})" if _CONTACT else "BVD-PCN-FuelOpt/1.0"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
OSRM = "https://router.project-osrm.org"
OVERPASS = "https://overpass-api.de/api/interpreter"

DEFAULT_TANK = 1000.0
DEFAULT_CONSUMPTION = 0.4
DEFAULT_RESERVE_L = 150.0          # litres, hard floor at every stop
                                   # (50 L above bare reserve gives the driver
                                   # ~125 km of comfort buffer for breaks,
                                   # traffic, weather, or a closed pump)
DEFAULT_DETOUR_KM = 25.0
DEFAULT_MIN_FILL_L = 80.0          # don't pull off the highway for less than this
DEFAULT_AVG_SPEED_KMH = 80.0       # used only for clock-time / per-day PDF lookup
DEFAULT_DEST_RESERVE_L = 100.0     # arrival-fuel floor at destination when it's
                                   # a BVD station (you'll refuel there anyway,
                                   # so the conservative 200 L mid-route reserve
                                   # is overkill). Falls back to --reserve when
                                   # the destination has no co-located BVD.

# Which per-litre price the LP optimizes on (set from --price-basis in main):
#   "economic"  pump price minus GST/HST/QST (recoverable as input tax credits)
#               minus provincial fuel tax (settled by km driven per province
#               under IFTA, regardless of where fuel was bought). This is the
#               true cost difference between stations. Default.
#   "pump"      tax-in price charged to the card (legacy behavior). Use only
#               to reproduce old plans or for cash-flow planning.
PRICE_BASIS = "economic"


def economic_price(pump: float, sales_tax: float, qst: float,
                   pft: float) -> Optional[float]:
    """Tax-adjusted decision price: pump minus recoverable sales taxes
    (GST/HST + QST input tax credits) minus provincial fuel tax (IFTA
    apportions PFT by km driven per jurisdiction, not purchase location).
    What remains — base fuel + freight + FET + carbon + local levies — is
    the cost that genuinely differs by where you buy.
    Returns None if the inputs don't produce a sane price."""
    econ = pump - sales_tax - qst - pft
    return econ if econ > 0.10 else None


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass
class Station:
    site: str
    name: str
    city: str
    prov: str
    price: float             # decision price (economic basis unless --price-basis pump)
    price_cash: float = 0.0  # tax-in pump price actually charged to the card
    network: str = "BVD"     # "BVD" or "Esso"
    coord: Optional[tuple[float, float]] = None
    d_from_origin: Optional[float] = None
    d_to_dest: Optional[float] = None
    detour: Optional[float] = None

    def __post_init__(self):
        if not self.price_cash:
            self.price_cash = self.price

    @property
    def key(self) -> str:
        """Unique cross-network identifier (e.g. 'BVD:58062', 'Esso:523529')."""
        return f"{self.network}:{self.site}"


@dataclass
class Stop:
    label: str
    km: float
    price: float
    price_cash: float = 0.0             # tax-in pump price (display/outlay only)
    site: str = ""
    network: str = "BVD"                # "BVD" or "Esso" — for price lookup
    can_buy: bool = True
    date: Optional[str] = None          # YYYY-MM-DD when truck arrives here
    arrival_time: Optional[str] = None  # HH:MM clock-time at arrival
    price_pdf_date: Optional[str] = None  # actual PDF date used (may differ on fallback)
    min_arrival_l: Optional[float] = None  # per-stop reserve override; None = use global

    def __post_init__(self):
        if not self.price_cash:
            self.price_cash = self.price

    @property
    def price_key(self) -> str:
        """Lookup key matching prices_for_date()'s output."""
        return f"{self.network}:{self.site}" if self.site else ""


# --------------------------------------------------------------------------
# Cache helpers
# --------------------------------------------------------------------------


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.lower()).strip("_")[:120]


def _cache_get(kind: str, key: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{kind}_{_slug(key)}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_set(kind: str, key: str, value) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{kind}_{_slug(key)}.json"
    p.write_text(json.dumps(value))


# --------------------------------------------------------------------------
# Network: Nominatim + OSRM
# --------------------------------------------------------------------------


_last_nominatim = 0.0


def geocode(query: str) -> Optional[tuple[float, float]]:
    """Cached geocode via Nominatim. Rate-limited to 1 req/sec per their TOS."""
    global _last_nominatim
    cached = _cache_get("geo", query)
    if cached is not None:
        return tuple(cached) if cached else None
    elapsed = time.time() - _last_nominatim
    if elapsed < 1.05:
        time.sleep(1.05 - elapsed)
    r = requests.get(
        NOMINATIM,
        params={"q": query, "format": "json", "limit": 1, "countrycodes": "ca"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    _last_nominatim = time.time()
    r.raise_for_status()
    results = r.json()
    if not results:
        _cache_set("geo", query, None)
        return None
    coord = (float(results[0]["lat"]), float(results[0]["lon"]))
    _cache_set("geo", query, list(coord))
    return coord


_PROV_MAP_REV = {
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB",
    "New Brunswick": "NB", "Nouveau-Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS", "Nouvelle-Écosse": "NS",
    "Ontario": "ON", "Prince Edward Island": "PE",
    "Quebec": "QC", "Québec": "QC",
    "Saskatchewan": "SK", "Yukon": "YT",
    "Northwest Territories": "NT", "Nunavut": "NU",
}


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Lat/lon -> 'City, PR' via Nominatim. Cached, rate-limited."""
    global _last_nominatim
    key = f"{lat:.4f},{lon:.4f}"
    cached = _cache_get("rgeo", key)
    if cached is not None:
        return cached if cached else None
    elapsed = time.time() - _last_nominatim
    if elapsed < 1.05:
        time.sleep(1.05 - elapsed)
    r = requests.get(NOMINATIM_REVERSE, params={
        "lat": lat, "lon": lon, "format": "json", "zoom": 10,
        "accept-language": "en",
    }, headers={"User-Agent": USER_AGENT}, timeout=30)
    _last_nominatim = time.time()
    r.raise_for_status()
    data = r.json()
    addr = data.get("address", {})
    city = (addr.get("city") or addr.get("town") or addr.get("village")
            or addr.get("hamlet") or addr.get("municipality")
            or addr.get("county"))
    prov = addr.get("state")
    if not city or not prov:
        _cache_set("rgeo", key, "")
        return None
    label = f"{city}, {_PROV_MAP_REV.get(prov, prov)}"
    _cache_set("rgeo", key, label)
    return label


def osrm_route_dist(a: tuple[float, float], b: tuple[float, float]) -> Optional[float]:
    key = f"{a[0]:.4f},{a[1]:.4f}->{b[0]:.4f},{b[1]:.4f}"
    cached = _cache_get("dist", key)
    if cached is not None:
        return float(cached)
    coord = f"{a[1]},{a[0]};{b[1]},{b[0]}"
    r = requests.get(
        f"{OSRM}/route/v1/driving/{coord}",
        params={"overview": "false"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        return None
    km = data["routes"][0]["distance"] / 1000.0
    _cache_set("dist", key, km)
    return km


def osrm_table(
    src: tuple[float, float], dests: list[tuple[float, float]]
) -> list[Optional[float]]:
    """Bulk distances (km) from src to each of dests. Batched + cached."""
    if not dests:
        return []
    # Key on a digest of ALL dest coords. The old key (count + first + last
    # only) silently returned a stale, row-misaligned table whenever the
    # middle of the station list changed (different PDF day, dropped
    # station, geocode update) — projecting stations onto wildly wrong km.
    coord_blob = ";".join(f"{lat:.4f},{lon:.4f}" for lat, lon in dests)
    digest = hashlib.md5(coord_blob.encode()).hexdigest()[:16]
    key = f"src={src[0]:.4f},{src[1]:.4f};n={len(dests)};h={digest}"
    cached = _cache_get("table", key)
    if cached is not None:
        return [None if x is None else float(x) for x in cached]
    out: list[Optional[float]] = []
    BATCH = 60
    for i in range(0, len(dests), BATCH):
        batch = dests[i : i + BATCH]
        coord = ";".join(f"{lon},{lat}" for lat, lon in [src] + batch)
        try:
            r = requests.get(
                f"{OSRM}/table/v1/driving/{coord}",
                params={"sources": "0", "annotations": "distance"},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok":
                out.extend([None] * len(batch))
                continue
            row = data["distances"][0][1:]
            out.extend(d / 1000.0 if d is not None else None for d in row)
        except Exception as e:
            print(f"  OSRM batch error: {e}", file=sys.stderr)
            out.extend([None] * len(batch))
    _cache_set("table", key, out)
    return out


# --------------------------------------------------------------------------
# PDF parsing
# --------------------------------------------------------------------------


_ROW_RE = re.compile(
    r"^(\d{5})\s+(.+?)\s+([A-Z]{2})\s+"          # site, name+city blob, prov
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"          # cost freight base
    r"\d+(?:\.\d+)?\s+([\d.]+)\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+"  # FET (PFT) PCT Local
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"          # FuelPrice (SalesTax) InTaxPrice
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"  # (QST) Retail (YourPrice) Savings
)
# capture groups: 1 site, 2 name+city, 3 prov, 4 cost, 5 freight, 6 base,
#                 7 PFT, 8 FuelPrice, 9 SalesTax, 10 InTax, 11 QST,
#                 12 Retail, 13 YourPrice, 14 Savings


def _parse_packed_row(text: str) -> Optional[tuple[str, str, str, str, float, float]]:
    """Parse a row that pdfplumber packed into a single cell.

    Returns (site, name, city, prov, econ_price, pump_price) or None.
    """
    text = text.replace("\n", " ").strip()
    m = _ROW_RE.match(text)
    if not m:
        return None
    site = m.group(1)
    name_city_blob = m.group(2).strip()
    prov = m.group(3)
    your_price = float(m.group(13))
    if your_price <= 0:
        return None    # e.g. a station listed with YourPrice 0.0000 — not sellable
    econ = economic_price(your_price, float(m.group(9)),
                          float(m.group(11)), float(m.group(7)))
    if econ is None:
        econ = your_price
    # The blob looks like "ALDERSYDE ALDERSYDE" or "BVD MISSISSAUGA Mississauga"
    # or "DRAYTON VALLEY-50TH ST DRAYTON VALLEY". Split into name+city by
    # finding the longest suffix that matches the prefix word-for-word
    # (city name appears at the end and often duplicates words from name).
    words = name_city_blob.split()
    if len(words) == 1:
        name, city = words[0], words[0]
    else:
        # Heuristic: the city is repeated at the end. Find the largest split
        # point such that words[k:] is a "reasonable" city (1-3 words).
        # Default: half/half.
        best_split = len(words) // 2
        # Try matching: city = last 1, 2, or 3 words; check those words also
        # appear in the prefix (sign of a duplicated city name).
        for nwords in (3, 2, 1):
            if nwords >= len(words):
                continue
            candidate_city = words[-nwords:]
            candidate_name = words[:-nwords]
            # If city words all appear inside name, this split likely correct
            if all(w in candidate_name for w in candidate_city):
                best_split = len(words) - nwords
                break
        name = " ".join(words[:best_split]) or words[0]
        city = " ".join(words[best_split:]) or name
    return site, name, city, prov, econ, your_price


def _clean_city(raw: str) -> str:
    """Tidy up a city string: collapse internal whitespace, dedupe
    consecutive identical words (e.g. 'Nipigon Nipigon' -> 'Nipigon',
    'BRAMPTON Brampton' -> 'Brampton') and Title Case the final result.
    """
    if not raw:
        return raw
    words = raw.replace("\n", " ").split()
    out: list[str] = []
    for w in words:
        if out and out[-1].lower() == w.lower():
            continue
        out.append(w)
    return " ".join(out).title()


def parse_pdf(pdf_path: Path) -> list[Station]:
    """Parse BVD price PDF -> list of Station.

    Station.price      = economic price: YourPrice − SalesTax − QST − PFT
                         (GST/HST/QST are recoverable input tax credits; PFT
                         is settled by km per province under IFTA — neither
                         differs by *where* you buy).
    Station.price_cash = 'Your Price' as billed at the pump (tax-in).

    Full 18-column layout: Site Name City Prov Cost Freight Base FET PFT PCT
    Local FuelPrice SalesTax InTax QST Retail YourPrice Savings.
    pdfplumber sometimes returns proper structured rows and sometimes packs
    the entire row into a single cell as whitespace-separated text. Handle
    both shapes. Stations with YourPrice <= 0 are dropped (a zero-priced row
    would read as free fuel and magnetize the LP).
    """
    stations: list[Station] = []
    seen: set[str] = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for row in table:
                    if not row:
                        continue
                    # Shape 1: structured row with site number in row[0]
                    site0 = (row[0] or "").strip()
                    if site0.isdigit() and len(row) >= 14 and (row[-2] or "").strip():
                        try:
                            price = float((row[-2] or "").strip())
                        except ValueError:
                            continue
                        if site0 in seen or price <= 0:
                            continue
                        # Tax columns by position from the right (only safe on
                        # the full 18-col shape): PFT=-10, SalesTax=-6, QST=-4
                        econ = None
                        if len(row) >= 18:
                            try:
                                econ = economic_price(
                                    price,
                                    float((row[-6] or "0").strip()),
                                    float((row[-4] or "0").strip()),
                                    float((row[-10] or "0").strip()),
                                )
                            except ValueError:
                                econ = None
                        seen.add(site0)
                        stations.append(Station(
                            site=site0,
                            name=(row[1] or "").strip().replace("\n", " "),
                            city=_clean_city(row[2] or ""),
                            prov=(row[3] or "").strip(),
                            price=econ if econ is not None else price,
                            price_cash=price,
                        ))
                        continue
                    # Shape 2: packed row in row[0]
                    if site0 and site0[:5].isdigit() and " " in site0:
                        parsed = _parse_packed_row(site0)
                        if parsed and parsed[0] not in seen:
                            seen.add(parsed[0])
                            stations.append(Station(
                                site=parsed[0],
                                name=parsed[1],
                                city=_clean_city(parsed[2]),
                                prov=parsed[3],
                                price=parsed[4],
                                price_cash=parsed[5],
                            ))
    return stations


def find_pdf_for_date(date_str: str, max_lookback_days: int = 60) -> Optional[Path]:
    """Locate the BVD PDF effective on `date_str`, falling back to the most
    recent earlier date if the exact one isn't available.

    Searches both the flat layout (`~/BVD_PCN/YYYY-MM-DD.pdf`) and the
    organized layout (`~/BVD_PCN/YYYY-MM/YYYY-MM-DD.pdf`). Prefers `_v2`
    over `_v1` over the bare name.
    """
    target = dt.date.fromisoformat(date_str)
    for delta in range(max_lookback_days + 1):
        d = target - dt.timedelta(days=delta)
        ds = d.isoformat()
        ym = d.strftime("%Y-%m")
        for stem in (f"{ds}_v2", ds, f"{ds}_v1"):
            for path in (PRICE_DIR / ym / f"{stem}.pdf",
                         PRICE_DIR / f"{stem}.pdf"):
                if path.exists():
                    return path
    return None


# Cache of {date_str -> ({station_key -> (econ, pump)}, actual_pdf_date_used)}
# station_key is "BVD:<site>" or "Esso:<site>" to disambiguate networks.
_price_cache: dict[str, tuple[dict[str, tuple[float, float]], str]] = {}


def prices_for_date(date_str: str) -> Optional[tuple[dict[str, tuple[float, float]], str]]:
    """Return ({'BVD:site' or 'Esso:site' -> (econ_price, pump_price)},
    actual_pdf_date_used).

    Combines BVD PDF + Esso CSV for the same effective date. Falls back to
    the most recent earlier date if the exact one isn't published yet.
    """
    if date_str in _price_cache:
        return _price_cache[date_str]
    prices: dict[str, tuple[float, float]] = {}
    actual_date = None

    # BVD PDF
    pdf = find_pdf_for_date(date_str)
    if pdf is not None:
        actual_date = re.match(r"(\d{4}-\d{2}-\d{2})", pdf.stem).group(1)
        for s in parse_pdf(pdf):
            prices[f"BVD:{s.site}"] = (s.price, s.price_cash)

    # Esso CSV (effective date = file date + 1)
    try:
        import esso
        csv_path = esso.find_csv_for_effective_date(date_str)
        if csv_path is not None:
            for rec in esso.parse_csv(csv_path):
                prices[f"Esso:{rec['site']}"] = (rec["price"], rec["price_cash"])
            if actual_date is None:
                actual_date = date_str  # use the requested date if no BVD anchor
    except ImportError:
        pass  # esso.py not installed; BVD-only mode

    if not prices:
        return None
    _price_cache[date_str] = (prices, actual_date or date_str)
    return _price_cache[date_str]


def datetime_for_km(start_dt: dt.datetime, km: float, avg_speed_kmh: float,
                    turnaround_km: Optional[float] = None,
                    turnaround_hours: float = 0.0) -> dt.datetime:
    """Wall-clock datetime when the truck has cumulatively driven `km`.

    `turnaround_km` and `turnaround_hours` add an idle period at the
    destination (every km past the turnaround point gets shifted later by
    `turnaround_hours`).
    """
    extra_h = (turnaround_hours
               if (turnaround_km is not None and km > turnaround_km) else 0.0)
    drive_h = km / max(avg_speed_kmh, 1.0)
    return start_dt + dt.timedelta(hours=drive_h + extra_h)


def latest_pdf(date_str: Optional[str]) -> Path:
    """Find a price PDF.

    Searches the top of ~/BVD_PCN/ and any YYYY-MM/ subfolders (created by
    organize.py). For a specific date, prefers `_v2` over the bare name
    over `_v1` (same order as find_pdf_for_date). Without a date, returns
    the lexically latest YYYY-MM-DD-named file (which is also the most
    recent).
    """
    candidates: list[Path] = []
    if date_str:
        for stem in (f"{date_str}_v2", date_str, f"{date_str}_v1"):
            for p in (PRICE_DIR.glob(f"{stem}.pdf"),
                      PRICE_DIR.glob(f"*/{stem}.pdf")):
                candidates.extend(p)
            if candidates:
                return candidates[0]
        raise FileNotFoundError(f"No PDF for {date_str} under {PRICE_DIR}")
    # All date-named PDFs (top + subfolders), sorted by date in filename
    for p in PRICE_DIR.rglob("*.pdf"):
        if re.match(r"^\d{4}-\d{2}-\d{2}(_v\d+)?\.pdf$", p.name):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No date-named PDFs under {PRICE_DIR}")
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


MAX_HOP_KM = 700.0   # max single-leg distance; > this, OSRM may cut via US.
                     # 700 km catches Sault Ste Marie -> Nipigon (~530 km) and
                     # other Lake Superior north-shore gaps. If a greedy step
                     # still finds nothing, build_canadian_chain_from_bvd
                     # auto-expands up to 1100 km before falling back to a
                     # direct (possibly US-routed) final leg.

CANADIAN_PROVINCES = {"AB", "BC", "MB", "NB", "NL", "NS",
                      "ON", "PE", "QC", "SK", "YT", "NT", "NU"}


def _prov_from_label(label: str) -> Optional[str]:
    """Extract a 2-letter province code from a 'City, PR' style label."""
    if not label:
        return None
    m = re.search(r",\s*([A-Z]{2})\b", label.upper())
    return m.group(1) if m and m.group(1) in CANADIAN_PROVINCES else None


def parse_highway_spec(spec: str, default_prov: Optional[str]) -> list[tuple[str, str]]:
    """Parse '--highway 11,ON-17' into [('ON','11'), ('ON','17')]."""
    out: list[tuple[str, str]] = []
    for chunk in spec.split(","):
        c = chunk.strip().upper().replace(" ", "")
        if not c:
            continue
        # Forms: "11", "ON11", "ON-11", "ON_11"
        m = re.match(r"^(?:([A-Z]{2})[-_]?)?(\w+)$", c)
        if not m:
            print(f"Could not parse highway spec {chunk!r}", file=sys.stderr)
            continue
        prov, num = m.group(1), m.group(2)
        if not prov:
            if default_prov is None:
                print(f"Cannot infer province for highway {num!r}; use "
                      f"e.g. 'ON-{num}' to disambiguate", file=sys.stderr)
                continue
            prov = default_prov
        if prov not in CANADIAN_PROVINCES:
            print(f"Unknown Canadian province {prov!r} in highway spec",
                  file=sys.stderr)
            continue
        out.append((prov, num))
    return out


def fetch_highway_waypoints(prov: str, num: str,
                            samples: int = 6) -> list[tuple[float, float]]:
    """Query Overpass for the highway's OSM ways, build a polyline, and sample
    `samples` evenly-spaced lat/lon points.  Cached forever on disk.
    """
    key = f"hwy_{prov}_{num}"
    cached = _cache_get("hwy", key)
    if cached is not None:
        coords = [tuple(c) for c in cached]
        return _sample(coords, samples)

    # Overpass QL — Canadian highways are tagged with just the number
    # (e.g. ref="11", name="Highway 11") with no province prefix.
    # Constrain to the right province by ISO3166-2 area filter.
    ql = (f'[out:json][timeout:60];'
          f'area["ISO3166-2"="CA-{prov}"]->.a;'
          f'way["ref"="{num}"]'
          f'["highway"~"^(motorway|trunk|primary)$"](area.a);'
          f'out geom;')

    print(f"  fetching OSM geometry for highway {prov}-{num} via Overpass...",
          file=sys.stderr)
    try:
        r = requests.post(OVERPASS, data={"data": ql},
                          headers={"User-Agent": USER_AGENT}, timeout=90)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Overpass error for {prov}-{num}: {e}", file=sys.stderr)
        return []

    # Concatenate all way geometries (rough — but adequate for waypoint sampling)
    polyline: list[tuple[float, float]] = []
    for el in data.get("elements", []):
        for pt in el.get("geometry") or []:
            polyline.append((pt["lat"], pt["lon"]))
    if not polyline:
        print(f"  no OSM ways found for highway {prov}-{num}", file=sys.stderr)
        return []
    _cache_set("hwy", key, [list(p) for p in polyline])
    print(f"  cached {len(polyline)} polyline points for {prov}-{num}",
          file=sys.stderr)
    return _sample(polyline, samples)


def _sample(coords: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Pick n evenly-spaced points from a polyline (order preserved)."""
    if not coords or n <= 0:
        return []
    if len(coords) <= n:
        return coords
    # Take indices 1..n (skip first as it's often the highway's edge, less useful)
    out = []
    step = (len(coords) - 1) / (n + 1)
    for i in range(1, n + 1):
        out.append(coords[int(i * step)])
    return out


def order_waypoints_along_route(
    origin_coord: tuple[float, float],
    dest_coord: tuple[float, float],
    waypoints: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort raw highway-sampled points by their distance-from-origin along
    the corridor, so the chain order makes sense for OSRM."""
    if not waypoints:
        return []
    dists = osrm_table(origin_coord, waypoints)
    paired = [(d, p) for d, p in zip(dists, waypoints) if d is not None]
    paired.sort()
    return [p for _, p in paired]


def build_canadian_chain_from_bvd(
    origin_coord: tuple[float, float],
    dest_coord: tuple[float, float],
    stations: list[Station],
    max_hop_km: float = MAX_HOP_KM,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Derive a Canadian-only waypoint chain from the BVD station network.

    Greedy: at each step, pick the BVD station within `max_hop_km` of the
    current node that makes the most progress toward `dest_coord`. Repeat
    until the destination is within `max_hop_km`. By keeping every leg
    short, we avoid OSRM's tendency to shortcut long Canadian pairs via the
    United States.
    """
    located = [st for st in stations if st.coord]
    coords = [st.coord for st in located]

    print(f"  building Canadian chain through BVD network "
          f"(max hop {max_hop_km:.0f} km, "
          f"{len(located)} candidate stations)...", file=sys.stderr)

    d_to_dest = osrm_table(dest_coord, coords)

    chain_coords: list[tuple[float, float]] = [origin_coord]
    chain_labels: list[str] = ["[origin]"]
    current = origin_coord
    used_idx: set[int] = set()

    for step in range(25):
        d_cur_to_dest = osrm_route_dist(current, dest_coord)
        if d_cur_to_dest is None:
            break
        if d_cur_to_dest <= max_hop_km:
            break
        d_cur_to_st = osrm_table(current, coords)
        # Auto-expand the hop radius if the default finds nothing — common
        # on the Lake Superior north shore where BVD stations are 500-700 km
        # apart. Try max_hop_km, then 1.5x, then 2x, then give up.
        best_idx = -1
        best_progress = 0.0
        for hop_limit in (max_hop_km, max_hop_km * 1.5, max_hop_km * 2.0):
            for j in range(len(located)):
                if j in used_idx:
                    continue
                dcs = d_cur_to_st[j]
                dsd = d_to_dest[j]
                if dcs is None or dsd is None:
                    continue
                if dcs > hop_limit or dcs < 1.0:
                    continue
                progress = d_cur_to_dest - dsd
                if progress > best_progress:
                    best_progress = progress
                    best_idx = j
            if best_idx != -1:
                if hop_limit > max_hop_km:
                    print(f"    (expanded hop to {hop_limit:.0f} km to find "
                          f"a viable Canadian waypoint)", file=sys.stderr)
                break
        if best_idx == -1:
            print(f"    no further BVD waypoint found from current even at "
                  f"2x max-hop; chain may detour via US for last leg",
                  file=sys.stderr)
            break
        st = located[best_idx]
        chain_coords.append(st.coord)
        chain_labels.append(f"{st.city.title().strip()} ({st.site})")
        used_idx.add(best_idx)
        current = st.coord
        print(f"    +leg {step+1}: -> {chain_labels[-1]} "
              f"(progress {best_progress:.0f} km, "
              f"hop {d_cur_to_st[best_idx]:.0f} km)", file=sys.stderr)

    chain_coords.append(dest_coord)
    chain_labels.append("[destination]")
    return chain_coords, chain_labels


def find_on_route_stations_chain(
    stations: list[Station],
    chain_coords: list[tuple[float, float]],
    leg_distances: list[float],
    detour_threshold: float,
) -> list[Station]:
    """Project stations onto a waypoint chain.

    For each station, try every leg (wp_i -> wp_{i+1}) and find the leg whose
    `d(wp_i,S) + d(S,wp_{i+1}) - d(wp_i,wp_{i+1})` (the detour) is smallest.
    Position = (cumulative km to wp_i) + d(wp_i, S).
    Stations must already have .coord set.
    """
    located = [st for st in stations if st.coord]
    print(f"  projecting {len(located)} stations onto "
          f"{len(chain_coords)}-waypoint chain...", file=sys.stderr)

    coords = [s.coord for s in located]
    # For each waypoint, distance from waypoint to every station.
    wp_to_st: list[list[Optional[float]]] = []
    for i, wp in enumerate(chain_coords):
        print(f"    leg distances from waypoint {i+1}/{len(chain_coords)}",
              file=sys.stderr)
        wp_to_st.append(osrm_table(wp, coords))

    cum = [0.0]
    for d in leg_distances:
        cum.append(cum[-1] + d)

    on_route: list[Station] = []
    for j, st in enumerate(located):
        best_pos: Optional[float] = None
        best_detour = float("inf")
        for i in range(len(chain_coords) - 1):
            d1 = wp_to_st[i][j]
            d2 = wp_to_st[i + 1][j]
            if d1 is None or d2 is None:
                continue
            detour = d1 + d2 - leg_distances[i]
            if detour < best_detour:
                best_detour = detour
                best_pos = cum[i] + d1
        if best_pos is not None and best_detour <= detour_threshold:
            st.d_from_origin = best_pos
            st.d_to_dest = cum[-1] - best_pos
            st.detour = best_detour
            on_route.append(st)
    return on_route


def build_stops(
    on_route_stations: list[Station],
    origin_label: str,
    dest_label: str,
    direct_distance: float,
    round_trip: bool,
    origin_station: Optional[Station],
    dest_station: Optional[Station],
    dest_reserve_l: Optional[float] = None,
) -> list[Stop]:
    """Build the linear sequence of stops. For round trips, mirror outbound."""
    stops: list[Stop] = []

    o_price = origin_station.price if origin_station else 9.99
    o_price_cash = origin_station.price_cash if origin_station else 9.99
    o_network = origin_station.network if origin_station else "BVD"

    def _name_with_network(st: Station) -> str:
        """'BVD MISSISSAUGA' if already prefixed, else 'BVD: MARKHAM'."""
        if st.name.upper().startswith(st.network.upper()):
            return st.name
        return f"{st.network} {st.name}"

    stops.append(
        Stop(
            label=(f"{origin_label} (start) "
                   f"[{_name_with_network(origin_station)} "
                   f"({origin_station.site}, {origin_station.city})]")
            if origin_station
            else f"{origin_label} (start) [no BVD/Esso]",
            km=0.0,
            price=o_price,
            price_cash=o_price_cash,
            site=origin_station.site if origin_station else "",
            network=o_network,
            can_buy=origin_station is not None,
        )
    )

    out_stations = sorted(on_route_stations, key=lambda s: s.d_from_origin)
    for st in out_stations:
        if st.d_from_origin <= 5 or st.d_from_origin >= direct_distance - 5:
            continue
        stops.append(
            Stop(
                label=f"{_name_with_network(st)} ({st.site}, {st.city}) outbound",
                km=st.d_from_origin,
                price=st.price,
                price_cash=st.price_cash,
                site=st.site,
                network=st.network,
                can_buy=True,
            )
        )

    d_price = dest_station.price if dest_station else 9.99
    d_price_cash = dest_station.price_cash if dest_station else 9.99
    # For ONE-WAY trips where the destination is co-located with a BVD station,
    # let the LP buy there — the truck will refuel before leaving anyway, so
    # it's cheaper to fill at the (often cheap) destination than overbuy
    # mid-route. For ROUND trips the destination is just a turnaround marker;
    # the truck doesn't refuel there.
    dest_buyable = (dest_station is not None) and (not round_trip)
    d_network = dest_station.network if dest_station else "BVD"
    stops.append(
        Stop(
            label=(
                (f"{dest_label} (turnaround) "
                 f"[{_name_with_network(dest_station)} "
                 f"({dest_station.site}, {dest_station.city})]")
                if (round_trip and dest_station)
                else ((f"{dest_label} (arrival) "
                       f"[{_name_with_network(dest_station)} "
                       f"({dest_station.site}, {dest_station.city})]")
                      if dest_station else f"{dest_label} (arrival)")
            ),
            km=direct_distance,
            price=d_price,
            price_cash=d_price_cash,
            site=dest_station.site if dest_station else "",
            network=d_network,
            can_buy=dest_buyable,
            min_arrival_l=dest_reserve_l if dest_buyable else None,
        )
    )

    if round_trip:
        for st in reversed(out_stations):
            if st.d_from_origin <= 5 or st.d_from_origin >= direct_distance - 5:
                continue
            stops.append(
                Stop(
                    label=f"{_name_with_network(st)} ({st.site}, {st.city}) return",
                    km=2 * direct_distance - st.d_from_origin,
                    price=st.price,
                    price_cash=st.price_cash,
                    site=st.site,
                    network=st.network,
                    can_buy=True,
                )
            )
        stops.append(
            Stop(
                label=f"{origin_label} (arrival)",
                km=2 * direct_distance,
                price=o_price,
                price_cash=o_price_cash,
                site=origin_station.site if origin_station else "",
                network=o_network,
                can_buy=False,
            )
        )

    return stops


# --------------------------------------------------------------------------
# LP solver
# --------------------------------------------------------------------------


def solve_fuel_lp(
    stops: list[Stop],
    s0: float,
    tank_cap: float,
    consumption: float,
    reserve: float,
    end_fuel_value: float = 0.0,
) -> dict:
    """
    Decision: f_i >= 0, litres bought at stop i.
    State at arrival of stop i: tank_arr_i = S0 + sum_{j<i} f_j - c * km_i.
    State at departure: tank_dep_i = tank_arr_i + f_i.
    Constraints:
        f_i = 0  if not can_buy
        tank_dep_i <= tank_cap   for i = 0..N-2
        tank_arr_i >= reserve    for i = 1..N-1
    Objective: minimize sum(f_i * price_i) - end_fuel_value * end_fuel
        where end_fuel = S0 + sum(f_i) - c * cum_last (litres still in tank at
        the final stop). The `end_fuel_value` credit captures that ending the
        trip at an expensive-fuel destination means each litre of cheap fuel
        carried forward avoids buying a litre there for the next trip.
        Equivalent to using per-stop objective coefficients (price_i - end_fuel_value).
    """
    N = len(stops)
    if N < 2:
        raise ValueError("Need at least 2 stops")

    cum = np.array([s.km for s in stops])
    prices = np.array([s.price for s in stops])
    can_buy = np.array([s.can_buy for s in stops])
    obj = prices - end_fuel_value

    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    A_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    for i in range(N):
        if not can_buy[i]:
            r = np.zeros(N)
            r[i] = 1.0
            A_eq.append(r)
            b_eq.append(0.0)

    # Capacity constraints at departure (i = 0..N-2)
    for i in range(N - 1):
        r = np.zeros(N)
        r[: i + 1] = 1.0
        A_ub.append(r)
        b_ub.append(tank_cap - s0 + consumption * cum[i])

    # Reserve constraints at arrival (i = 1..N-1).
    # Per-stop override (stop.min_arrival_l) is used when set — e.g. the
    # destination of a one-way trip with a co-located BVD station relaxes
    # the 200 L mid-route reserve to the smaller --dest-reserve, since the
    # truck refuels there anyway.
    for i in range(1, N):
        stop_rsv = stops[i].min_arrival_l if stops[i].min_arrival_l is not None else reserve
        r = np.zeros(N)
        r[:i] = -1.0
        A_ub.append(r)
        b_ub.append(-stop_rsv + s0 - consumption * cum[i])

    bounds = [(0.0, tank_cap) for _ in range(N)]

    res = linprog(
        obj,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return {"feasible": False, "msg": res.message, "stops": stops}

    fills = np.array(res.x, dtype=float)
    fills[fills < 0.05] = 0.0  # numerical noise

    tank_arr, tank_dep = [], []
    fuel = s0
    for i in range(N):
        if i > 0:
            fuel -= consumption * (cum[i] - cum[i - 1])
        tank_arr.append(fuel)
        fuel += fills[i]
        tank_dep.append(fuel)

    end_fuel = float(s0 + fills.sum() - consumption * cum[-1])
    total_cost = float((fills * prices).sum())
    end_fuel_credit = end_fuel * end_fuel_value
    return {
        "feasible": True,
        "fills": fills.tolist(),
        "tank_arr": tank_arr,
        "tank_dep": tank_dep,
        "total_cost": total_cost,
        "end_fuel": end_fuel,
        "end_fuel_value": end_fuel_value,
        "end_fuel_credit": end_fuel_credit,
        "net_cost": total_cost - end_fuel_credit,
        "stops": stops,
    }


def solve_with_min_fill(
    stops: list[Stop],
    s0: float,
    tank_cap: float,
    consumption: float,
    reserve: float,
    min_fill_l: float,
    end_fuel_value: float = 0.0,
) -> dict:
    """Solve the LP, then iteratively block any stop whose optimal fill is
    below `min_fill_l` litres and re-solve. Captures the operational rule
    'don't pull off the highway for a tiny top-up'.

    If blocking makes the LP infeasible (the suppressed stop was actually
    needed to bridge a fuel gap), unblock the most-recently-blocked stop
    and accept the previous feasible solution.
    """
    if min_fill_l <= 0:
        return solve_fuel_lp(stops, s0, tank_cap, consumption, reserve,
                             end_fuel_value=end_fuel_value)

    blocked: list[int] = []      # stop indices blocked for being too small
    last_feasible: Optional[dict] = None

    for iteration in range(30):
        result = solve_fuel_lp(stops, s0, tank_cap, consumption, reserve,
                               end_fuel_value=end_fuel_value)
        if not result["feasible"]:
            # Roll back the last block — that stop was actually needed
            if blocked:
                last = blocked.pop()
                stops[last].can_buy = True
                continue
            return result
        last_feasible = result
        fills = result["fills"]
        # Find the smallest non-zero fill below threshold; block it next round
        candidates = [
            (f, i) for i, f in enumerate(fills)
            if 0.5 < f < min_fill_l and stops[i].can_buy
        ]
        if not candidates:
            result["suppressed_indices"] = list(blocked)
            return result
        candidates.sort()  # smallest first
        _, idx = candidates[0]
        stops[idx].can_buy = False
        blocked.append(idx)

    if last_feasible is not None:
        last_feasible["suppressed_indices"] = list(blocked)
        return last_feasible
    return result


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def print_plan(result: dict, args) -> None:
    if not result["feasible"]:
        print(f"\nINFEASIBLE: {result.get('msg', 'unknown')}")
        return

    stops = result["stops"]
    fills = result["fills"]
    tank_arr = result["tank_arr"]
    tank_dep = result["tank_dep"]

    print()
    print("=" * 110)
    print(f"BVD fueling plan: {args.from_}  ->  {args.to}"
          f"{'  (round trip)' if not args.one_way else '  (one-way)'}")
    print(f"  Tank {args.tank:.0f} L | "
          f"consumption {args.consumption*100:.0f} L/100km "
          f"({1/args.consumption:.2f} km/L) | "
          f"reserve {args.reserve:.0f} L | "
          f"start fuel {args.start_fuel:.0f} L")
    if PRICE_BASIS == "economic":
        print("  Price basis: ECONOMIC — $/L below exclude GST/HST/QST "
              "(recovered as ITCs) and provincial fuel tax (settled by km "
              "via IFTA)")
    else:
        print("  Price basis: PUMP — tax-in prices as charged to the card")
    print("=" * 110)

    fmt = "{:>2} {:<10} {:<5} {:<46} {:>6} {:>6} {:>6} {:>6} {:>9} {:>9}"
    print(fmt.format("#", "Date", "Time", "Stop", "Km", "Arrive", "Buy", "Depart", "$/L", "Cost"))
    print("-" * 128)

    total_buy = 0.0
    for i, (stop, fill, ta, td) in enumerate(
        zip(stops, fills, tank_arr, tank_dep), 1
    ):
        cost = fill * stop.price
        date_label = stop.date or "-"
        if stop.price_pdf_date and stop.date and stop.price_pdf_date != stop.date:
            date_label = f"{stop.date}*"
        print(fmt.format(
            i,
            date_label,
            stop.arrival_time or "-",
            stop.label[:46],
            f"{stop.km:.0f}",
            f"{ta:.0f}",
            f"{fill:.0f}" if fill > 0.5 else "-",
            f"{td:.0f}",
            f"${stop.price:.4f}" if fill > 0.5 else "-",
            f"${cost:.2f}" if fill > 0.5 else "-",
        ))
        total_buy += fill
    print("-" * 128)
    print("  * = price PDF for that date wasn't available, used most recent prior "
          "(see stderr)")
    print(f"  Total fuel purchased: {total_buy:.0f} L")
    if PRICE_BASIS == "economic":
        pump_outlay = sum(f * s.price_cash for s, f in zip(stops, fills))
        print(f"  Economic cost:        ${result['total_cost']:.2f}  "
              f"(true cost after ITC recovery + IFTA settlement)")
        print(f"  Pump outlay:          ${pump_outlay:.2f}  "
              f"(cash charged to card; "
              f"${pump_outlay - result['total_cost']:.2f} flows back via "
              f"ITCs/IFTA)")
    else:
        print(f"  Total cost:           ${result['total_cost']:.2f}")
    end_fuel = result.get("end_fuel", 0.0)
    end_value = result.get("end_fuel_value", 0.0)
    if end_value > 0 and end_fuel > 0:
        credit = result.get("end_fuel_credit", 0.0)
        print(f"  End-fuel banked:      {end_fuel:.0f} L @ ${end_value:.4f}  =  "
              f"−${credit:.2f}  (avoided next-trip fill at this price)")
        print(f"  Net trip cost:        ${result['net_cost']:.2f}")
    print(f"  Cost per km:          ${result['total_cost'] / max(stops[-1].km,1):.4f}")
    print(f"  Blended price:        ${result['total_cost'] / max(total_buy,1):.4f}/L")
    print()

    # Per-station summary (combining outbound + return passes)
    print("Per-station purchase summary (multi-pass collapsed):")
    by_site: dict[str, dict] = {}
    for stop, fill in zip(stops, fills):
        if fill < 0.5:
            continue
        key = stop.site or stop.label
        if key not in by_site:
            by_site[key] = {
                "label": re.sub(r" (outbound|return)$", "", stop.label),
                "price": stop.price,
                "liters": 0.0,
                "passes": 0,
            }
        by_site[key]["liters"] += fill
        by_site[key]["passes"] += 1
    for key, info in sorted(by_site.items(), key=lambda kv: -kv[1]["liters"]):
        cost = info["liters"] * info["price"]
        passes = f"({info['passes']}x)" if info["passes"] > 1 else "    "
        print(
            f"  {info['label']:<55} {passes} "
            f"{info['liters']:>6.0f} L @ ${info['price']:.4f} = ${cost:>9.2f}"
        )
    print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--from", dest="from_",
                        help='Origin city, e.g. "Mississauga, ON". '
                             'Omit if --truck is given — origin is set from '
                             'the truck\'s live FleetHunt position.')
    parser.add_argument("--to", required=True,
                        help='Destination city, e.g. "Kentville, NS"')
    parser.add_argument("--start-fuel", type=float, default=None,
                        help="Starting fuel litres. If --truck is given, "
                             "defaults to fuel_percentage * tank / 100. "
                             "Otherwise defaults to 0.")
    parser.add_argument("--truck",
                        help="FleetHunt truck name, id, VIN, or plate. "
                             "Auto-fills --from from live GPS and "
                             "--start-fuel from the truck's reported "
                             "fuel_percentage. Both can still be overridden.")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip auto-registering this trip in trips.json. "
                             "Use for one-off what-if queries where you don't "
                             "want the watcher to start scoring refuel events.")
    parser.add_argument("--tank", type=float, default=None,
                        help=f"Tank size in litres. Default: per-truck value "
                             f"from trucks.json (e.g. 3007=900, others=1000) "
                             f"when --truck is given, else {DEFAULT_TANK:.0f}.")
    parser.add_argument("--consumption", type=float, default=DEFAULT_CONSUMPTION,
                        help="Litres per km (default 0.4 = 40 L/100km)")
    parser.add_argument("--reserve", type=float, default=DEFAULT_RESERVE_L,
                        help=f"Minimum fuel reserve in litres at every stop "
                             f"(default {DEFAULT_RESERVE_L:.0f} L)")
    parser.add_argument("--dest-reserve", type=float, default=DEFAULT_DEST_RESERVE_L,
                        help=f"Arrival fuel floor at destination when it's a "
                             f"BVD station (default {DEFAULT_DEST_RESERVE_L:.0f} L; "
                             f"reasoning: the truck refuels there immediately, "
                             f"so the 200 L mid-route reserve is overkill). "
                             f"Set equal to --reserve to disable the relaxation.")
    parser.add_argument("--date", default=None,
                        help="Pin a single date for ALL stops (overrides date-aware "
                             "pricing). Default: use --start-date and price each "
                             "stop by the day the truck reaches it.")
    parser.add_argument("--start-date", default=dt.date.today().isoformat(),
                        help="Trip departure date YYYY-MM-DD (default: today). "
                             "Each stop's price is taken from the PDF effective "
                             "on the day the truck arrives there.")
    parser.add_argument("--avg-speed", type=float, default=DEFAULT_AVG_SPEED_KMH,
                        help=f"Average driving speed in km/h (default "
                             f"{DEFAULT_AVG_SPEED_KMH:.0f}). Used to convert km "
                             f"into clock time so each stop is priced from the "
                             f"PDF effective on the day the truck actually arrives.")
    parser.add_argument("--start-time", default=None,
                        help="Trip start clock time HH:MM (24h). Default: the "
                             "truck's last GPS-fix time if --truck is given, "
                             "else the current local time when start-date is "
                             "today, else 00:00.")
    parser.add_argument("--turnaround-hours", type=float, default=0.0,
                        help="Hours to wait at destination before returning "
                             "(default 0; only matters for round-trip)")
    parser.add_argument("--one-way", action="store_true")
    parser.add_argument("--detour-km", type=float, default=DEFAULT_DETOUR_KM)
    parser.add_argument("--min-fill-liters", type=float, default=DEFAULT_MIN_FILL_L,
                        help=f"Suppress any stop where the optimal fill would "
                             f"be less than this (default {DEFAULT_MIN_FILL_L:.0f} L). "
                             f"Reflects the rule 'don't pull off the highway "
                             f"for a tiny top-up'. Set 0 to disable.")
    parser.add_argument("--end-fuel-value", type=float, default=None,
                        help="Per-litre value of fuel still in the tank at "
                             "destination. Use this when the next trip will "
                             "start from an expensive-fuel area (e.g. Quebec): "
                             "set to the destination station's price so the LP "
                             "overfills at the cheap origin-side stops. "
                             "If omitted and --carry-forward is set, defaults "
                             "to the destination BVD price.")
    parser.add_argument("--carry-forward", action="store_true",
                        help="Shortcut: auto-set --end-fuel-value to the "
                             "destination BVD station's price, so the truck "
                             "carries max cheap fuel forward for the next trip.")
    parser.add_argument("--carry-min-spread", type=float, default=0.02,
                        help="Haircut ($/L, default 0.02) applied to the "
                             "carry-forward credit: leftover litres are "
                             "valued at destination price MINUS this margin, "
                             "so the LP only overbuys at stops that beat the "
                             "destination by more than the margin. Covers "
                             "the cost of hauling extra fuel weight, "
                             "day-to-day price noise, and the risk the next "
                             "trip doesn't start where planned. Ignored when "
                             "--end-fuel-value is given explicitly. Also "
                             "gates carry-forward auto-enable. Set 0 for the "
                             "old face-value behavior.")
    parser.add_argument("--price-basis", choices=("economic", "pump"),
                        default="economic",
                        help="Price the LP optimizes on. 'economic' (default) "
                             "= pump price minus GST/HST/QST (recovered as "
                             "input tax credits) minus provincial fuel tax "
                             "(settled by km driven per province under IFTA, "
                             "not by purchase location) — the true cost "
                             "difference between stations. 'pump' = tax-in "
                             "card price (pre-2026-07 legacy behavior).")
    parser.add_argument("--via", default=None,
                        help='Pipe-separated waypoint cities to force a Canadian route, '
                             'e.g. "Cornwall, ON|Quebec City, QC|Moncton, NB". '
                             'Auto-set for known cross-Canada corridors.')
    parser.add_argument("--highway", default=None,
                        help='Comma-separated highway specs, e.g. "11" or "ON-11" '
                             'or "ON-11,ON-17". Province inferred from origin/dest '
                             'when omitted; otherwise prefix as "PR-num". '
                             'Geometry is pulled from OpenStreetMap once per '
                             'highway and cached locally. Sampled waypoints get '
                             'injected before any manual --via cities.')
    args = parser.parse_args()

    global PRICE_BASIS
    PRICE_BASIS = args.price_basis
    print(f"  price basis: {PRICE_BASIS}"
          + ("  (ex-GST/HST/QST + ex-PFT; pump outlay shown separately)"
             if PRICE_BASIS == "economic" else "  (tax-in card prices)"),
          file=sys.stderr)

    # FleetHunt --truck integration: ALWAYS fetches fresh from the FleetHunt
    # API on every run (no caching). Auto-fills --from and --start-fuel from
    # the truck's live position and reported fuel_percentage.
    if args.truck:
        import fleet as fleet_mod  # local module
        import truck_config as tc_mod  # noqa
        fetch_time = dt.datetime.now()
        # Resolve tank: explicit --tank overrides; else trucks.json by name
        if args.tank is None:
            args.tank = tc_mod.tank_l(args.truck)
        ctx = fleet_mod.get_truck_context(args.truck, tank_capacity_l=args.tank)
        d = ctx["device"]
        # Compute staleness of the truck's last GPS fix
        last_fix_str = d.get("dt_tracker") or ""
        stale_label = ""
        try:
            last_fix_dt = dt.datetime.strptime(last_fix_str, "%Y-%m-%d %H:%M:%S")
            age_minutes = (fetch_time - last_fix_dt).total_seconds() / 60
            if age_minutes < 5:
                stale_label = f"  ({age_minutes:.0f}m old — fresh)"
            elif age_minutes < 60:
                stale_label = f"  ({age_minutes:.0f}m old)"
            elif age_minutes < 24 * 60:
                stale_label = f"  ({age_minutes/60:.1f}h old — STALE)"
            else:
                stale_label = f"  ({age_minutes/60:.0f}h old — VERY STALE)"
        except ValueError:
            pass

        print(f"FleetHunt LIVE fetch at "
              f"{fetch_time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(no cache, /api/devices)", file=sys.stderr)
        print(f"  truck {d.get('name')!r}  "
              f"(id {d.get('id')}, VIN {d.get('vin')}, "
              f"plate {d.get('license_plate_no', '-')})",
              file=sys.stderr)
        print(f"  position: {ctx['from_city']}  "
              f"({d['latitude']:.4f}, {d['longitude']:.4f})",
              file=sys.stderr)
        print(f"  state:    {d.get('state')}, "
              f"speed {d.get('speed')} {d.get('unit_of_speed', 'km/h').lower()}, "
              f"ignition {'ON' if d.get('ignition') else 'OFF'}",
              file=sys.stderr)
        print(f"  last fix: {last_fix_str}{stale_label}", file=sys.stderr)
        if ctx["fuel_pct"] is not None:
            print(f"  fuel:     {ctx['fuel_pct']:.1f}%  ->  "
                  f"{ctx['start_fuel_l']:.0f} L (tank {args.tank:.0f} L)",
                  file=sys.stderr)
        else:
            print(f"  fuel:     (not reported by sensor)", file=sys.stderr)

        if not args.from_:
            args.from_ = ctx["from_city"]
        else:
            print(f"  (--from manually set to {args.from_!r}, "
                  f"overriding live position)", file=sys.stderr)
        if args.start_fuel is None and ctx["start_fuel_l"] is not None:
            args.start_fuel = ctx["start_fuel_l"]

    if not args.from_:
        parser.error("--from is required (or pass --truck to derive it from GPS)")
    if args.start_fuel is None:
        args.start_fuel = 0.0
    if args.tank is None:
        args.tank = DEFAULT_TANK

    start_date = dt.date.fromisoformat(args.start_date)

    # Resolve trip start datetime.
    #   priority: explicit --start-time
    #          -> truck's last GPS fix (if --truck given and same date)
    #          -> "now" if start-date == today
    #          -> 00:00 of start-date
    start_time: Optional[dt.time] = None
    if args.start_time:
        try:
            h, m = args.start_time.split(":")
            start_time = dt.time(int(h), int(m))
        except (ValueError, IndexError):
            print(f"Could not parse --start-time {args.start_time!r}; expected HH:MM",
                  file=sys.stderr)
            return 2
    elif args.truck:
        # Try to use the truck's last fix as the trip start (only if same date)
        try:
            truck_dt = dt.datetime.strptime(
                ctx["device"].get("dt_tracker", ""),
                "%Y-%m-%d %H:%M:%S",
            )
            if truck_dt.date() == start_date:
                start_time = truck_dt.time()
        except (ValueError, KeyError, NameError):
            pass
    if start_time is None:
        if start_date == dt.date.today():
            start_time = dt.datetime.now().time().replace(microsecond=0)
        else:
            start_time = dt.time(0, 0)
    start_datetime = dt.datetime.combine(start_date, start_time)
    print(f"  trip start: {start_datetime.isoformat(' ')} "
          f"(avg speed {args.avg_speed:.0f} km/h)", file=sys.stderr)

    # The "reference" PDF (start-date or --date override) gives us the station
    # network we use for chain-building, geocoding, and on-route filtering.
    # Per-stop prices are recomputed by date later (unless --date is set).
    if args.date:
        ref_pdf = latest_pdf(args.date)
    else:
        ref_pdf = find_pdf_for_date(args.start_date)
        if ref_pdf is None:
            ref_pdf = latest_pdf(None)
    print(f"Reference PDF: {ref_pdf.name}", file=sys.stderr)
    stations = parse_pdf(ref_pdf)
    bvd_count = len(stations)
    # Esso CSV for the same effective date — adds another fueling network
    try:
        import esso
        esso_csv = esso.find_csv_for_effective_date(args.start_date)
        if esso_csv is not None:
            for rec in esso.parse_csv(esso_csv):
                stations.append(Station(
                    site=rec["site"], name=rec["name"],
                    city=_clean_city(rec["city"]),
                    prov=rec["prov"], price=rec["price"],
                    price_cash=rec.get("price_cash", rec["price"]),
                    network="Esso",
                ))
            print(f"  +Esso CSV: {esso_csv.name} ({len(stations) - bvd_count} stations)",
                  file=sys.stderr)
        else:
            print(f"  (no Esso CSV found for effective {args.start_date})",
                  file=sys.stderr)
    except ImportError:
        pass
    print(f"  total stations: {len(stations)}  "
          f"(BVD {bvd_count}, Esso {len(stations) - bvd_count})",
          file=sys.stderr)

    # parse_pdf/esso.parse_csv put the ECONOMIC price in .price; on pump
    # basis, swap the tax-in card price back in as the decision price.
    if PRICE_BASIS == "pump":
        for st in stations:
            st.price = st.price_cash

    o_coord = geocode(args.from_)
    d_coord = geocode(args.to)
    if not o_coord or not d_coord:
        print("Could not geocode origin or destination", file=sys.stderr)
        return 2

    # All BVD stations need coords up front for the chain builder + on-route
    # filter. Geocoding is rate-limited & cached.
    print(f"Geocoding {len(stations)} station cities (cached)...", file=sys.stderr)
    for st in stations:
        st.coord = geocode(f"{st.city}, {st.prov}, Canada")
    located_count = sum(1 for s in stations if s.coord)
    print(f"  {located_count}/{len(stations)} stations located",
          file=sys.stderr)

    # Build the waypoint chain.
    #
    #   --highway HWY[,HWY,...]  injects geometry-sampled waypoints from
    #                            OSM (forces the route through that highway)
    #   --via "City1, PR|City2"  hand-picked waypoint cities
    #   neither                  auto-built from the BVD network so the
    #                            chain stays Canadian
    #
    # When both --highway and --via are supplied, highway points come first
    # (sorted by distance from origin), then manual --via after.
    highway_via_cities: list[str] = []   # 'City, PR' resolved from highway sampling
    if args.highway:
        # Infer default province from origin/destination if they agree
        o_prov = _prov_from_label(args.from_)
        d_prov = _prov_from_label(args.to)
        default_prov = o_prov if o_prov == d_prov else None
        specs = parse_highway_spec(args.highway, default_prov)
        raw_pts: list[tuple[float, float]] = []
        for prov, num in specs:
            raw_pts.extend(fetch_highway_waypoints(prov, num))
        # Sort along the corridor, then reverse-geocode each to "City, PR"
        if raw_pts:
            ordered = order_waypoints_along_route(o_coord, d_coord, raw_pts)
            print(f"  --highway {args.highway}: resolving {len(ordered)} "
                  f"sampled point(s) to nearest city...", file=sys.stderr)
            seen: set[str] = set()
            for lat, lon in ordered:
                label = reverse_geocode(lat, lon)
                if label and label not in seen:
                    seen.add(label)
                    highway_via_cities.append(label)
            print(f"  suggested --via (paste this for next run): "
                  f'"{"|".join(highway_via_cities)}"', file=sys.stderr)

    # Manual --via cities (added after highway-derived ones)
    manual_via_cities: list[str] = []
    if args.via:
        manual_via_cities = [x.strip() for x in args.via.split("|") if x.strip()]

    all_via = highway_via_cities + manual_via_cities

    if all_via:
        chain_labels = [args.from_] + all_via + [args.to]
        chain_coords = []
        for c in chain_labels:
            coord = geocode(c)
            if not coord:
                print(f"Failed to geocode waypoint: {c}", file=sys.stderr)
                return 2
            chain_coords.append(coord)
    else:
        chain_coords, chain_labels = build_canadian_chain_from_bvd(
            o_coord, d_coord, stations
        )

    leg_distances: list[float] = []
    for i in range(len(chain_coords) - 1):
        d = osrm_route_dist(chain_coords[i], chain_coords[i + 1])
        if d is None:
            print(f"OSRM failed for leg {i}", file=sys.stderr)
            return 2
        leg_distances.append(d)

    direct_distance = sum(leg_distances)
    print(f"  chain route: {direct_distance:.0f} km "
          f"({len(chain_coords)-1} legs)", file=sys.stderr)

    on_route = find_on_route_stations_chain(
        stations, chain_coords, leg_distances, args.detour_km
    )
    print(f"  {len(on_route)} stations on-route (detour <= {args.detour_km} km)",
          file=sys.stderr)

    # Pick origin/destination BVD stations. Strategy: prefer city-name match
    # (case-insensitive substring) and lowest price; fall back to nearest.
    def _pick_endpoint_station(
        target_city: str, position: float, prefer_far: bool = False
    ) -> Optional[Station]:
        target = target_city.split(",")[0].strip().lower()
        # First: any station whose city contains target name AND lies near position
        candidates = [
            st for st in on_route
            if target in (st.city or "").lower()
            and abs((st.d_from_origin or 0) - position) < 50
        ]
        if candidates:
            return min(candidates, key=lambda s: s.price)
        # Fallback: closest station within 30 km of position
        candidates = [
            st for st in on_route
            if abs((st.d_from_origin or 0) - position) < 30
        ]
        if candidates:
            return min(candidates, key=lambda s: s.price)
        return None

    origin_station = _pick_endpoint_station(args.from_, 0.0)
    dest_station = _pick_endpoint_station(args.to, direct_distance)

    # Drop stations within ENDPOINT_BUFFER km of origin/destination. Those
    # are effectively the depot — letting them appear as separate stops causes
    # the LP to "visit" them, which burns real detour fuel for no price benefit
    # since the actual depot fill is the same chain anyway.
    ENDPOINT_BUFFER = 50.0
    direct_distance_total = direct_distance
    middle = [
        st for st in on_route
        if st is not origin_station and st is not dest_station
        and st.d_from_origin is not None
        and st.d_from_origin > ENDPOINT_BUFFER
        and st.d_to_dest is not None
        and st.d_to_dest > ENDPOINT_BUFFER
    ]

    # Cluster nearby on-route stations: within CLUSTER_KM of each other, keep
    # only the cheapest. (Two BVD sites in the same city, or two adjacent
    # towns on the same highway, give no fueling-strategy choice — they're
    # one stop in practice.)
    CLUSTER_KM = 30.0
    middle.sort(key=lambda s: s.d_from_origin or 0.0)
    pruned: list[Station] = []
    for st in middle:
        if pruned and abs((st.d_from_origin or 0.0) -
                          (pruned[-1].d_from_origin or 0.0)) < CLUSTER_KM:
            if st.price < pruned[-1].price:
                pruned[-1] = st
            continue
        pruned.append(st)

    if origin_station:
        print(f"  origin station:  {origin_station.name} ({origin_station.site}) "
              f"@ ${origin_station.price}", file=sys.stderr)
    if dest_station:
        print(f"  dest station:    {dest_station.name} ({dest_station.site}) "
              f"@ ${dest_station.price}", file=sys.stderr)

    stops = build_stops(
        pruned, args.from_, args.to, direct_distance,
        not args.one_way, origin_station, dest_station,
        dest_reserve_l=args.dest_reserve,
    )
    print(f"  built {len(stops)} stops in trip sequence", file=sys.stderr)

    # Date-aware repricing: replace each stop's price with the value from the
    # PDF effective on the day the truck reaches it. Use --date as a
    # single-day override; otherwise compute real clock-time per stop using
    # avg_speed, then look up the PDF for that arrival date.
    if not args.date:
        turnaround_km = direct_distance if not args.one_way else None
        dates_used: dict[str, str] = {}
        missing: list[str] = []
        for stop in stops:
            arrival_dt = datetime_for_km(
                start_datetime, stop.km, args.avg_speed,
                turnaround_km=turnaround_km,
                turnaround_hours=args.turnaround_hours,
            )
            arrival_date = arrival_dt.date().isoformat()
            stop.date = arrival_date
            stop.arrival_time = arrival_dt.strftime("%H:%M")
            if not stop.site:
                continue
            result = prices_for_date(arrival_date)
            if result is None:
                missing.append(arrival_date)
                continue
            prices_dict, actual_pdf_date = result
            stop.price_pdf_date = actual_pdf_date
            dates_used[arrival_date] = actual_pdf_date
            # Try network-prefixed key first (e.g. "BVD:58062" or "Esso:523529");
            # fall back to bare site for back-compat with stops built without
            # the network field set.
            pair = prices_dict.get(stop.price_key) or prices_dict.get(stop.site)
            if pair is not None:
                econ, pump = pair
                stop.price = econ if PRICE_BASIS == "economic" else pump
                stop.price_cash = pump
        if dates_used:
            print(f"  date-aware pricing across {len(dates_used)} day(s):",
                  file=sys.stderr)
            for d in sorted(dates_used):
                tag = "" if dates_used[d] == d else f"  (fallback)"
                print(f"    {d}  ->  PDF {dates_used[d]}{tag}", file=sys.stderr)
        if missing:
            print(f"  WARNING: no PDF available for: {', '.join(sorted(set(missing)))} "
                  f"(used reference price)", file=sys.stderr)
    else:
        for stop in stops:
            stop.date = args.date
            stop.price_pdf_date = args.date
            stop.arrival_time = None

    # Determine end-fuel-value for multi-trip awareness.
    # Use the date-aware price at the final stop (stops[-1] was already
    # repriced to the PDF effective on the arrival date), not the reference
    # PDF price — otherwise the credit lags real prices when the trip crosses
    # a day boundary and prices have shifted.
    end_fuel_value = args.end_fuel_value
    # Auto-enable carry-forward when the destination station is genuinely
    # cheaper than the origin — that's the case where the LP should defer
    # buying to the (cheap) destination instead of stocking up at the
    # (expensive) origin. User can still override with explicit flags.
    carry_fwd = args.carry_forward
    auto_reason = ""
    if (end_fuel_value is None and not carry_fwd
            and dest_station and origin_station
            and dest_station.price < origin_station.price - args.carry_min_spread):
        carry_fwd = True
        auto_reason = (f"  (auto-enabled: dest ${dest_station.price:.4f} "
                       f"< origin ${origin_station.price:.4f} by more than "
                       f"the ${args.carry_min_spread:.2f} margin)")
    if end_fuel_value is None:
        if carry_fwd and dest_station and stops:
            # Value leftover litres BELOW the destination price by the
            # carry margin: hauling fuel isn't free (weight burn, price
            # noise, dispatch risk), so only overbuy at stops that beat
            # the destination by more than the margin.
            end_fuel_value = max(stops[-1].price - args.carry_min_spread, 0.0)
        else:
            end_fuel_value = 0.0
    if end_fuel_value > 0:
        margin_note = ""
        if args.end_fuel_value is None and stops:
            margin_note = (f"  (dest ${stops[-1].price:.4f} − "
                           f"${args.carry_min_spread:.2f} carry margin)")
        print(f"  end-fuel credit: ${end_fuel_value:.4f}/L"
              f"{margin_note}{auto_reason}", file=sys.stderr)

    result = solve_with_min_fill(
        stops, args.start_fuel, args.tank, args.consumption, args.reserve,
        args.min_fill_liters, end_fuel_value=end_fuel_value,
    )
    # Report which stops the threshold suppressed (helpful for transparency).
    # On round trips the same site has separate Stop objects per pass — if
    # one was suppressed but the other passes still bought fuel, say so
    # explicitly so the message doesn't contradict the plan table.
    suppressed_idx = result.get("suppressed_indices", [])
    if suppressed_idx:
        # Group by site so we don't double-report
        fills = result.get("fills", [])
        groups: dict[str, dict] = {}
        for i in suppressed_idx:
            s = stops[i]
            key = s.site or s.label
            groups.setdefault(key, {"label": s.label, "kms": [], "other_buys": 0.0})
            groups[key]["kms"].append(s.km)
        # For each site, check if any *other* pass at the same site bought fuel
        for key, info in groups.items():
            for j, s in enumerate(stops):
                if (s.site or s.label) == key and j not in suppressed_idx:
                    info["other_buys"] += fills[j] if j < len(fills) else 0.0
        print(f"  suppressed {len(suppressed_idx)} stop(s) below "
              f"{args.min_fill_liters:.0f} L threshold:", file=sys.stderr)
        for key, info in groups.items():
            kms = "/".join(f"{k:.0f}" for k in info["kms"])
            label = re.sub(r" (outbound|return)$", "", info["label"])
            note = ""
            if info["other_buys"] > 0.5:
                note = (f"  (note: another pass at this site still buys "
                        f"{info['other_buys']:.0f} L)")
            print(f"    {label}  (km {kms}){note}", file=sys.stderr)
    print_plan(result, args)

    # Auto-register the trip so the fleet watcher can score real-time refuel
    # events against this plan. Skipped if --truck wasn't given (no real
    # truck to attach the trip to) or --no-register was passed.
    if args.truck and not getattr(args, "no_register", False):
        try:
            import trips as trips_mod
            extra_args = []
            if args.one_way: extra_args.append("--one-way")
            if args.carry_forward: extra_args.append("--carry-forward")
            if args.via: extra_args.extend(["--via", args.via])
            if args.highway: extra_args.extend(["--highway", args.highway])
            trips_mod.upsert_trip(str(args.truck), {
                "from": args.from_,
                "to": args.to,
                "start_date": args.start_date,
                "start_fuel": float(args.start_fuel),
                "tank": float(args.tank),
                "consumption": float(args.consumption),
                "reserve": float(args.reserve),
                "extra_args": extra_args,
                "last_plan_cost": float(result.get("total_cost", 0.0)),
                "last_plan_fills": [
                    {"site": s.site, "label": s.label, "km": s.km,
                     "litres": f, "price": s.price, "date": s.date,
                     "time": s.arrival_time}
                    for s, f in zip(stops, result.get("fills", []))
                    if f > 0.5
                ],
            })
            print(f"  registered trip in {trips_mod.TRIPS_FILE.name} "
                  f"(use trips.py end {args.truck!r} when complete)",
                  file=sys.stderr)
        except Exception as e:
            print(f"  WARN: trip auto-register failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
