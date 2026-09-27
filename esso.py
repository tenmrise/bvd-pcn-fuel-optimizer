"""
Esso daily price CSV parser.

File format: RP1DEN-{accountnum}-YYYYMMDD.csv
Note: the filename date is the *publication* date — prices in the file are
effective the NEXT day. (May 20 file is for May 21 prices.) This mirrors
how BVD publishes a day before the effective date.

Columns of interest:
  SITE NUMBER, NAME, LOCATION, PROV., PRODUCT,
  NET PRICE, FET, PFT, PCT, LOCAL, PRICE/LTR,
  GST/HST/FNT, PST/QST, TOTAL PRICE, PRICE CHANGE

Two diesel products per site:
  - DSL EFF D    (effective duty)
  - DSL EFF LS   (low-sulfur)  <-- this is what the fleet uses

Prices in the CSV are in CENTS per litre (e.g. 197.40 = $1.9740/L).
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Optional

ESSO_DIR = Path.home() / "BVD_PCN" / "esso"
DEFAULT_PRODUCTS = ("DIESEL LS", "DSL EFF LS")  # both low-sulfur diesel grades
DEFAULT_PRICE_COL = "TOTAL PRICE"


def find_csv_for_effective_date(
    effective_date: str,
    max_lookback_days: int = 30,
) -> Optional[Path]:
    """Find the Esso CSV whose prices are effective on `effective_date`.

    The convention: file dated YYYYMMDD contains prices effective the
    NEXT calendar day. So for effective 2026-05-21 we look for
    RP1DEN-*-20260520.csv. If exact day missing, fall back to most
    recent earlier file (cap at max_lookback_days).
    """
    target = dt.date.fromisoformat(effective_date) - dt.timedelta(days=1)
    for delta in range(max_lookback_days + 1):
        d = target - dt.timedelta(days=delta)
        ymd = d.strftime("%Y%m%d")
        matches = sorted(ESSO_DIR.glob(f"RP1DEN-*-{ymd}.csv"))
        if matches:
            return matches[-1]
    return None


def parse_csv(
    csv_path: Path,
    products: tuple[str, ...] = DEFAULT_PRODUCTS,
    price_col: str = DEFAULT_PRICE_COL,
) -> list[dict]:
    """Parse an Esso price CSV. Returns list of dicts with keys:
    site, name, city, prov, price, price_cash (both $/L), network='Esso',
    product.

    price      = ECONOMIC price: TOTAL − GST/HST/FNT − PST/QST − PFT.
                 GST/HST and QST come back as input tax credits and PFT is
                 settled by km per province under IFTA, so neither should
                 influence *where* to buy. Matches optimize.parse_pdf's basis
                 so BVD and Esso stations compare fairly.
    price_cash = tax-in TOTAL PRICE as billed to the card.

    Includes any row whose PRODUCT is in `products` (default: both DIESEL LS
    and DSL EFF LS — the two low-sulfur diesel grades). When a single site
    has multiple matching products, keeps the CHEAPER one (by economic price).

    Prices converted from cents to dollars per litre.
    """
    best_per_site: dict[str, dict] = {}
    with csv_path.open(newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        header = [h.strip() for h in header]
        try:
            idx_site = header.index("SITE NUMBER")
            idx_name = header.index("NAME")
            idx_city = header.index("LOCATION")
            idx_prov = header.index("PROV.")
            idx_prod = header.index("PRODUCT")
            idx_price = header.index(price_col)
            idx_gst = header.index("GST/HST/FNT")
            idx_pst = header.index("PST/QST")
            idx_pft = header.index("PFT")
        except ValueError as e:
            raise RuntimeError(f"Unexpected Esso CSV header in {csv_path.name}: {e}")

        def _f(row, idx) -> float:
            try:
                return float((row[idx] or "").strip())
            except ValueError:
                return 0.0

        for row in rdr:
            if len(row) < len(header):
                continue
            prod = (row[idx_prod] or "").strip()
            if prod not in products:
                continue
            try:
                price_cents = float((row[idx_price] or "").strip())
            except ValueError:
                continue
            if price_cents <= 0:
                continue
            econ_cents = (price_cents - _f(row, idx_gst) - _f(row, idx_pst)
                          - _f(row, idx_pft))
            if econ_cents <= 10.0:      # tax math went sideways; don't trust row
                econ_cents = price_cents
            site = (row[idx_site] or "").strip()
            rec = {
                "site": site,
                "name": (row[idx_name] or "").strip(),
                "city": (row[idx_city] or "").strip(),
                "prov": (row[idx_prov] or "").strip(),
                "price": round(econ_cents / 100.0, 4),        # economic $/L
                "price_cash": round(price_cents / 100.0, 4),  # tax-in $/L
                "network": "Esso",
                "product": prod,
            }
            # Keep the cheaper of the two products if both reported at a site
            existing = best_per_site.get(site)
            if existing is None or rec["price"] < existing["price"]:
                best_per_site[site] = rec
    return list(best_per_site.values())


def main() -> int:
    """CLI: inspect today's effective Esso prices."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="Effective date YYYY-MM-DD (default: today)")
    ap.add_argument("--product", default=None,
                    help="Restrict to one product (default: both DIESEL LS and DSL EFF LS)")
    ap.add_argument("--cheapest", type=int, default=10,
                    help="Show top N cheapest stations (default 10)")
    args = ap.parse_args()

    csv_path = find_csv_for_effective_date(args.date)
    if not csv_path:
        print(f"No Esso CSV found for effective {args.date}")
        return 1
    print(f"Using {csv_path.name} (effective {args.date}, product {args.product})")
    products = (args.product,) if args.product else DEFAULT_PRODUCTS
    stations = parse_csv(csv_path, products=products)
    print(f"{len(stations)} stations\n")
    stations.sort(key=lambda s: s["price"])
    fmt = "{:<7} {:<40} {:<20} {:<3} {:>9} {:>9}"
    print(fmt.format("SITE", "NAME", "LOCATION", "PR", "ECON$/L", "PUMP$/L"))
    print("-" * 95)
    for s in stations[:args.cheapest]:
        print(fmt.format(s["site"], s["name"][:40], s["city"][:20], s["prov"],
                         f"${s['price']:.4f}", f"${s['price_cash']:.4f}"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
