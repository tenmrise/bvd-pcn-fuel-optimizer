#!/usr/bin/env python3
"""
Organize BVD price PDFs in ~/BVD_PCN/.

Default behavior:
  - Find every `pcn-*.pdf` directly inside ~/BVD_PCN/ (the names BVD's portal
    uses, e.g. `pcn-cad-8628227-7555.pdf`).
  - Open each, read the "Effective Date" from page 1.
  - Rename to YYYY-MM-DD.pdf and move into ~/BVD_PCN/YYYY-MM/.
  - If a file already exists at the destination, overwrite it (newer wins).
  - If multiple PCN files share the same effective date, the one with the
    latest mtime wins (later mtimes are processed last and overwrite).

Optional with --reorganize-existing:
  - Also move any already-date-named PDFs (e.g. 2026-05-04.pdf, 2026-05-04_v2.pdf)
    sitting at the top of ~/BVD_PCN/ into their YYYY-MM/ subfolder. The
    `_v1`/`_v2` suffix is preserved if present, so nothing is overwritten
    accidentally.

Usage:
    python organize.py                       # process new PCN files only
    python organize.py --reorganize-existing # plus move legacy date-named files
    python organize.py --dry-run             # preview, no changes
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import pdfplumber

PRICE_DIR = Path.home() / "BVD_PCN"
PCN_RE = re.compile(r"^pcn[-_].*\.pdf$", re.IGNORECASE)
DATE_NAMED_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(_v\d+)?\.pdf$", re.IGNORECASE)
EFFECTIVE_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def extract_effective_date(pdf_path: Path) -> Optional[str]:
    """Pull the 'Effective Date' from page 1.

    BVD's PDF format varies — older sheets show a single date
    (`2026-04-11`), newer ones a range (`2026-05-04 to 2026-05-05`).
    Either way, the first YYYY-MM-DD on page 1 is the effective date.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text() or ""
    except Exception as e:
        print(f"  [error] {pdf_path.name}: {e}", file=sys.stderr)
        return None
    # Look only inside the "Effective Date ..." section to avoid grabbing
    # any random date that might appear later (e.g. inside a row).
    header_idx = text.lower().find("effective date")
    search_in = text[header_idx:header_idx + 200] if header_idx >= 0 else text
    m = EFFECTIVE_DATE_RE.search(search_in)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def organize_pcn_file(pdf_path: Path, dry_run: bool) -> bool:
    date = extract_effective_date(pdf_path)
    if not date:
        print(f"  [skip] {pdf_path.name}: no effective date found in page 1")
        return False
    target_dir = PRICE_DIR / date[:7]   # YYYY-MM
    target = target_dir / f"{date}.pdf"
    rel = target.relative_to(PRICE_DIR)
    if dry_run:
        existing = "  (overwrites existing)" if target.exists() else ""
        print(f"  [dry] {pdf_path.name}  ->  {rel}{existing}")
        return True
    target_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"  [overwrite] {pdf_path.name}  ->  {rel}")
        target.unlink()
    else:
        print(f"  [move]      {pdf_path.name}  ->  {rel}")
    shutil.move(str(pdf_path), str(target))
    return True


def reorganize_existing(dry_run: bool) -> int:
    """Move date-named PDFs at the top of PRICE_DIR into YYYY-MM/ subfolders."""
    count = 0
    for p in sorted(PRICE_DIR.iterdir()):
        if not p.is_file():
            continue
        m = DATE_NAMED_RE.match(p.name)
        if not m:
            continue
        date = m.group(1)
        target_dir = PRICE_DIR / date[:7]
        target = target_dir / p.name
        if target == p:
            continue
        rel = target.relative_to(PRICE_DIR)
        if dry_run:
            existing = "  (target exists, would skip)" if target.exists() else ""
            print(f"  [dry] {p.name}  ->  {rel}{existing}")
            count += 1
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"  [skip] {p.name}: target {rel} already exists")
            continue
        print(f"  [move] {p.name}  ->  {rel}")
        shutil.move(str(p), str(target))
        count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview actions without changing files")
    ap.add_argument("--reorganize-existing", action="store_true",
                    help="Also move legacy date-named PDFs into YYYY-MM/")
    args = ap.parse_args()

    if not PRICE_DIR.exists():
        print(f"Price directory not found: {PRICE_DIR}", file=sys.stderr)
        return 2

    pcn_files = sorted(
        (p for p in PRICE_DIR.iterdir()
         if p.is_file() and PCN_RE.match(p.name)),
        key=lambda p: p.stat().st_mtime,
    )

    print(f"Scanning {PRICE_DIR}")
    if pcn_files:
        print(f"\n=== {len(pcn_files)} PCN file(s) to rename + organize ===")
        for p in pcn_files:
            organize_pcn_file(p, args.dry_run)
    else:
        print("\n(no pcn-*.pdf files at top of folder)")

    if args.reorganize_existing:
        print(f"\n=== Reorganizing existing date-named files ===")
        n = reorganize_existing(args.dry_run)
        if n == 0:
            print("  (nothing to move)")

    if args.dry_run:
        print("\nDry run — no files were changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
