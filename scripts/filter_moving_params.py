#!/usr/bin/env python3
"""Print parameter combinations where any movement metric > 0.
Usage: python scripts/filter_moving_params.py [--scan-dir PATH]
If --scan-dir omitted, uses the most recent scan under output/Parameter scans/data.
"""

import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scan-dir", help="Path to scan directory (contains scan_extended_results.csv)")
args = parser.parse_args()

base = os.path.join(".", "output", "Parameter scans", "data")
if args.scan_dir:
    scan_dir = args.scan_dir
else:
    if not os.path.isdir(base):
        raise SystemExit(f"No scans found under {base}")
    existing = sorted([d for d in os.listdir(base) if d.startswith("scan")])
    if not existing:
        raise SystemExit(f"No scans found under {base}")
    scan_dir = os.path.join(base, existing[-1])

results_csv = os.path.join(scan_dir, "scan_extended_results.csv")
if not os.path.exists(results_csv):
    raise SystemExit(f"No results CSV at {results_csv}")

out_rows = []
with open(results_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            mm = float(row.get("mean_move", "0") or 0)
            med = float(row.get("median_move", "0") or 0)
            mx = float(row.get("max_move", "0") or 0)
            pm = float(row.get("prop_moved", "0") or 0)
        except Exception:
            continue
        if mm > 0 or med > 0 or mx > 0 or pm > 0:
            out_rows.append(row)

if not out_rows:
    print("No parameter combinations with movement > 0 were found in", results_csv)
else:
    # print header then rows
    header = out_rows[0].keys()
    w = csv.DictWriter(open(os.devnull, "w"), fieldnames=header)
    print(",".join(header))
    for r in out_rows:
        print(",".join(str(r.get(h, "")) for h in header))
