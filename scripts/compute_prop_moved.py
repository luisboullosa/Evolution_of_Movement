#!/usr/bin/env python3
import json
import os
import sys


def compute(sim_dir):
    path = os.path.join(sim_dir, "individuals.jsonl")
    moved = set()
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    for ind in obj.get("individuals", []):
                        iid = ind.get("id")
                        if iid is None:
                            continue
                        seen.add(iid)
                        try:
                            ld = int(ind.get("last_delta", 0))
                        except Exception:
                            ld = 0
                        if ld != 0:
                            moved.add(iid)
                except Exception:
                    # skip malformed lines
                    continue
    except FileNotFoundError:
        print("ERROR: individuals.jsonl not found in", sim_dir, file=sys.stderr)
        return 2

    prop = float(len(moved)) / float(len(seen)) if seen else 0.0
    print(f"moved={len(moved)}, seen={len(seen)}, prop_moved={prop:.8f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: compute_prop_moved.py <sim_dir>")
        sys.exit(2)
    sys.exit(compute(sys.argv[1]))
