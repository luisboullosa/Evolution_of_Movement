#!/usr/bin/env python3
"""Generate a synthetic progress.json for testing the scan_live viewer."""
import json
import os
import random
from pathlib import Path


def atomic_write(path, data):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    try:
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.remove(str(path))
            os.replace(tmp, str(path))
        except Exception:
            pass


def generate(nx=20, ny=18, seed=42, outdir='output/scan_population'):
    random.seed(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mv_vals = [round(0.05 + i * 0.045, 3) for i in range(nx)]
    mc_vals = [round(0.005 + i * 0.005, 4) for i in range(ny)]

    entries = []
    total = nx * ny
    k = 0
    for i, mc in enumerate(mc_vals):
        for j, mv in enumerate(mv_vals):
            k += 1
            # make a synthetic value: center has high prop_moved
            cx = nx/2; cy = ny/2
            dist = ((i-cy)**2 + (j-cx)**2)**0.5
            prop = max(0.0, 1.0 - (dist / max(nx, ny) * 1.5))
            prop = max(0.0, min(1.0, prop + (random.random()-0.5)*0.15))
            entry = {
                'completed': k,
                'total': total,
                'movement_cost_factor': mv,
                'maintenance_cost_per_node': mc,
                'final_population': int(prop * 2000),
                'prop_moved': round(prop, 4),
            }
            entries.append(entry)

    out = Path(outdir) / 'progress.json'
    atomic_write(out, entries)
    print(f'Wrote sample progress to {out}')


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Generate synthetic progress.json for webviewer')
    p.add_argument('--nx', type=int, default=20)
    p.add_argument('--ny', type=int, default=18)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', type=str, default='output/scan_population')
    args = p.parse_args()
    generate(nx=args.nx, ny=args.ny, seed=args.seed, outdir=args.outdir)
