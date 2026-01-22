#!/usr/bin/env python3
"""Generate synthetic scan outputs for a quick end-to-end test.

Writes to `output/scan_population/`:
- `results_short.csv` (csv of rows)
- `progress.json` (atomic write)
- `heatmap.png` (matplotlib heatmap)
"""

import json
import os
import random
from pathlib import Path

outdir = Path(os.environ.get("OUTDIR", "output/scan_population"))
outdir.mkdir(parents=True, exist_ok=True)

mv_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
mc_vals = [0.0001, 0.0005, 0.001, 0.005]

rows = []
entries = []
idx = 0
total = len(mv_vals) * len(mc_vals)
for i, mc in enumerate(mc_vals):
    for j, mv in enumerate(mv_vals):
        idx += 1
        # synthetic metric: prop_moved peaks at medium mv, low mc
        base = max(0.0, 1.0 - abs(j - 2) * 0.25 - i * 0.1)
        prop = max(0.0, min(1.0, base + (random.random() - 0.5) * 0.08))
        final_pop = int(prop * 2000)
        rows.append((mv, mc, 0.0, 0.0, 0.0, 0.0, 50, final_pop, round(prop, 4)))
        entries.append(
            {
                "completed": idx,
                "total": total,
                "movement_cost_factor": mv,
                "maintenance_cost_per_node": mc,
                "final_population": final_pop,
                "prop_moved": round(prop, 4),
            }
        )

# write results_short.csv
csv_path = outdir / "results_short.csv"
header = [
    "movement_cost_factor",
    "maintenance_cost_per_node",
    "reproduction_threshold",
    "reproduction_cost",
    "env_time_speed",
    "death_risk",
    "grid_width",
    "final_population",
    "prop_moved",
]
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(",".join(header) + "\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

# atomic write progress.json
prog_path = outdir / "progress.json"
tmp = prog_path.with_suffix(".json.tmp")
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)
    f.flush()
    try:
        os.fsync(f.fileno())
    except Exception:
        pass
os.replace(tmp, prog_path)

# produce heatmap.png using matplotlib (average prop_moved per cell)
try:
    import matplotlib.pyplot as plt
    import numpy as np

    mvArr = sorted(set([e["movement_cost_factor"] for e in entries]))
    mcArr = sorted(set([e["maintenance_cost_per_node"] for e in entries]))
    z = [[float("nan")] * len(mvArr) for _ in range(len(mcArr))]
    count = [[0] * len(mvArr) for _ in range(len(mcArr))]
    for e in entries:
        i = mcArr.index(e["maintenance_cost_per_node"])
        j = mvArr.index(e["movement_cost_factor"])
        v = e.get("prop_moved", e.get("final_population", 0))
        if not isinstance(v, (int, float)):
            continue
        if not isinstance(z[i][j], float) or not (z[i][j] == z[i][j]):
            z[i][j] = 0.0
        z[i][j] += v
        count[i][j] += 1
    for i in range(len(mcArr)):
        for j in range(len(mvArr)):
            if count[i][j] > 0:
                z[i][j] = z[i][j] / count[i][j]
            else:
                z[i][j] = float("nan")
    arr = np.array(z)
    fig, ax = plt.subplots()
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=(float(min(mvArr)), float(max(mvArr)), float(min(mcArr)), float(max(mcArr))),
    )
    ax.set_xlabel("movement_cost_factor")
    ax.set_ylabel("maintenance_cost_per_node")
    ax.set_title("synthetic prop_moved")
    fig.colorbar(im, ax=ax)
    outp = outdir / "heatmap.png"
    fig.savefig(outp)
    plt.close(fig)
except Exception as exc:
    print("Could not render heatmap:", exc)

print("Synthetic test outputs written to", outdir)
