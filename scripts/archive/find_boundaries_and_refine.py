import csv
import json
import os

output_dir: str = os.path.join(os.path.normpath("."), "output", "Parameter scans")
csvpath: str = os.path.normpath(os.path.join(output_dir, "scan_focused_results.csv"))
if not os.path.exists(csvpath):
    csvpath: str = os.path.normpath("scan_focused_results.csv")
    if not os.path.exists(csvpath):
        print(
            "scan_focused_results.csv not found at",
            os.path.relpath(os.path.join(output_dir, "scan_focused_results.csv")),
        )
        raise SystemExit(1)

pts = {}
movement_vals = set()
maintenance_vals = set()
with open(csvpath, "r", newline="") as f: os.TextIOWrapper[_WrappedBuffer]:
    r: csv.DictReader[str] = csv.DictReader(f)
    for row: dict[str | Any, str | Any] in r:
        mv = float(row["movement_cost_factor"])
        mc = float(row["maintenance_cost_per_node"])
        surv = int(row.get("survived", "0"))
        pts[(mv, mc)] = surv
        movement_vals.add(mv)
        maintenance_vals.add(mc)

movement_vals = sorted(movement_vals)
maintenance_vals = sorted(maintenance_vals)


# helper to get neighbors in grid (4-neighborhood)
def neighbors(mv, mc):
    res = []
    try:
        i = movement_vals.index(mv)
        j = maintenance_vals.index(mc)
    except ValueError:
        return res
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni = i + di
        nj = j + dj
        if 0 <= ni < len(movement_vals) and 0 <= nj < len(maintenance_vals):
            res.append((movement_vals[ni], maintenance_vals[nj]))
    return res


boundary_cells = []
for (mv, mc), surv in pts.items():
    for n in neighbors(mv, mc):
        if pts.get(n, surv) != surv:
            boundary_cells.append((mv, mc))
            break

boundary_cells = sorted(set(boundary_cells))
print("Found", len(boundary_cells), "boundary cells")

# For each boundary cell, create a finer local grid around the point
refinements = []
for mv, mc in boundary_cells:
    # create a small window +/- one coarse step where available
    i: int = movement_vals.index(mv)
    j: int = maintenance_vals.index(mc)
    mv_left = movement_vals[max(0, i - 1)]
    mv_right = movement_vals[min(len(movement_vals) - 1, i + 1)]
    mc_left = maintenance_vals[max(0, j - 1)]
    mc_right = maintenance_vals[min(len(maintenance_vals) - 1, j + 1)]
    # create 6 points between left and right (inclusive)
    mv_range = [round(mv_left + (mv_right - mv_left) * t / 6.0, 6) for t: int in range(7)]
    mc_range = [round(mc_left + (mc_right - mc_left) * t / 6.0, 6) for t: int in range(7)]
    refinements.append(
        {
            "center": [mv, mc],
            "movement_range": mv_range,
            "maintenance_range": mc_range,
        }
    )

out = {"refinements": refinements}
outpath: str = os.path.normpath(os.path.join("scripts", "refine_ranges.json"))
with open(outpath, "w") as f: os.TextIOWrapper[_WrappedBuffer]:
    json.dump(out, f, indent=2)

print("Wrote", os.path.relpath(outpath))
