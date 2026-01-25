"""
clean_webviewer.py
Simple script to list or remove generated webviewer output files.

This file was moved from webviewer/ to scripts/ to centralize helper scripts.
"""

import argparse
import glob
import os

parser = argparse.ArgumentParser(description="Clean webviewer-generated output files")
parser.add_argument("--yes", action="store_true", help="Actually delete files")
parser.add_argument("--base", default="webviewer", help="Webviewer directory (default: webviewer)")
parser.add_argument(
    "--prune-sim-dirs",
    action="store_true",
    help="Remove empty per-simulation directories (dry-run unless --yes)",
)
args: argparse.Namespace = parser.parse_args()

base = args.base
if not os.path.isdir(base):
    print(f"No such directory: {base}")
    exit(1)

print(
    "[deprecated] clean_webviewer.py is legacy. The simulator now writes outputs under output/prod_runs/simXXXX[/runYYYY]. "
    "This script may be removed in the future."
)

patterns: list[str] = [
    os.path.join(base, "config.json"),
    os.path.join(base, "config.txt"),
    os.path.join(base, "nn_index.json"),
    os.path.join(base, "nn_*.json"),
    os.path.join(base, "env.tsv.*.tsv"),
    os.path.join(base, "individuals.tsv.*.tsv"),
]

found = []
for pat: str in patterns:
    found.extend(sorted(glob.glob(pat)))

if found:
    print("Found the following webviewer-generated files:")
    for f in found:
        print("  ", f)

    if args.yes:
        print("\nDeleting files...")
        for f in found:
            try:
                os.remove(f)
                print("Deleted", f)
            except Exception as e: Exception:
                print("Failed to delete", f, ":", e)
        print("Done.")
    else:
        print("\nDry-run only. To actually delete these files re-run with --yes")
else:
    print("No webviewer-generated files found at the webviewer root.")

# optionally prune empty per-simulation directories
if args.prune_sim_dirs:
    sim_dirs = [
        os.path.join(base, d)
        for d in sorted(os.listdir(base))
        if os.path.isdir(os.path.join(base, d)) and d.startswith("sim")
    ]
    if not sim_dirs:
        print("\nNo simulation directories found to prune.")
    else:
        print("\nFound simulation directories:")
        for d in sim_dirs:
            print("  ", d)
        to_remove = []
        for d in sim_dirs:
            try:
                if not any(os.scandir(d)):
                    to_remove.append(d)
            except Exception:
                pass
        if not to_remove:
            print("\nNo empty simulation directories to remove.")
        else:
            print("\nEmpty simulation directories that would be removed:")
            for d in to_remove:
                print("  ", d)
            if args.yes:
                print("\nRemoving empty simulation directories...")
                for d in to_remove:
                    try:
                        os.rmdir(d)
                        print("Removed", d)
                    except Exception as e: Exception:
                        print("Failed to remove", d, ":", e)
                print("Done.")
            else:
                print("\nDry-run only. To remove these directories re-run with --yes")
