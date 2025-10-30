"""
clean_webviewer.py
Simple script to list or remove generated webviewer output files.
Usage:
    python webviewer/clean_webviewer.py [--yes] [--base DIR] [--prune-sim-dirs]

Options:
    --yes             Actually delete files. Without this flag the script prints what
                        it would delete.
    --base DIR        Directory where viewer files live (default: webviewer)
    --prune-sim-dirs  After cleaning, optionally remove empty per-simulation folders
                        (dry-run unless --yes).

By default this script only removes top-level generated files in the `webviewer/`
directory and will NOT remove files that live inside simulation subdirectories
(e.g. webviewer/sim0000/).
If you prefer to remove empty simulation folders after cleaning top-level files, pass
`--prune-sim-dirs` (use `--yes` to actually remove folders).

The script removes generated files created by the simulator at the webviewer root:
    - frames.tsv
    - config.json or config.txt
    - nn_index.json
    - nn_*.json (legacy per-id files)
    - env.tsv.*.tsv
    - individuals.tsv.*.tsv

This is safe to run; it won't touch files that don't match these patterns.
"""

import os
import glob
import argparse

parser = argparse.ArgumentParser(description="Clean webviewer-generated output files")
parser.add_argument("--yes", action="store_true", help="Actually delete files")
parser.add_argument(
    "--base", default="webviewer", help="Webviewer directory (default: webviewer)"
)
parser.add_argument(
    "--prune-sim-dirs",
    action="store_true",
    help="Remove empty simulation directories after cleaning (dry-run unless --yes)",
)
args = parser.parse_args()

base = args.base
if not os.path.isdir(base):
    print(f"No such directory: {base}")
    exit(1)

patterns = [
    os.path.join(base, "frames.tsv"),
    os.path.join(base, "config.json"),
    os.path.join(base, "config.txt"),
    os.path.join(base, "nn_index.json"),
    os.path.join(base, "nn_*.json"),
    os.path.join(base, "env.tsv.*.tsv"),
    os.path.join(base, "individuals.tsv.*.tsv"),
]
# NOTE: we intentionally do NOT add per-simulation subfolder patterns here. Files inside
# webviewer/simNNNN/ are left alone by default. Use --prune-sim-dirs to remove empty
# simNNNN directories after cleaning the top-level files.

found = []
for pat in patterns:
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
            except Exception as e:
                print("Failed to delete", f, ":", e)
        print("Done.")
    else:
        print("\nDry-run only. To actually delete these files re-run with --yes")
else:
    print("No webviewer-generated files found at the webviewer root.")
    # do not exit here; allow pruning of sim directories if requested

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
            # directory is empty if it contains no files after the top-level cleaning
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
                    except Exception as e:
                        print("Failed to remove", d, ":", e)
                print("Done.")
            else:
                print(
                    "\nDry-run only. To actually remove, re-run with --yes"
                )
