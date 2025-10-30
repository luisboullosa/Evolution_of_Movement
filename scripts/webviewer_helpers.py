import os
import glob
import json


def iter_individuals_frames(sim_dir):
    """Yield (t, list_of_individual_dicts) from individuals.jsonl or frames.tsv

    This function is tolerant of mixed environments. It parses NDJSON lines from
    individuals.jsonl and returns the 'individuals' list for each frame. If the
    JSONL file is missing, it falls back to frames.tsv and yields empty lists
    (only frame metadata available).
    """
    jsonl_path = os.path.join(sim_dir, "individuals.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("t", None)
                inds = obj.get("individuals", [])
                # ensure last_delta is an int when possible
                for ind in inds:
                    try:
                        if 'last_delta' in ind:
                            ind['last_delta'] = int(ind['last_delta'])
                    except Exception:
                        # keep original value if conversion fails
                        pass
                yield t, inds
        return
    # fallback to legacy TSV per-frame files
    pattern = os.path.join(sim_dir, "individuals.tsv.*.tsv")
    files = sorted(glob.glob(pattern))
    for fn in files:
        # extract t from filename if possible
        t = None
        try:
            base = os.path.basename(fn)
            parts = base.split('.')
            if len(parts) >= 3:
                t = int(parts[1])
        except Exception:
            t = None
        inds = []
        try:
            with open(fn, "r") as f:
                for line in f.read().splitlines()[1:]:
                    if not line.strip():
                        continue
                    parts = line.split('\t')
                    # id,pos,resources,last_spend,last_delta,age -> index mapping
                    try:
                        iid = int(parts[0])
                    except Exception:
                        continue
                    pos = int(parts[1]) if len(parts) > 1 else 0
                    resources = float(parts[2]) if len(parts) > 2 else 0.0
                    last_spend = float(parts[3]) if len(parts) > 3 else 0.0
                    last_delta = int(parts[4]) if len(parts) > 4 else 0
                    age = int(parts[5]) if len(parts) > 5 else 0
                    inds.append({
                        "id": iid,
                        "pos": pos,
                        "resources": resources,
                        "last_spend": last_spend,
                        "last_delta": last_delta,
                        "age": age,
                    })
        except Exception:
            continue
        yield (t, inds)


def iter_env_frames(sim_dir):
    """Yield (t, resources_list) for each frame in sim_dir.
    Supports env.jsonl (one JSON object per line with "t" and "resources") or legacy per-frame TSV env files.
    """
    jsonl_path = os.path.join(sim_dir, "env.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("t", None)
                res = obj.get("resources", [])
                yield (t, res)
        return

    pattern = os.path.join(sim_dir, "env.tsv.*.tsv")
    files = sorted(glob.glob(pattern))
    for fn in files:
        t = None
        try:
            base = os.path.basename(fn)
            parts = base.split('.')
            if len(parts) >= 3:
                t = int(parts[1])
        except Exception:
            t = None
        res = []
        try:
            with open(fn, "r") as f:
                for line in f.read().splitlines()[1:]:
                    if not line.strip():
                        continue
                    parts = line.split('\t')
                    try:
                        res.append(float(parts[1]))
                    except Exception:
                        res.append(0.0)
        except Exception:
            continue
        yield (t, res)
