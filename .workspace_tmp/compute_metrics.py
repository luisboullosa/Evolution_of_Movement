import json, statistics, sys, os

# Path to the pilot sim individuals JSONL
p = r"d:\\Development\\Movement\\output\\Parameter scans\\data\\scan0015\\sim0000\\individuals.jsonl"

if not os.path.exists(p):
    print({'error': 'not found', 'path': p})
    sys.exit(1)


def extract_json_objects(s):
    """Yield JSON object substrings found in string s by scanning for balanced braces.

    This is resilient to extra trailing text or concatenated objects on the same line.
    """
    i = 0
    n = len(s)
    while i < n:
        # find next opening brace
        start = s.find('{', i)
        if start == -1:
            return
        depth = 0
        j = start
        while j < n:
            ch = s[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    # yield substring from start..j inclusive
                    yield s[start:j+1]
                    i = j+1
                    break
            j += 1
        else:
            # no matching closing brace in this string
            return


frames = 0
records = 0
moved = 0
vals = []

with open(p, 'r', encoding='utf-8') as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        frames += 1
        for obj_str in extract_json_objects(line):
            try:
                obj = json.loads(obj_str)
            except Exception:
                # skip objects we can't parse
                continue
            for ind in obj.get('individuals', []):
                ld = ind.get('last_delta', 0)
                try:
                    v = float(ld)
                except Exception:
                    continue
                records += 1
                if abs(v) > 0:
                    moved += 1
                vals.append(abs(v))

summary = {'file': p, 'frames': frames, 'individual_records': records}
if vals:
    summary.update({
        'mean_move': round(statistics.mean(vals), 6),
        'median_move': round(statistics.median(vals), 6),
        'max_move': float(max(vals)),
        'prop_moved': round(moved / records, 6),
    })

print(summary)
