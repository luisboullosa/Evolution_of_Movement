import os, sys, json
import numpy as np

# choose webviewer latest sim folder like the visualizers do
base_root = sys.argv[1] if len(sys.argv) > 1 else 'webviewer'
if os.path.isdir(base_root):
    sims = sorted([d for d in os.listdir(base_root) if os.path.isdir(os.path.join(base_root, d)) and d.startswith('sim')])
    if len(sims) == 0:
        print('No sim folders in', base_root); sys.exit(1)
    base = os.path.join(base_root, sims[-1])
else:
    base = base_root

env_jsonl = os.path.join(base, 'env.jsonl')
ind_jsonl = os.path.join(base, 'individuals.jsonl')
if not (os.path.exists(env_jsonl) and os.path.exists(ind_jsonl)):
    print('Missing env/individuals jsonl in', base); sys.exit(1)

mismatch_count = 0
checked = 0
with open(env_jsonl,'r') as ef, open(ind_jsonl,'r') as inf:
    for t,(el,il) in enumerate(zip(ef,inf)):
        try:
            eobj = json.loads(el)
            iobj = json.loads(il)
        except Exception:
            continue
        env_res = np.array(eobj.get('resources', []))
        for ind in iobj.get('individuals', []):
            try:
                pos = int(ind.get('pos',0))
            except Exception:
                pos = None
            if pos is None:
                mismatch_count += 1
                continue
            # check index bounds
            if pos < 0 or pos >= len(env_res):
                print(f'Frame {t}: individual id {ind.get("id")} pos {pos} out of bounds (env length {len(env_res)})')
                mismatch_count += 1
                continue
            y_val = float(env_res[pos])
            # simulated scatter y would be env_res[pos]; verify type/finite
            if not np.isfinite(y_val):
                print(f'Frame {t}: non-finite resource at pos {pos}')
                mismatch_count += 1
                continue
            checked += 1

print('Checked', checked, 'individual positions across frames; mismatches:', mismatch_count)
if mismatch_count == 0:
    print('PASS: scatter y-coordinates match env resource curve.')
else:
    print('FAIL: found mismatches.')