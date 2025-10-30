import os, sys, json
import numpy as np

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

first_mismatch_frame = None
mismatch_frames = []
frame_summaries = []

with open(env_jsonl,'r') as ef, open(ind_jsonl,'r') as inf:
    for t,(el,il) in enumerate(zip(ef,inf)):
        try:
            eobj = json.loads(el)
            iobj = json.loads(il)
        except Exception:
            frame_summaries.append((t, 0, 0, 0))
            continue
        env_res = np.array(eobj.get('resources', []))
        n_inds = 0
        n_checked = 0
        n_mismatch = 0
        for ind in iobj.get('individuals', []):
            n_inds += 1
            try:
                pos = int(ind.get('pos',0))
            except Exception:
                pos = None
            if pos is None:
                n_mismatch += 1
                continue
            if pos < 0 or pos >= len(env_res):
                n_mismatch += 1
                continue
            y_val = float(env_res[pos])
            if not np.isfinite(y_val):
                n_mismatch += 1
                continue
            n_checked += 1
        frame_summaries.append((t, n_inds, n_checked, n_mismatch))
        if n_mismatch > 0:
            mismatch_frames.append(t)
            if first_mismatch_frame is None:
                first_mismatch_frame = t

# print a concise report
print(f"Frames analyzed: {len(frame_summaries)}")
if first_mismatch_frame is None:
    print("No mismatches found in any frame. All plotted points align with env resources.")
else:
    print(f"First mismatch at frame: {first_mismatch_frame}")
    print(f"Total mismatch frames: {len(mismatch_frames)}")

# print summary for first 5 and last 5 frames and any mismatch frames (up to 10)
print('\nSummary (frame, total_inds, checked, mismatches) - first 5 frames:')
for item in frame_summaries[:5]:
    print(item)
print('\n...')
for item in frame_summaries[-5:]:
    print(item)

if mismatch_frames:
    print('\nMismatch frames (up to 10):')
    for f in mismatch_frames[:10]:
        print(f)

# Optionally write CSV of summaries for external inspection
out_csv = os.path.join(base, 'frame_positions_summary.csv')
with open(out_csv, 'w') as of:
    of.write('frame,total_inds,checked,mismatches\n')
    for t,tot,chk,mis in frame_summaries:
        of.write(f"{t},{tot},{chk},{mis}\n")
print(f'Wrote per-frame summary to {out_csv}')
