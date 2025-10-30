import os
import pandas as pd
import json

base = os.path.join('.', 'output', 'Parameter scans', 'data')
scans = [d for d in os.listdir(base) if d.startswith('scan')]
if not scans:
    raise SystemExit('No scans found')
# pick the most recently modified scan directory
scan = max(scans, key=lambda d: os.path.getmtime(os.path.join(base, d)))
scan_dir = os.path.join(base, scan)
results_csv = os.path.join(scan_dir, 'scan_extended_results.csv')
if not os.path.exists(results_csv):
    raise SystemExit(f'results CSV not found: {results_csv}')

print('Using scan dir:', scan_dir)
print('Reading results:', results_csv)

df = pd.read_csv(results_csv)
# ensure numeric columns
for c in ['mean_move','median_move','max_move','prop_moved','recorded_steps']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

# filter combos with prop_moved > 0.3 and extinction_time >= 100000
candidates = df[(df['prop_moved'] > 0.3) & (df['extinction_time'] >= 100000)]

out_cmds = []
if candidates.empty:
    print('No candidate rows found with prop_moved>0.3 and extinction_time>=100000')
else:
    print('Found', len(candidates), 'candidates')
    for _, row in candidates.iterrows():
        m = row['maintenance_cost_per_node']
        rt = row['reproduction_threshold']
        rc = row['reproduction_cost']
        ets = row['env_time_speed']
        dr = row['death_risk']
        mp = int(row['max_population']) if 'max_population' in row else 0
        # build command - use large initial population for better sampling (50k)
        cmd = []
        exe = os.path.normpath('./build/Release/evo-movement-sim.exe')
        if not os.path.exists(exe):
            exe = os.path.normpath('./build/evo-movement-sim.exe')
        cmd.append(exe)
        cmd.append(f'--steps=1001000')
        cmd.append(f'--burn-in=1000000')
        cmd.append(f'--initial-population=50000')
        cmd.append(f'--max-population={mp}')
        cmd.append(f'--death-risk={dr}')
        cmd.append(f'--maintenance-cost-per-node={m}')
        cmd.append(f'--reproduction-threshold={rt}')
        cmd.append(f'--reproduction-cost={rc}')
        cmd.append(f'--env-time-speed={ets}')
        cmd.append('--write-config')
        cmd.append('--write-frames')
        # write to same scan dir but create distinct output folder per run (timestamp)
        #cmd.append(f'--out-dir={os.path.join(scan_dir, f"longrun_m{m}_rt{rt}_rc{rc}_ets{ets}_dr{dr}_mp{mp}")}')
        out_cmds.append(' '.join(map(str, cmd)))

# write commands file
cmd_file = os.path.join(scan_dir, 'run_long_sims.ps1')
with open(cmd_file, 'w', encoding='utf-8') as cf:
    cf.write('# PowerShell script to run long evo-movement-sim.exe runs for selected parameter combos\n')
    cf.write('# Created by generate_long_run_commands.py\n')
    cf.write('Set-StrictMode -Version Latest\n')
    cf.write('if (-not (Test-Path ".\\build\\Release\\evo-movement-sim.exe")) { Write-Host "Warning: Release exe not found; ensure path is correct" }\n')
    cf.write('\n')
    if not out_cmds:
        cf.write('Write-Host "No candidate parameter combos found; nothing to run."\n')
    else:
        for c in out_cmds:
            cf.write(f'Write-Host "Running: {c}"\n')
            cf.write(f'{c}\n')

print('Wrote commands to', cmd_file)

# Simple analysis: compute Pearson correlations of numeric params vs prop_moved and survived
'''num_cols = []
for c in ['maintenance_cost_per_node','reproduction_threshold','reproduction_cost','env_time_speed','death_risk','max_population','prop_moved','survived']:
    if c in df.columns:
        num_cols.append(c)

analysis = []
if 'prop_moved' in df.columns:
    for p in ['maintenance_cost_per_node','reproduction_threshold','reproduction_cost','env_time_speed','death_risk','max_population']:
        if p in df.columns:
            corr = df[p].corr(df['prop_moved'])
            analysis.append((p, 'prop_moved', corr))
if 'survived' in df.columns:
    for p in ['maintenance_cost_per_node','reproduction_threshold','reproduction_cost','env_time_speed','death_risk','max_population']:
        if p in df.columns:
            corr = df[p].corr(df['survived'])
            analysis.append((p, 'survived', corr))

analysis_file = os.path.join(scan_dir, 'scan_analysis.txt')
with open(analysis_file, 'w', encoding='utf-8') as af:
    af.write('Simple correlation analysis between parameters and outcomes\n')
    af.write('Parameter,Outcome,PearsonCorr\n')
    for p,o,corr in analysis:
        af.write(f'{p},{o},{corr}\n')

print('Wrote analysis to', analysis_file)
'''

print('Done')
