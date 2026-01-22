# Remote scan runner (Orange Pi)

This adds a minimal SSH/SCP-based workflow to run scan jobs on a remote machine (e.g., an Orange Pi), collect pilot summaries, and fetch population snapshots from deep runs.

## Files

- scripts/remote_scan_orchestrator.py — runs pilot jobs remotely, optionally deep jobs, writes local results + downloads snapshots.
- scripts/remote/remote_run_job.py — helper that runs on the remote host (copied automatically).

## Prereqs

Remote host:
- `cargo` and a Rust toolchain installed.
- Python 3 available (`python3`).
- Either:
	- This repo already present on the remote host (so `rust/Cargo.toml` exists), or
	- You run the orchestrator with `--bootstrap` to upload the repo to the remote machine.

Local (Windows):
- OpenSSH `ssh` and `scp` available in PATH.

## Quick start

1) (Optional) If the repo is already present on the remote host, build once:

- `cd <repo>`
- `cargo build --release --manifest-path rust/Cargo.toml --bin evo_movement_rust`

2) From Windows, run a pilot scan (200 sampled jobs):

- `python scripts/remote_scan_orchestrator.py --remote-host 192.168.2.129 --ssh-user root --bootstrap --workers 1 --sample 200 --steps 200 --initial-population 200 --threshold 200`

If you want to build the Rust bin on the Orange Pi as part of bootstrapping, add `--bootstrap-build`.

3) Pilot + deep with snapshots:

- `python scripts/remote_scan_orchestrator.py --remote-host 192.168.2.129 --ssh-user root --bootstrap --bootstrap-build --workers 1 --sample 200 --steps 200 --initial-population 200 --threshold 200 --deep --deep-steps 10000 --deep-max-population 20000 --max-deep 25`

### Convenience executable (Windows)

Use the PowerShell wrapper (recommended on Windows). It bakes in your Orange Pi defaults and forwards extra arguments to the Python orchestrator:

- Pilot sample:
	- `powershell -ExecutionPolicy Bypass -File scripts/remote_scan.ps1 -Bootstrap -BootstrapBuild -Sample 200 -Workers 1`

- Pilot + deep snapshots:
	- `powershell -ExecutionPolicy Bypass -File scripts/remote_scan.ps1 -Bootstrap -BootstrapBuild -Sample 200 -Workers 1 -Deep -MaxDeep 25`

### Bootstrapping the repo

If the repo is not already on the Orange Pi, pass `--bootstrap`.

That uploads a tarball of your local repo to the remote machine and extracts it into the parent folder of `--remote-repo-dir`.
Defaults are set for your setup:
- `--remote-repo-dir /root/projects/Movement`
- `--remote-run-base /root/projects/scan_runs`

The upload excludes bulky folders like `build/`, `output/`, `rust/target/`, and `.git/`.

## Outputs

The orchestrator creates a local run directory under `output/remote_scan/runYYYYMMDD_HHMMSS/`:
- `results_pilot.csv` — pilot job summaries
- `promising.json` — promising pilot jobs (sorted by final population)
- `promising_after_burnin.json` — promising jobs after burnin evaluation (if `--burnin-eval`)
- `results_deep.csv` — deep job summaries (if `--deep`)
- `deep_snapshots/*.cbor` — downloaded `population_snapshot.cbor` files (if `--deep`)
- `jobs.sqlite` — tracking DB for job status + metadata
- `progress.json` — append-only progress feed

## Burnin evaluation (movement filter)

By default, promising parameter sets are selected based on final population surviving the pilot run. To ensure the selected parameter sets produce *moving* populations rather than just survivors, use `--burnin-eval`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\remote_scan.ps1 `
  -Sample 200 -Workers 2 -BurninEval -Deep
```

This adds a burnin evaluation phase between pilot and deep scans:

1. **Pilot phase**: Runs sampled parameter sets, saves snapshots of promising runs (population ≥ threshold)
2. **Burnin evaluation**: For each promising run:
   - Load the pilot snapshot
   - Randomize the environment (new Perlin seed via `replay_env_seed`)
   - Randomize individual positions (`randomize_start_positions`)
   - Run for 100 steps (configurable with `--burnin-steps`)
   - Measure `prop_moved` (proportion of individuals that moved)
3. **Filter**: Only parameter sets with `prop_moved > burnin_threshold` proceed to deep scans

### Rationale

Some parameter combinations produce stable populations that don't move because they adapt to specific environmental features. Randomizing the environment and individual positions tests whether the evolved behavior is robust and generalizable, ensuring deep scans focus on truly adaptive movement strategies.

### Parameters

- `-BurninEval` / `--burnin-eval`: Enable burnin evaluation phase
- `-BurninSteps <N>` / `--burnin-steps <N>`: Steps to run during burnin (default: 100)
- `-BurninThreshold <X>` / `--burnin-threshold <X>`: Minimum prop_moved to pass (default: 0.0, meaning any movement)

## Performance benchmarks

### Timing estimates (Orange Pi)

Based on parameter scans with varying population sizes and complexity:

- **Pilot jobs** (10,000 steps, pop 50-10,000): ~30-60 seconds per job
- **Deep jobs** (50,000 steps, pop up to 20,000): ~150-300 seconds per job (5x pilot)
- **Conservative average**: 45 sec/pilot, 225 sec/deep

### Sample size calculation

For a target runtime with multiple workers:

```
available_compute = target_hours × num_workers × 3600 sec/hr
pilot_jobs = available_compute / avg_pilot_time
```

Example: 7-hour run with 4 workers:
```
100,800 worker-seconds ÷ 45 sec/job = 2,240 pilot jobs
With deep scan overhead → target ~2,000 samples
```

Command:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\remote_scan.ps1 `
  -Sample 2000 -Workers 4 -Steps 10000 `
  -InitialPopulation 200 -MaxPopulation 10000 -Threshold 10 `
  -Deep -DeepSteps 50000 -DeepMaxPopulation 20000 -MaxDeep 10
```

### Orange Pi vs local machine

| Configuration | Orange Pi (ARM64) | Local Windows (x64) |
|---------------|-------------------|---------------------|
| Architecture | ARM Cortex-A53/A55 | Intel/AMD x86-64 |
| Typical cores | 4-8 | 8-16+ |
| Pilot (10k steps) | 30-60 sec | 10-30 sec (est.) |
| Deep (50k steps) | 150-300 sec | 50-150 sec (est.) |
| Advantage | Low power, always-on | Higher performance |
| Use case | Long overnight scans | Interactive development |

The Orange Pi is ideal for long parameter sweeps that can run unattended, while local execution is better for rapid iteration during development.

## Notes

- Deep jobs use `--output-run-dir` to write outputs into a deterministic per-job folder on the remote machine, making snapshot collection reliable.
- Pilot jobs default to `--no-write-outputs` and do not write per-step files.
- Deep runs are configured to write snapshots, and the orchestrator only downloads `population_snapshot.cbor` ("snapshots only").
- If your SSH uses a non-default key/port, pass `--ssh-key` / `--ssh-port`.
