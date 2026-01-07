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
- `results_deep.csv` — deep job summaries (if `--deep`)
- `deep_snapshots/*.cbor` — downloaded `population_snapshot.cbor` files (if `--deep`)
- `jobs.sqlite` — tracking DB for job status + metadata
- `progress.json` — append-only progress feed

## Notes

- Deep jobs use `--output-run-dir` to write outputs into a deterministic per-job folder on the remote machine, making snapshot collection reliable.
- Pilot jobs default to `--no-write-outputs` and do not write per-step files.
- Deep runs are configured to write snapshots, and the orchestrator only downloads `population_snapshot.cbor` ("snapshots only").
- If your SSH uses a non-default key/port, pass `--ssh-key` / `--ssh-port`.
