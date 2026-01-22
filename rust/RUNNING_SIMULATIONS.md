# Running Rust Simulations and Outputs

This document describes how to run the Rust simulation binary, where outputs are written, and how to generate the TSV/JSONL data used by the web viewer.

## 1. Binaries and entry points

- Crate root: `rust/`
- Main CLI entrypoints:
  - Library: `evo_movement_rust`
  - Binaries:
    - `evo_movement_rust` (current CLI)
    - `evo_movement_bin_old` (legacy CLI, kept for reference/tests)

To build everything:

```powershell
cd rust
cargo build
```

To run the current CLI with defaults:

```powershell
cd rust
# Running Rust Simulations and Outputs

This document describes how to run the Rust simulation binary, where outputs are written, and how to generate the TSV/JSONL data used by the web viewer.

## 1. Binaries and entry points

- Crate root: `rust/`
- Main CLI entrypoints:
  - Library: `evo_movement_rust`
  - Binaries:
    - `evo_movement_rust` (current CLI)
    - `evo_movement_bin_old` (legacy CLI, kept for reference/tests)

To build everything:

```powershell
cd rust
cargo build
```

To run the current CLI with defaults:

```powershell
cd rust
cargo run --bin evo_movement_rust
```

This uses the default `Config` from `src/config.rs` and behaves as described below.

## 2. Default output layout

By default, the Rust simulation writes per-run artifacts under the repository root:

- Base directory: `output/prod_runs`
- Per-run directory: `output/prod_runs/simXXXX`
  - `sim0000`, `sim0001`, etc., auto-incremented.

Each run directory may contain:

- `config.json` – Resolved configuration used for this run (always written).
- `population_snapshot.cbor` – Binary snapshot of the final environment and population (always written).
- `frames.tsv` – (legacy / deprecated) ASCII frame data, one row per reporting step.
- `env.jsonl` – Environment NDJSON (written when detailed outputs are enabled).
- `individuals.jsonl` or `individuals.cbor` – Per-timestep individuals (written when detailed outputs are enabled).
- `nn_index.json` – Neural network metadata index (written when detailed outputs are enabled).

The exact encoding of `individuals.*` is controlled by `Config.individuals_encoding` (see below).

## 3. Config defaults and flags

Key defaults from `Config::default()` (see `src/config.rs`):

- `output_dir_base = "output/prod_runs"`
- `write_config = true`
- `write_snapshot = true`
- `write_outputs = false`  
  → By default, runs write only `config.json` and `population_snapshot.cbor` into `output/prod_runs/simXXXX`.
- `write_frames = true` (deprecated; only used when `write_outputs` is true).
- `show_progress = true`  
  → Default runs show a textual progress bar in the terminal.

Individuals encoding options (`Config.individuals_encoding`):

- `"jsonl"`, `"json"`, `"pretty"`, `"ndjson"` → human-readable JSONL (`individuals.jsonl`).
- `"jsonl-compact"`, `"jsonl_compact"`, `"compact"`, `"json-compact"`, `"compact-json"` → compact JSONL (`individuals.jsonl`).
- `"cbor"` → compact CBOR (`individuals.cbor`).

### CLI flags (current `evo_movement_rust` binary)

The CLI in `src/main.rs` supports a subset of configuration via flags. Important ones for outputs:

- `--write-outputs`  
  Enable detailed outputs (`env.jsonl`, `individuals.*`, `nn_index.json`, and optionally legacy `frames.tsv`).
- `--no-write-outputs`  
  Disable detailed outputs (only `config.json` and `population_snapshot.cbor`).
- `--config-file=PATH`  
  Load a JSON config file and then apply subsequent CLI overrides.
- `--steps=N`  
  Override `Config.steps`.
- `--report-interval=N`  
  Override `Config.report_interval`.
- `--initial-population=N`  
  Override `Config.initial_population`.
- `--no-progress` / `--show-progress`  
  Toggle the console progress bar.

Other flags configure environment dynamics, movement cost, maintenance cost, etc. See `src/main.rs` for the full list.

## 4. Typical workflows

### 4.1 Minimal production run (config + snapshot only)

This is the default behavior: run a simulation, but only persist the binary population snapshot and the resolved config.

Example:

```powershell
cd rust
cargo run --bin evo_movement_rust -- `
  --steps=100 `
  --initial-population=2000
```

Results:

- New directory: `output/prod_runs/simXXXX` (e.g. `sim0000`).
- Files:
  - `output/prod_runs/simXXXX/config.json`
  - `output/prod_runs/simXXXX/population_snapshot.cbor`

No `env.jsonl`, `individuals.*`, or `nn_index.json` are written in this mode.

### 4.2 Full outputs (JSONL/CBOR)

When you want the full per-timestep outputs directly from the binary:

```powershell
cd rust
cargo run --bin evo_movement_rust -- `
  --steps=100 `
  --initial-population=2000 `
  --write-outputs `
  --no-print-frames
```

- `--write-outputs` enables writing `env.jsonl`, `individuals.jsonl` or `individuals.cbor`, and `nn_index.json`.
- `frames.tsv` is legacy/deprecated and is controlled by `write_frames`.
- `--no-print-frames` suppresses verbose ASCII frames in the terminal; it does not affect the files.

Resulting directory (for a new run):

- `output/prod_runs/simXXXX/config.json`
- `output/prod_runs/simXXXX/population_snapshot.cbor`
- `output/prod_runs/simXXXX/env.jsonl`
- `output/prod_runs/simXXXX/individuals.jsonl` (or `individuals.cbor`)
- `output/prod_runs/simXXXX/nn_index.json`

The legacy `webviewer/index.html` expects a TSV frames file (typically `frames.tsv`).

## 5. PS1 helper: generate webviewer outputs from a config

There is a convenience PowerShell script that runs the Rust binary with the correct flags for webviewer outputs:

- Script: `scripts/generate_webviewer_outputs.ps1`

Usage:

```powershell
cd D:\Development\Movement

# Debug build
./scripts/generate_webviewer_outputs.ps1 ./output/prod_runs/sim0000/config.json

# Release build
./scripts/generate_webviewer_outputs.ps1 ./output/prod_runs/sim0000/config.json -Release
```

What it does:

- Resolves the repository root from the script location.
- Changes into `rust/`.
- Runs one of:
  - `cargo run --bin evo_movement_rust -- --config-file <ConfigPath> --write-outputs --no-print-frames`
  - `cargo run --release --bin evo_movement_rust -- --config-file <ConfigPath> --write-outputs --no-print-frames`

This produces a **new** `simYYYY` directory under `output/prod_runs/` with full webviewer outputs, while keeping the original `simXXXX` (with its `config.json` and `population_snapshot.cbor`) unchanged.

Note: each new run also copies a `generate_webviewer_outputs.ps1` helper into its own `simXXXX` directory, so you can:

```powershell
cd D:\Development\Movement\output\prod_runs\sim0000
./generate_webviewer_outputs.ps1 ./config.json
```

to generate full webviewer outputs based on that run's config.

## 6. Individuals binary and replay

The `population_snapshot.cbor` file is the compact binary representation of the final environment and population from a run.

- Location: `output/prod_runs/simXXXX/population_snapshot.cbor`.
- Created automatically at the end of every run (default `write_snapshot = true`).

The `Simulation::load_snapshot_from_path` method allows replaying from such a snapshot (used by the legacy binary in `src/bin/evo_movement_rust.rs`). For new workflows, the main use is to keep a compact record of the final population while generating webviewer-friendly outputs on demand via config.

## 7. Quick reference

- **Minimal production run (config + snapshot only)**
  - `cargo run --bin evo_movement_rust`
- **Production run with full outputs (TSV + JSONL/CBOR)**
  - `cargo run --bin evo_movement_rust -- --write-outputs --no-print-frames`
- **Postprocess an existing config into full outputs**
  - `./scripts/generate_webviewer_outputs.ps1 ./output/prod_runs/simXXXX/config.json`

All run artifacts are placed under `output/prod_runs/simXXXX` from the repository root.