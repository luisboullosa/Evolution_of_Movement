# evo-movement-sim

Minimal C++ simulation of evolving movement in digital individuals.

Build (Windows, PowerShell):

```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022"; cmake --build . --config Release
```

Run (Release):

```powershell
.
# from build/Release or run the exe directly
./evo-movement-sim.exe
```

This project is intentionally small and dependency-free (uses only the C++ standard library).

Rust port
---------
A Rust port of the simulator lives in the `rust/` subdirectory. To build and run the Rust
binary use Cargo:

```powershell
pushd rust
cargo build --release
./target/release/evo_movement_rust.exe --help
popd
```

Note: Perlin temporal evolution is always enabled in the Rust port (the previous
`--perlin-time`/`--no-perlin-time` toggle and `env_change_rate` additive-noise option
have been removed). Use the `--env-time-speed`, `--env-time-amplitude`,
`--env-perlin-spatial-scale` and `--env-perlin-time-scale` flags to tune the time-varying
component.

Web viewer
----------
The `webviewer/index.html` page is a legacy viewer that loads a TSV frames file (typically `frames.tsv`).
This workflow is deprecated; newer runs use JSONL/CBOR outputs (e.g. `env.jsonl`, `individuals.jsonl`/`individuals.cbor`, `nn_index.json`).

Scan driver (Rust)
------------------
A convenience driver for batch parameter scans using the Rust binary is provided at `scripts/scan_rust_population_scan.py`.
It performs a grid or sampled parameter sweep and writes summary outputs for consumption by the webviewer.

Key flags:
- `--outdir`: output base directory (default `output/scan_population`). A single timestamped `jobs...` folder will be created under this path containing summary files.
- `--workers`: number of parallel workers for the sweep.
- `--steps`: number of simulation steps for short/pilot runs.
- `--initial-population`: initial population size for short runs.
- `--pilot`: use a smaller, pilot parameter grid.
- `--no-write-outputs`: tell the Rust simulator to suppress per-step outputs (env.jsonl, individuals.jsonl, frames, nn_index).
- `--no-deep`: do not run deeper follow-up simulations for promising parameter sets.
- `--deep-steps` / `--deep-population`: when deep runs are enabled, control their length and starting population.

Behavior:
- The driver writes per-job temporary config files to the system temp directory so the output folder only contains the summary files.
- It forces `report_interval` to be larger than the number of steps and sets `write_outputs`/`write_frames` to `false` by default to avoid producing per-step files when possible.
- Summary files produced in the jobs folder include `results_short.csv`, `deep_results.csv` (if deep runs executed), `progress.json`, and `promising.csv`.

Example (short pilot, no deep runs):

```powershell
python scripts\scan_rust_population_scan.py --pilot --workers 2 --steps 200 --initial-population 100 --outdir output/scan_population --no-deep
```

After the scan starts, open the webviewer at `webviewer/scan_live.html` served from a local HTTP server (see below) to watch `progress.json` updates.

Remote scan runner (Orange Pi)
------------------------------
For distributing scan jobs to a remote Linux machine over SSH/SCP (e.g. an Orange Pi) and fetching deep-run population snapshots, see `scripts/REMOTE_SCAN.md`.

On Windows, you can use the convenience wrapper `scripts/remote_scan.ps1`.

Cleaning webviewer output
-------------------------
The `webviewer/`-root output layout and associated cleanup tooling are deprecated.
Historically, the simulator wrote generated files directly into `webviewer/` (timestamped `env.tsv.*.tsv`,
`individuals.tsv.*.tsv`, `nn_index.json`, and sometimes `frames.tsv`).
Newer workflows write into `output/prod_runs/simXXXX[/runYYYY]/`.

From PowerShell (dry-run):

```powershell
python scripts\clean_webviewer.py
```

To actually delete the discovered files add `--yes`:

```powershell
python scripts\clean_webviewer.py --yes
```

Alternatively, if you use the included CMake target you can run:

```powershell
# from the project root
cmake --build build --target clean-webviewer --config Release
```

The CMake target runs the cleaner in dry-run mode by default and prints which files would be removed; to actually delete files run the Python script with `--yes` as shown above.

Running with burn-in and steps
-----------------------------
You can set a burn-in period (timesteps to skip before recording frames/snapshots) and number of steps from the command line:

```powershell
# run 500 timesteps but only start recording after timestep 200
.
# from build/Release or run the exe directly
./evo-movement-sim.exe --burn-in=200 --steps=500 --config-out=webviewer\simNNNN\config.json
```

`--burn-in=N` causes the simulator to perform all timesteps but suppress console/frame/snapshot reporting until `t >= N`.



Running the tests
-----------------
Test instructions were moved to `TESTS.md`. See `TESTS.md` for commands to build and run the project's tests.

