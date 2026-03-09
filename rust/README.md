# Rust migration workspace

This folder contains an initial Rust workspace to start porting the C++ project.

Quick start:

```bash
cd rust
cargo build
cargo run
```

For details on running simulations, where `simXXXX` output folders are written,
and how to generate TSV/JSONL outputs for the webviewer, see
`RUNNING_SIMULATIONS.md` in this directory and the helper script
`scripts/generate_webviewer_outputs.ps1` in the repository root.

Interactive dashboards

----------------------
If you want to explore the environment/NN behavior without running the binary, start the Dash app at `webviewer/interactive_dash.py` (from repo root):

```bash
pip install dash plotly numpy
python webviewer/interactive_dash.py --port 8050
```

It mirrors the Rust model's Perlin-ish environment, NN movement threshold (|resp| >= 0.5 moves; sign sets direction), and includes a coarse carrying-capacity (K) estimator based on maintenance cost and death risk.

Next steps: port core modules from `src/` in the C++ project into Rust modules under `rust/src/`.
