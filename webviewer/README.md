**Overview**

This folder contains the interactive visualization tools used to inspect simulator
replays. Two main viewer scripts are provided:

- **`visualize_population.py`**: shows the full population on the environment as a
  clickable scatter. Click an individual to inspect its reaction-norm and small
  neural-network diagram. Supports headless screenshots with `--headless`.

- **`visualize_individual.py`**: focused single-individual inspector. Shows
  per-frame timelines, response traces and detailed NN visualizations. Also
  supports headless/screenshot mode.

Files expected in a simulation/run folder (the viewers read these):

- `env.jsonl` — NDJSON with per-frame environment state (resources, time `t`).
- `individuals.jsonl` — NDJSON with per-frame list of individuals and fields
  like `id`, `pos`, `resources`, `last_spend`, etc.
- `nn_index.json` (optional) — maps neural-network ids to weight/bias arrays.

**Quick usage**

- Run population viewer (interactive):
  - `python webviewer/visualize_population.py path/to/simdir`

- Run individual viewer (interactive):
  - `python webviewer/visualize_individual.py path/to/simdir`

- Headless screenshot (population):
  - `python webviewer/visualize_population.py path/to/simdir --headless`

**Interactive (Shiny-like) playground**

- Start the Dash app to experiment without a recorded run:
  - `python webviewer/interactive_dash.py --port 8050`
- Requires: `pip install dash plotly`
- Panels:
  1) Environment explorer: tune seed/scale/time and watch the 1D Perlin-ish landscape evolve.
  2) NN response explorer: edit weights/biases and see the movement vs gather decision (|response| >= 0.5 moves; sign sets direction).
  3) Carrying-capacity (K) estimator: rough bound using maintenance cost, resource scale, and death risk (caps at 50k for readability).

**Creating a self-contained webviewer bundle**

If you want a single folder containing the static viewer code and a `run/`
subfolder with the replay outputs, use the helper script `make_bundle.py`.

Example:

```
python webviewer/make_bundle.py --sim-dir output/visualize/replay_pilot_000005 \
    --out output/visualize/replay_pilot_000005_webviewer_bundle

# Then serve the bundle locally:
cd output/visualize/replay_pilot_000005_webviewer_bundle
python -m http.server 8001
# Open http://localhost:8001 in your browser and point to the run folder
```

The helper copies the `webviewer/` static files (Python viewers + helpers)
into the bundle and the `sim` run files into `bundle/run/`. It avoids copying
`__pycache__` or large intermediate build outputs.

**Selector to launch viewers**

- List runs/snapshots and launch viewers:
  - `python webviewer/select_and_visualize.py --viewer both`
- Optional: filter by build id and replay a snapshot to 1000 steps before viewing:
  - `python webviewer/select_and_visualize.py --build-id 47a0979 --viewer population`
- To skip the prompt, pass `--select <index>` after reading the printed list.

**Notes & troubleshooting**

- If copying fails on Windows due to open handles or permission issues, close
  any running Python/cargo processes that might hold files and retry.
- The viewers expect `env.jsonl` and `individuals.jsonl` to be present and
  readable (NDJSON); they will fall back to per-frame TSVs if available.

If you'd like, I can run the bundle step now and verify the resulting bundle.