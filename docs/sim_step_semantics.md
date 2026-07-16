# Simulation Step Semantics

All CPU and GPU paths share logic in `rust/src/step_engine.rs`. The canonical step is:

## Phase 1 — Snapshot occupancy

At the start of each step, build `occupancy[pos]` = count of individuals at each cell.

Forward passes use this **snapshot** for sensor input index 6 (`local_population`) and for gather denominators. Forward does not see occupancy changes from earlier individuals in the same step.

## Phase 2 — Sensors

### Perceptron (3 dimensions)

| Index | Source |
| --- | --- |
| 0 | Resource at current hex (`env.resources[pos]`) |
| 1 | Snapshot occupancy at position (competition) |
| 2 | Individual internal resource stores |

### Other policies (8 dimensions)

| Index | Source |
| --- | --- |
| 0–5 | Edge resources (ring: `[L,L,C,R,R,C]`; hex: six neighbor cells) |
| 6 | Snapshot occupancy at position |
| 7 | Individual resources |

Gather / rest-forage uses the **current hex** resource (`env.resources[pos]`) divided by local population, not the neighbor average.

## Phase 3 — Brain forward

`BrainGenome::forward` dispatches by kind:

- **Perceptron** — multi-input layer → `move_signal = sum(hidden)`, `direction = hidden[1]` or `hidden[0]`
- **MLP** — `NeuralNetwork` decode (same as legacy `decode_outputs_from_layer`)
- **SpatialGraph** — observe/update embeddings, message passing, readout (mutates `spatial_runtime`)
- **GooGraph** — topo-ordered ops; `KeepBest` tie-break = lowest parent id

## Phase 4 — Decision row

From forward outputs (no position/env mutation yet):

- `move_signal <= 0` → gather: `gathered = center / max(local_pop, 1)`; no movement spend
- `move_signal > 0` → move: `movement_spend = min(abs(move_signal) * movement_cost_factor, resources)`

Direction: `direction_degrees = ((tanh(direction_signal)+1)/2 clamped) * 360`

## Phase 5 — Apply (sequential, index order)

For each individual in stable index order:

- **Gather** — add `gathered` to resources; subtract `gathered` from `env.resources[pos]`
- **Move** — when `move_signal > 0` and `move_cooldown == 0`: subtract `movement_spend`, move one hex, set `move_cooldown = movement_interval - 1`
- **Move wait** — when `move_signal > 0` but `move_cooldown > 0`: decrement cooldown; rest-forage at current cell (`center / local_pop`)

`movement_interval` (from `movement_speed` preset or explicit config): 1 = fast (one hex/step), 5 = medium, 25 = slow.

## Phase 6 — Age, reproduction, metabolism, death

After movement/gather (Phases 1–5):

1. **Age** — every individual’s `age` increments by 1.
2. **Reproduction** — scan individuals in stable index order. An individual reproduces when `resources >= reproduction_threshold` and `population < max_population` (`max_population == 0` means unlimited). Each eligible individual may reproduce at most once per step; reproduction stops when the cap is reached. Parent pays `reproduction_cost`; child is a mutated clone (with optional structural mutation per `architecture_policy`). Child spawns on the parent’s cell.
3. **Metabolism** — subtract `basal_metabolic_cost + maintenance_cost_per_node × brain_complexity` from each survivor.
4. **Death** — remove individuals with `resources <= 0`, and apply optional age-dependent `death_risk`.

Headless and interactive both use `population_dynamics.rs` for phases 2–4.

## GPU path

When `config.use_gpu_step = true` and the population fits GPU packing caps:

1. `PopulationBrainTensor::sync_from_individuals`
2. `brain_step_kernel` — one thread per individual; writes `DecisionRowGpu` only
3. `apply_decisions_sequential` on CPU (same as CPU path)
4. If the kernel ran, copy `spatial_runtime` embeddings back from device

CUDA init failure falls back to CPU forward with identical semantics.

## Config

- `use_gpu_step` (default `false`) — opt-in GPU forward
- `max_brain_nodes` — padding cap; oversize individuals force CPU-only stepping
- `movement_speed` — `fast` | `medium` | `slow`; sets `movement_interval` and ecology knobs via `ecology_presets.rs`
- `movement_interval` — explicit override for steps between hex moves (default 1)
- `hex_dish_torus` — when `hex_dish_radius > 0` (headless batch), edge moves wrap through the dish bounding box instead of dead-ending at the rim. Interactive viewer unchanged.

## Divergences removed (legacy tensor/CUDA)

| Bug | Old tensor/CUDA | Correct |
| --- | --- | --- |
| Inputs | 5 | 8 |
| Gather | `magnitude < 0.5` | `move_signal <= 0` |
| Ring move | `round(spend)` multi-cell | ±1 by direction |
| Env depletion | none | subtract `gathered` at `pos` |
| Apply | inside kernel | CPU sequential apply |
