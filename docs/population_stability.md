# Population Stability in Movement Simulations

This document describes what keeps populations stable and evolvable in the current simulator, how the three **movement speed** presets interact with ecology, and how to tune or validate runs. It applies equally to all brain frameworks (Perceptron, MLP, Spatial Graph, GoO) — they share the same population dynamics in `population_dynamics.rs`.

Depletable stock with Perlin regrowth (carrying-capacity maps) is **not** implemented yet; ecology tuning below works with the current model where Perlin resets cell abundance each step and within-step gather depletes locally.

---

## Two layers of population control

### 1. Soft equilibrium (ecological)

These mechanisms produce an emergent carrying capacity without a hard stop:

| Knob | Role |
|------|------|
| **`resource_scale`** | Multiplier on Perlin-derived cell abundance. Primary “how rich is the world?” dial. |
| **`resource_contrast`** | Stretches Perlin samples away from 0.5 before scaling (1 = off). Sharper rich/poor patches. |
| **Local crowding** | `gathered = center / max(local_pop, 1)`. More agents per cell → less food per capita. |
| **`basal_metabolic_cost`** | Flat per-step drain on every survivor. |
| **`maintenance_cost_per_node`** | Extra drain proportional to brain complexity — biases selection against runaway structural growth. |
| **`movement_cost_factor`** | Energy spent when moving; scales with locomotion speed preset. |
| **`reproduction_threshold`** | Minimum stored resources before a birth is allowed. |
| **`reproduction_cost`** | Parent pays this on birth — limits runaway reproduction when food is tight. |
| **`death_risk`** | Small age-dependent hazard; adds generational turnover for evolution. |
| **Starvation** | `resources <= 0` removes individuals immediately after metabolism. |

A **stable** run settles into a band where mean income (gather) ≈ mean expenses (metabolism + movement + reproduction) and births ≈ deaths. Population should **not** sit pinned at `max_population` for long stretches.

### 2. Hard safety cap (`max_population`)

`max_population` stops new births once `individuals.len() >= cap`. It is an OOM guardrail, not the ecological regulator.

**Rule of thumb:** set `max_population` ~35–50% above the expected soft equilibrium (see preset table below). If final population equals `max_population` every run, either raise the cap or tighten ecology (higher metabolism / reproduction threshold / movement cost).

---

## Movement speed presets

Configured via `"movement_speed": "fast" | "medium" | "slow"` in JSON (resolved in `ecology_presets.rs` on load) or `--movement-speed=` on the headless CLI.

| Preset | `movement_interval` | Meaning | `env_time_speed` | Notes |
|--------|---------------------|---------|------------------|-------|
| **fast** | 1 | Up to 1 hex per step when `move_signal > 0` | 0.018 | Landscape shifts quickly relative to travel; higher movement cost. |
| **medium** | 5 | Up to 1 hex per 5 steps | 0.0036 | Default for `pipeline_native` and `pipeline_100k`. Balanced exploration vs. foraging. |
| **slow** | 25 | Up to 1 hex per 25 steps | 0.00072 | Agents rest-forage on cooldown steps; richer `resource_scale` compensates for less exploration. |

While on move cooldown, agents **rest-forage** at their current cell (`center / local_pop`) so they are not penalized with zero income.

Explicit override: `"movement_interval": N` in JSON or `--movement-interval=N`.

---

## Ecology preset values (hex dish, radius 58)

Canonical numbers live in `rust/src/ecology_presets.rs`. Suggested batch-run population sizes:

| Preset | `resource_scale` | `repro_threshold` | `basal` | `maint/node` | `move_cost` | Suggested `initial_pop` | Suggested `max_pop` |
|--------|------------------|-------------------|---------|--------------|-------------|-------------------------|---------------------|
| fast | 0.58 | 40 | 0.095 | 0.042 | 1.05 | 3000 | 10000 |
| medium | 0.64 | 38 | 0.088 | 0.040 | 0.72 | 2500 | 9000 |
| slow | 0.66 | 38 | 0.082 | 0.034 | 0.40 | 2000 | 8000 |

Slower locomotion uses slightly **higher** `resource_scale` and **lower** reproduction thresholds so agents can still accumulate energy despite fewer hex visits per wall-clock step.

`movement_speed` does **not** overwrite `initial_population` in interactive configs — only ecology/metabolism/movement fields are applied. Batch configs set population explicitly.

---

## Architecture-specific ecology offsets

After the movement-speed preset is applied, `apply_architecture_ecology_offset` adjusts metabolism and richness for heavier founder brains:

| Policy | `resource_scale` | `maintenance_cost_per_node` | `reproduction_threshold` |
|--------|------------------|----------------------------|--------------------------|
| Perceptron / MLP | (baseline) | (baseline) | (baseline) |
| Spatial Graph | ×1.20 | ×0.50 | −4 (floor 20) |
| GoO | ×1.45 | ×0.28 | −8 (floor 18) |

Basal metabolic cost is also reduced slightly for spatial / GoO. These offsets keep evolvable architectures viable under the same movement-speed presets without giving them a separate config matrix.

Under **`ecology_profile: "plateau80"`**, spatial / GoO offsets are gentler (smaller `resource_scale` boost, less maintenance discount) so heavy brains still pay meaningful complexity cost and populations stay below `max_population`.

---

## Plateau-80 profile

Set `"ecology_profile": "plateau80"` with a `movement_speed` preset for long runs that target **~70–85% of `max_population`** at equilibrium instead of pinning the cap. Batch configs in `medium_2k/` use `max_population: 2600` (~30% headroom) so a ~1600–1900 soft equilibrium can evolve without sitting pinned at the cap for entire runs.

| Preset | `resource_scale` | `repro_threshold` | `basal` | `maint/node` | `move_cost` | Landscape |
|--------|------------------|-------------------|---------|--------------|-------------|-----------|
| fast | 0.34 | 49 | 0.100 | 0.030 | 1.12 | contrast 1.4, amp 1.45 |
| medium | 0.355 | 48 | 0.097 | 0.028 | 0.90 | contrast 1.4, amp 1.45 |
| slow | 0.38 | 47 | 0.090 | 0.026 | 0.50 | contrast 1.4, amp 1.45 |

Plateau-80 also sets `env_perlin_spatial_scale: 0.038` (slightly larger patches) and `resource_contrast: 1.4`, stretching Perlin samples away from 0.5 so gradient-following and movement matter more.

Suggested `architecture_mutation_rate` for evolvable policies in this profile: **0.01** (≈0.25% structural change per birth at default reproduction rates).

Configs: `rust/configs/ecology_plateau80/` and `medium_2k/` for 2k-cap batch runs.

---

## Config layout

```
rust/configs/ecology/
  perceptron_fast.json … goo_slow.json   # 12 configs: 4 architectures × 3 speeds
rust/configs/pipeline_native/            # medium speed, production snapshots
rust/configs/pipeline_smoke/             # medium speed, small grid, quick CI
rust/configs/pipeline_100k/              # medium speed, long runs
```

Example:

```bash
cargo run --release --bin evo_movement_perceptron -- \
  --config-file=configs/ecology/perceptron_medium.json --no-write-outputs
```

---

## Signs of a well-tuned run

**Healthy (stable + evolvable)**

- Final population between ~40% and ~90% of `max_population`, not equal to the cap.
- Population fluctuates over time rather than monotonic blow-up or extinction.
- Some fraction of individuals move each step (`prop_moved` > 0 in scan output).
- Reproduction occurs regularly but not every individual every step.
- Lineages diverge over thousands of steps (`architecture_mutation_rate` > 0 for evolvable policies).

**Too sparse / dying out**

- Population crashes to near zero within burn-in.
- **Fix:** raise `resource_scale`, lower `reproduction_threshold`, lower metabolism, or lower `movement_cost_factor`.

**Too dense / pinned at cap**

- `final_population == max_population` consistently.
- **Fix:** lower `resource_scale`, raise `reproduction_threshold` or `reproduction_cost`, raise metabolism, or raise `movement_cost_factor`. Increase `max_population` only as a safety margin after ecology is balanced.

**No evolution**

- `architecture_mutation_rate` and `mutation_rate` both zero, or population too small / too short.
- **Fix:** enable mutation, lengthen `steps`, ensure population stays in the hundreds+.

---

## Parameter sweep tooling

`scripts/scan_rust_population_scan.py` sweeps `movement_cost_factor`, `maintenance_cost_per_node`, `reproduction_threshold`, `resource_scale`, `death_risk`, and grid size over short runs. Use it to refine presets for new dish sizes or after enabling depletable-stock ecology.

---

## Architecture-specific notes

All frameworks share ecology. Differences that affect stability indirectly:

- **Perceptron** — lowest complexity; cheapest metabolism baseline.
- **MLP / Spatial / GoO** — `maintenance_cost_per_node × complexity()` grows with structural mutations; complex brains need richer environments or they go extinct under the same global knobs.
- **`max_brain_nodes`** — caps GPU packing and structural mutation extent; does not replace ecological limits.

---

## Future: depletable Perlin stock

Step 1 on the ecology roadmap is to treat Perlin as a **local carrying capacity** with regrowth toward a cap and persistent depletion across steps. When that lands, `resource_scale` will modulate *K* rather than instantaneous abundance, and these presets will need re-validation — the qualitative relationships in this document still apply.
