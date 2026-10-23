# Graph Report - Movement  (2026-10-23)

## Corpus Check
- 242 files · ~133,354 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .wgsl 1, .broken 1)

## Summary
- 1974 nodes · 4648 edges · 121 communities (86 shown, 35 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 166 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `77266ca8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Environment
- WebviewerWriter
- brain_diagram.rs
- run_staged_sweep
- LabStructuralOp
- remote_scan_orchestrator.py
- interactive.rs
- interactive_dash.py
- brain_step_kernel.cu
- visualize_individual.py
- visualize_population.py
- population_dynamics.rs
- .update
- brain_playground.rs
- os
- scan_rust_population_scan.py
- spatial_blob.rs
- Simulation
- MovementSpeed
- json
- compare_cuda_simulations.py
- evo_fitness_landscape.rs
- HexCoord
- SpatialGraphBrain
- worker_pool.py
- MlpBrain
- PopulationTensor
- serve_fitness_landscape.py
- SweepCell
- gpu_pack.rs
- Individual
- BrainGenome
- tensor.rs
- GooGraphBrain
- Interactive Evolution Simulation Viewer
- GridTopology
- Completed Features
- skeleton.rs
- BrainKind
- brain_playground_wasm/src/lib.rs
- snapshot_registry.rs
- probe_timeline.rs
- draw_dish_run_config_panel
- lab_dish.rs
- mutation.rs
- graphs_nn.py
- ArchitecturePolicy
- StructuralMutation
- NeuralNetwork
- run_record.rs
- PetriDish
- Perlin
- Serialize
- genotype_signature
- evo_movement_rust.rs
- MovementOutputMode
- SpatialBlobLayout
- GpuResourceUpdater
- subprocess
- plot_fitness_landscape.R
- config.rs
- copy_tensor_to_nn
- app.js
- import_snapshots.rs
- Remote scan runner (Orange Pi)
- Population Stability in Movement Simulations
- super
- RowRngManager
- viewer_embed.rs
- _set_lines_ydata
- select_and_visualize.py
- step_engine.rs
- test_plot_live_scan_html.py
- play_toggle
- main
- build_info.rs
- Invoke-HeadlessSimWithProgress
- graphs_env.py
- pathlib
- hex_background_color
- save_state.rs
- Hex Coordinates in the Interactive Rust Viewer
- Simulation Step Semantics
- brain_step_kernel
- numpy
- Config
- brain_playground_wasm
- Irreversible Plasticity of Movement: Theoretical Framework & Mechanism Design
- open_sim_dropdown
- 1. Commands by Scenario
- CUDA Kernel Argument Alignment
- GPU Optimization Notes
- individual.rs
- brain/mod.rs
- tensor_kernel.cu
- FounderWeightMode
- RUNNING_SIMULATIONS.md
- rust/README.md

## God Nodes (most connected - your core abstractions)
1. `Config` - 129 edges
2. `Individual` - 76 edges
3. `BrainGenome` - 68 edges
4. `Simulation` - 68 edges
5. `ArchitecturePolicy` - 46 edges
6. `PetriDish` - 37 edges
7. `GooGraphBrain` - 36 edges
8. `PopulationTensor` - 34 edges
9. `NeuralNetwork` - 33 edges
10. `main()` - 32 edges

## Surprising Connections (you probably didn't know these)
- `CUDA Kernel Argument Alignment` --references--> `brain_step_kernel()`  [INFERRED]
  docs/cuda_kernel_arguments.md → rust/cuda/brain_step_kernel.cu
- `Remaining opportunities` --references--> `tensor_step_kernel()`  [INFERRED]
  docs/gpu_optimization.md → rust/cuda/tensor_kernel.cu
- `GPU path` --references--> `DecisionRowGpu`  [INFERRED]
  docs/sim_step_semantics.md → rust/cuda/brain_step_kernel.cu
- `GPU path` --references--> `brain_step_kernel()`  [INFERRED]
  docs/sim_step_semantics.md → rust/cuda/brain_step_kernel.cu
- `structural_mutation_can_grow_mlp()` --calls--> `reproduce_brain()`  [INFERRED]
  rust/src/individual.rs → rust/src/brain/mutation.rs

## Import Cycles
- 1-file cycle: `rust/src/simulation.rs -> rust/src/simulation.rs`
- 1-file cycle: `rust/src/perlin.rs -> rust/src/perlin.rs`
- 2-file cycle: `rust/src/run_record.rs -> rust/src/simulation.rs -> rust/src/run_record.rs`
- 3-file cycle: `rust/src/run_record.rs -> rust/src/simulation.rs -> rust/src/snapshot_registry.rs -> rust/src/run_record.rs`

## Communities (121 total, 35 thin omitted)

### Community 0 - "Environment"
Cohesion: 0.15
Nodes (12): Environment, scaled_resource(), HexDishLayout, edge_resources_hex(), edge_resources_hex_dish(), edge_resources_ring(), flat_to_hex(), hex_neighbor_wraps() (+4 more)

### Community 1 - "WebviewerWriter"
Cohesion: 0.08
Nodes (12): BufferSummary, BufferWriter, BufferWriter<'a>, CompactIndividual, CompactIndividualsFrame, ensure_nn_index_entry(), IndividualsEncoding, Cbor (+4 more)

### Community 2 - "brain_diagram.rs"
Cohesion: 0.09
Nodes (47): bias_to_color(), BrainDiagram, ColorRgb, ColorRgba, diagram_from_brain(), DiagramEdge, DiagramNode, DiagramText (+39 more)

### Community 3 - "run_staged_sweep"
Cohesion: 0.18
Nodes (26): aggregate_cells(), architecture_labels(), average_trajectories(), backfill_extinction_flags(), cell_diagrams_need_probe_backfill(), CellDiagram, CellProbeMeta, combined_latest_runs() (+18 more)

### Community 4 - "LabStructuralOp"
Cohesion: 0.06
Nodes (21): DEFAULT_DISH_RADIUS, DEFAULT_HEX_SIZE, hexagon_points(), lab_default_config(), LabApp, main(), LabStructuralOp, .ALL (+13 more)

### Community 5 - "remote_scan_orchestrator.py"
Cohesion: 0.11
Nodes (32): _atomic_write_json(), _bootstrap_repo_if_missing(), _db_mark_done(), _db_mark_started(), _db_upsert_job(), _deploy_remote_helper(), _ensure_remote_dir(), _init_db() (+24 more)

### Community 6 - "interactive.rs"
Cohesion: 0.08
Nodes (28): AppOptions, collect_snapshots_from_dir(), compute_headless_genotype_stats(), compute_session_id(), DEFAULT_DISH_RADIUS_CELLS, ExportedIndividualRecord, genotype_color(), genotype_signature_from_snapshot() (+20 more)

### Community 7 - "interactive_dash.py"
Cohesion: 0.08
Nodes (16): build_app(), recompute_base(), sync_nn(), update_env_plot(), update_k(), update_nn_plot(), carrying_capacity(), cauchy_mutation() (+8 more)

### Community 8 - "brain_step_kernel.cu"
Cohesion: 0.21
Nodes (15): build_hex_edges(), build_perceptron_sensors(), decode_outputs(), direction_signal_to_degrees(), flat_to_hex(), forward_goo(), forward_mlp(), forward_perceptron() (+7 more)

### Community 9 - "visualize_individual.py"
Cohesion: 0.11
Nodes (14): anim_step(), draw_env_with_density(), on_next(), on_prev(), on_rand(), on_slider(), on_text_submit(), _safe_set_label() (+6 more)

### Community 10 - "visualize_population.py"
Cohesion: 0.12
Nodes (18): discover_sims(), find_and_load_initial(), find_index_by_id(), _load_individual_frames(), _load_individuals_delta(), load_sim(), next_sim(), on_click() (+10 more)

### Community 11 - "population_dynamics.rs"
Cohesion: 0.24
Nodes (11): grid_cell_count(), apply_metabolism_and_cull(), BirthRecord, pop_cap_blocks_births(), pop_cap_reached(), record_births_in_cladogram(), record_births_in_lineage(), run_reproduction_phase() (+3 more)

### Community 12 - ".update"
Cohesion: 0.13
Nodes (6): copy_dynamics_config(), EvolutionApp, export_individual_brain_json(), hexagon_points(), merge_run_config_from_snapshot(), open_session_folder()

### Community 13 - "brain_playground.rs"
Cohesion: 0.14
Nodes (27): goo_add_node_json(), goo_cycle_op_json(), random_goo_json(), random_mlp_json(), random_perceptron_json(), random_spatial_json(), brain_to_json(), build_probe_sensors() (+19 more)

### Community 14 - "os"
Cohesion: 0.09
Nodes (5): main(), _parse_last_summary_json(), iter_env_frames(), iter_individuals_frames(), fail()

### Community 15 - "scan_rust_population_scan.py"
Cohesion: 0.10
Nodes (15): cmd_builder(), deep_result_builder(), find_sim_folder_for_prefix(), _make_param_tuple(), make_progress_writer(), atomic_write(), writer(), _parse_float_list() (+7 more)

### Community 16 - "spatial_blob.rs"
Cohesion: 0.13
Nodes (13): blob_forward_runs(), cardinal_pool_inputs(), cardinal_pools_match_compass_recipe(), Compass8, hex_edges_from_sensors(), hex_to_compass8(), SECTOR_E, SECTOR_NE (+5 more)

### Community 17 - "Simulation"
Cohesion: 0.08
Nodes (8): ArchitectureLineageEvent, write_lineage_jsonl(), clamp_initial_population(), progress_update_interval(), should_emit_progress(), Simulation, test_common_garden_assay_evaluates_motility(), test_morph_threshold_stats_calculation()

### Community 18 - "MovementSpeed"
Cohesion: 0.15
Nodes (15): MovementSpeed, Fast, Medium, Slow, apply_architecture_ecology_offset(), apply_ecology_tuning(), apply_movement_speed(), apply_plateau80_landscape() (+7 more)

### Community 19 - "json"
Cohesion: 0.11
Nodes (18): classify_trajectory(), load_events(), main(), main(), read_final_population(), main(), build_bundle(), _compress_frame() (+10 more)

### Community 20 - "compare_cuda_simulations.py"
Cohesion: 0.32
Nodes (7): build_map(), hash_bytes(), build_map(), diff_outputs(), hash_bytes(), main(), run_simulation()

### Community 21 - "evo_fitness_landscape.rs"
Cohesion: 0.15
Nodes (26): build_cell_diagrams(), cell_snapshot_path(), DEFAULT_SWEEP_STAGES, fitness_score(), is_spatial_architecture(), mean_complexity(), mean_resources(), parse_arg_value() (+18 more)

### Community 22 - "HexCoord"
Cohesion: 0.16
Nodes (10): max_cell_occupancy(), HexCell, HexCoord, HexGrid, test_hex_coord_from_flat(), test_hex_coord_neighbors(), test_hex_coord_to_flat(), test_hex_dish_cell_count() (+2 more)

### Community 23 - "SpatialGraphBrain"
Cohesion: 0.17
Nodes (6): all_sectors_mask(), flip_random_sector(), random_sector_mask(), SPATIAL_SECTOR_COUNT, SpatialEdge, SpatialGraphBrain

### Community 24 - "worker_pool.py"
Cohesion: 0.09
Nodes (10): build_index(), load_promising(), load_results(), main(), _parse_extinction(), run_param_sweep(), worker(), start_static_server() (+2 more)

### Community 27 - "serve_fitness_landscape.py"
Cohesion: 0.20
Nodes (9): main(), make_handler(), do_GET(), do_POST(), end_headers(), __init__(), _json(), log_message() (+1 more)

### Community 28 - "SweepCell"
Cohesion: 0.29
Nodes (16): build_cells_to_sweep(), build_goo_cells(), build_spatial_cells(), cells_to_grid_specs(), evolution_hunt_grand_total(), evolution_hunt_scope_is_108_cells_24_heatmaps(), expand_output_mode_cells(), load_named() (+8 more)

### Community 29 - "gpu_pack.rs"
Cohesion: 0.12
Nodes (16): DecisionRow, DecisionRowGpu, GOO_OP_AGGREGATE, GOO_OP_GENERATE, goo_op_id(), GOO_OP_KEEP_BEST, GOO_OP_SCORE, individual_fits_gpu() (+8 more)

### Community 30 - "Individual"
Cohesion: 0.17
Nodes (8): Individual, .MAINTENANCE_COST_PER_NODE, .MOVEMENT_COST_FACTOR, .MUTATION_RATE, .MUTATION_SIGMA, .REPRODUCTION_COST, .REPRODUCTION_THRESHOLD, ReproductionResult

### Community 32 - "tensor.rs"
Cohesion: 0.10
Nodes (17): cached_cuda_module(), CachedCudaRuntime, DecisionRowGpu, device_ptr_or_null(), GpuForwardState, cuda_debug!, make_env_view(), run_brain_forward() (+9 more)

### Community 33 - "GooGraphBrain"
Cohesion: 0.19
Nodes (3): GooGraphBrain, GooNode, grow_to_size_can_produce_mixed_ops()

### Community 34 - "Interactive Evolution Simulation Viewer"
Cohesion: 0.06
Nodes (31): Cleaning webviewer output, Command-Line Simulation, evo-movement-sim, Interactive Viewer (New!), Remote scan runner (Orange Pi), Running with burn-in and steps, Architecture, Build Instructions (+23 more)

### Community 35 - "GridTopology"
Cohesion: 0.30
Nodes (13): GridTopology, Hex2d, Ring1d, assert_forward_parity(), assert_sim_parity(), parity_config(), parity_goo(), parity_hex_mlp() (+5 more)

### Community 36 - "Completed Features"
Cohesion: 0.09
Nodes (22): 1. Hexagonal Grid System ✅, 2. Color Inheritance System ✅, 3. Cladogram/Phylogenetic Tree ✅, 4. Neural Network Visualization ✅, 5. Interactive GUI Application ✅, 6. Simulation Controls ✅, 7. Save/Load System ✅, 8. Environment Integration ✅ (+14 more)

### Community 37 - "skeleton.rs"
Cohesion: 0.26
Nodes (17): heatmaps_for_scope(), is_promising_cell(), select_promoted_cells(), axis_labels_sort_numeric_suffix(), build_expected_heatmaps(), build_heatmaps_for_live(), build_heatmaps_from_cells(), CellAggregate (+9 more)

### Community 38 - "BrainKind"
Cohesion: 0.12
Nodes (9): BrainKind, GooGraph, Mlp, Perceptron, SpatialBlob, SpatialGraph, brain_kind_id(), SNAPSHOT_VERSION (+1 more)

### Community 39 - "brain_playground_wasm/src/lib.rs"
Cohesion: 0.10
Nodes (6): demo_goo_branching_json(), diagram_json(), validate_brain_json(), wrap_json(), brain_to_diagram_json(), demo_goo_branching_brain()

### Community 40 - "snapshot_registry.rs"
Cohesion: 0.18
Nodes (17): RunRecordContext, collect_snapshots_from_dir(), find_repo_root(), import_existing_snapshot(), is_source_registered(), list_registry_entries(), list_registry_snapshot_paths(), manifest_label() (+9 more)

### Community 41 - "probe_timeline.rs"
Cohesion: 0.05
Nodes (57): main(), write_empty_tensor_kernel_path(), config_path(), dedupe_runs(), load_config(), load_config_named(), load_runs_from_state(), main() (+49 more)

### Community 42 - "draw_dish_run_config_panel"
Cohesion: 0.29
Nodes (6): dish_info_line(), dish_run_config_candidates(), dish_run_config_display(), draw_brain_json_export_ui(), draw_dish_run_config_panel(), reveal_in_file_manager()

### Community 43 - "lab_dish.rs"
Cohesion: 0.17
Nodes (7): genotype_color(), genotype_signature(), GOLDEN_ANGLE, hsv_to_rgb(), LabDish, resource_background_color(), tick_move_wait()

### Community 44 - "mutation.rs"
Cohesion: 0.25
Nodes (10): apply_goo_structural(), apply_lab_structural(), apply_mlp_structural(), apply_spatial_structural(), init_founder_weights(), init_founder_weights_with_mode(), mutate_parametric(), perceptron_to_mlp() (+2 more)

### Community 45 - "graphs_nn.py"
Cohesion: 0.14
Nodes (5): compute_output_vector(), draw_info(), draw_nn_diagram(), draw_nn_graph(), forward_single_with_nn()

### Community 46 - "ArchitecturePolicy"
Cohesion: 0.10
Nodes (13): ArchitecturePolicy, EvolveGoo, EvolveMlp, EvolveSpatialGraph, PerceptronOnly, parse_architecture_policy(), parse_arg_value(), parse_grid_topology() (+5 more)

### Community 47 - "StructuralMutation"
Cohesion: 0.11
Nodes (18): StructuralMutation, GooAddAggregate, GooAddGenerate, GooAddKeepBest, GooAddScore, GooChangeOp, KindChange, MlpAddLayer (+10 more)

### Community 48 - "NeuralNetwork"
Cohesion: 0.09
Nodes (12): Cladogram, LineageNode, test_cladogram_basic(), test_genetic_distance(), test_lineage_path_and_descendants(), test_mark_death_filters_living_descendants(), forward_multi_simple(), forward_single_buf_matches() (+4 more)

### Community 49 - "run_record.rs"
Cohesion: 0.26
Nodes (16): append_global_history(), BinaryInfo, build_sim_params(), build_worker_info(), current_binary_name(), persist_run_record(), policy_label(), repo_output_dir() (+8 more)

### Community 50 - "PetriDish"
Cohesion: 0.23
Nodes (5): directional_movement_maps_to_expected_neighbor_sectors(), DishCheckpoint, make_direction_test_dish(), movement_is_suppressed_when_below_threshold(), PetriDish

### Community 51 - "Perlin"
Cohesion: 0.22
Nodes (5): Perlin, resource_contrast_spreads_around_midpoint(), sample_2d_time_is_deterministic(), sample_2d_time_varies_with_time(), sample_2d_time_varies_with_y_dimension()

### Community 52 - "Serialize"
Cohesion: 0.16
Nodes (10): MlpBrainSerde, hidden_sum_output(), split_channels_output(), BrainOutput, EMBED_DIM, l2_slices(), PERCEPTRON_SENSOR_DIM, SENSOR_DIM (+2 more)

### Community 53 - "genotype_signature"
Cohesion: 0.40
Nodes (3): assign_genotype_colors(), genotype_signature(), GenotypeLineageRow

### Community 54 - "evo_movement_rust.rs"
Cohesion: 0.38
Nodes (4): main(), parse_arg_value(), print_help(), sanitize_basename()

### Community 55 - "MovementOutputMode"
Cohesion: 0.11
Nodes (20): CARDINAL_ANGLES_DEG, COMPASS_EPS, default_movement_output_mode(), direction_roundtrip_near_equator(), direction_signal_from_degrees(), hex_compass_cardinal_uniform_is_flat(), hex_compass_favors_east_activation(), hex_compass_magnitude() (+12 more)

### Community 56 - "SpatialBlobLayout"
Cohesion: 0.40
Nodes (3): SpatialBlobLayout, Cardinal4, HexEdge

### Community 58 - "subprocess"
Cohesion: 0.15
Nodes (14): build_cmd(), main(), run_one(), main(), run_single_job(), main(), run_single_job(), main() (+6 more)

### Community 59 - "plot_fitness_landscape.R"
Cohesion: 0.15
Nodes (10): args, cells, data, json_path, out_dir, p_heat, p_time, runs (+2 more)

### Community 60 - "config.rs"
Cohesion: 0.22
Nodes (11): default_allow_concurrent_eat_and_move(), default_developmental_steps(), default_dispersal_on_low_cue(), default_flagella_maintenance_cost(), default_morph_threshold(), default_movement_cost_factor(), default_run_duration(), default_spatial_blob_layout() (+3 more)

### Community 62 - "app.js"
Cohesion: 0.29
Nodes (11): canvas, clamp(), ctx, drawFrame(), fpsInput, frameCount(), frameInput, loadBundle() (+3 more)

### Community 63 - "import_snapshots.rs"
Cohesion: 0.35
Nodes (10): collect_legacy_snapshots(), find_repo_root(), import_one(), load_run_record(), main(), read_text_file(), resolve_config_path(), RunRecordImport (+2 more)

### Community 64 - "Remote scan runner (Orange Pi)"
Cohesion: 0.12
Nodes (15): Bootstrapping the repo, Burnin evaluation (movement filter), Convenience executable (Windows), Files, Notes, Orange Pi vs local machine, Outputs, Parameters (+7 more)

### Community 65 - "Population Stability in Movement Simulations"
Cohesion: 0.14
Nodes (13): 1. Soft equilibrium (ecological), 2. Hard safety cap (`max_population`), Architecture-specific ecology offsets, Architecture-specific notes, Config layout, Ecology preset values (hex dish, radius 58), Future: depletable Perlin stock, Movement speed presets (+5 more)

### Community 66 - "super"
Cohesion: 0.29
Nodes (5): dish_cell_count_formula(), open_dish_edge_neighbor_is_self(), resolve_neighbor_index(), torus_dish_wraps_across_bounding_box(), wrap_coord_in_box()

### Community 68 - "viewer_embed.rs"
Cohesion: 0.17
Nodes (16): copy_probe_wasm_bundle(), flush_live_snapshot(), is_file_lock_error(), load_viewer_template(), now_ms(), viewer_json_payload(), write_html(), write_landscape_outputs() (+8 more)

### Community 70 - "select_and_visualize.py"
Cohesion: 0.44
Nodes (7): choose(), discover_runs(), discover_snapshots(), launch_viewers(), main(), read_build_id(), run_replay()

### Community 71 - "step_engine.rs"
Cohesion: 0.17
Nodes (21): direction_to_sector(), neighbor_flat_dish(), ActionDecision, ActionDecision, apply_decision(), apply_gather_at(), build_perceptron_sensor_inputs(), build_sensor_inputs() (+13 more)

### Community 73 - "play_toggle"
Cohesion: 0.67
Nodes (4): play_toggle(), start_timer(), stop_timer(), timer_step()

### Community 74 - "main"
Cohesion: 0.15
Nodes (18): backfill_probe_diagrams(), cell_matches_arch_filter(), config_path(), load_config(), load_config_named(), main(), parse_architectures(), parse_sweep_profile() (+10 more)

### Community 75 - "build_info.rs"
Cohesion: 0.40
Nodes (4): BUILD_ID, CRATE_VERSION, CUDA_ARCH, VERSION_LABEL

### Community 77 - "graphs_env.py"
Cohesion: 0.29
Nodes (3): draw_env_line(), setup_env_axis(), update_env()

### Community 78 - "pathlib"
Cohesion: 0.12
Nodes (4): main(), tree_balance_factor(), copytree(), main()

### Community 79 - "hex_background_color"
Cohesion: 0.40
Nodes (5): competition_heatmap_color(), hex_background_color(), HexBackgroundMode, Competition, Resources

### Community 80 - "save_state.rs"
Cohesion: 0.36
Nodes (4): CladogramNodeState, HexCellState, IndividualState, SimulationState

### Community 81 - "Hex Coordinates in the Interactive Rust Viewer"
Cohesion: 0.15
Nodes (12): Cladogram and spatial independence, Coordinate system, Core types, Dish shape and active cells, Geometry and rendering, Hex Coordinates in the Interactive Rust Viewer, HexCell, HexCoord (+4 more)

### Community 82 - "Simulation Step Semantics"
Cohesion: 0.17
Nodes (11): Config, Divergences removed (legacy tensor/CUDA), Other policies (8 dimensions), Perceptron (3 dimensions), Phase 1 — Snapshot occupancy, Phase 2 — Sensors, Phase 3 — Brain forward, Phase 4 — Decision row (+3 more)

### Community 83 - "brain_step_kernel"
Cohesion: 0.20
Nodes (12): GPU path, brain_step_kernel(), build_dish_edges(), build_ring_edges(), build_sensors(), decision_from_output(), DecisionRowGpu, direction_degrees (+4 more)

### Community 84 - "numpy"
Cohesion: 0.10
Nodes (14): main(), main(), main(), main(), main(), main(), compute_nn_response_for(), init_response() (+6 more)

### Community 85 - "Config"
Cohesion: 0.24
Nodes (3): Config, move_allowed(), start_move_cooldown()

### Community 89 - "Irreversible Plasticity of Movement: Theoretical Framework & Mechanism Design"
Cohesion: 0.20
Nodes (9): 1. Research Motivation & Core Premise, 2. The Ecological Dilemma: Swimming vs. Settling, 3. Sensory Cue Specification, 4. Strictly Irreversible Mechanism (Phase 1 Specification), 5. Evolutionary Predictions on the Botero Landscape, 6. Future Extension: Continuous Evolvable Reversibility, Irreversible Plasticity of Movement: Theoretical Framework & Mechanism Design, Mathematical Formulation: (+1 more)

### Community 109 - "1. Commands by Scenario"
Cohesion: 0.22
Nodes (8): 1. Commands by Scenario, 2. Checklist: Regenerating & Syncing Obsidian Notes, 3. Privacy & Safety Guarantee, After Dependency or Architecture Changes, After Meaningful Code Changes, After Pulling Git Changes, Graphify Workflow & Maintenance Guide, Querying the Code Graph

### Community 111 - "CUDA Kernel Argument Alignment"
Cohesion: 0.29
Nodes (6): Argument table, CUDA Kernel Argument Alignment, DecisionRowGpu layout (`repr(C)`), Launch configuration, Parity verification, Persistent device buffers

### Community 112 - "GPU Optimization Notes"
Cohesion: 0.29
Nodes (6): Benchmarking, CCCL evaluation (spike plan), Changes in this branch, Current state (branch `feature/evolvable-graph-networks`), GPU Optimization Notes, Remaining opportunities

### Community 113 - "individual.rs"
Cohesion: 0.28
Nodes (5): ForwardContext, decide_action_gathers_when_move_signal_nonpositive(), founder_perceptron(), make_buffers(), structural_mutation_can_grow_mlp()

### Community 114 - "brain/mod.rs"
Cohesion: 0.48
Nodes (6): BrainInner, GooGraph, Mlp, Perceptron, SpatialBlob, SpatialGraph

### Community 115 - "tensor_kernel.cu"
Cohesion: 0.47
Nodes (3): nn_activate(), signum(), tensor_step_kernel()

### Community 117 - "FounderWeightMode"
Cohesion: 0.29
Nodes (5): push_run(), FounderWeightMode, Neutral, RandomSmall, RandomUniform

### Community 118 - "RUNNING_SIMULATIONS.md"
Cohesion: 0.50
Nodes (3): 1. Binaries and entry points, Running Rust Simulations and Outputs, Windows only

## Knowledge Gaps
- **83 isolated node(s):** `evo_movement_rust`, `brain_playground_wasm`, `move_signal`, `direction_degrees`, `movement_spend` (+78 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 622 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `Environment`, `WebviewerWriter`, `LabStructuralOp`, `interactive.rs`, `population_dynamics.rs`, `.update`, `brain_playground.rs`, `Simulation`, `MovementSpeed`, `evo_fitness_landscape.rs`, `PopulationTensor`, `SweepCell`, `gpu_pack.rs`, `Individual`, `BrainGenome`, `tensor.rs`, `GridTopology`, `snapshot_registry.rs`, `probe_timeline.rs`, `lab_dish.rs`, `mutation.rs`, `ArchitecturePolicy`, `run_record.rs`, `PetriDish`, `evo_movement_rust.rs`, `MovementOutputMode`, `config.rs`, `import_snapshots.rs`, `RowRngManager`, `step_engine.rs`, `main`, `save_state.rs`, `individual.rs`, `brain/mod.rs`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **What connects `evo_movement_rust`, `brain_playground_wasm`, `move_signal` to the rest of the system?**
  _83 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `WebviewerWriter` be split into smaller, more focused modules?**
  _Cohesion score 0.07712765957446809 - nodes in this community are weakly interconnected._
- **Why does `BrainGenome` connect `BrainGenome` to `brain_diagram.rs`, `interactive.rs`, `BrainKind`, `brain_playground_wasm/src/lib.rs`, `probe_timeline.rs`, `step_engine.rs`, `population_dynamics.rs`, `mutation.rs`, `brain_playground.rs`, `NeuralNetwork`, `individual.rs`, `brain/mod.rs`, `evo_fitness_landscape.rs`, `MlpBrain`, `gpu_pack.rs`, `Individual`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Should `brain_diagram.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.09465891062929667 - nodes in this community are weakly interconnected._
- **Why does `Individual` connect `Individual` to `WebviewerWriter`, `LabStructuralOp`, `interactive.rs`, `population_dynamics.rs`, `.update`, `Simulation`, `PopulationTensor`, `gpu_pack.rs`, `BrainGenome`, `tensor.rs`, `BrainKind`, `probe_timeline.rs`, `lab_dish.rs`, `NeuralNetwork`, `PetriDish`, `genotype_signature`, `copy_tensor_to_nn`, `RowRngManager`, `step_engine.rs`, `save_state.rs`, `Config`, `individual.rs`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Should `LabStructuralOp` be split into smaller, more focused modules?**
  _Cohesion score 0.06280193236714976 - nodes in this community are weakly interconnected._