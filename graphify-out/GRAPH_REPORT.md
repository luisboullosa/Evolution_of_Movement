# Graph Report - Movement  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1826 nodes · 4319 edges · 106 communities (73 shown, 33 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 155 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2b908e25`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 85

## God Nodes (most connected - your core abstractions)
1. `BrainGenome` - 68 edges
2. `ArchitecturePolicy` - 46 edges
3. `Simulation` - 38 edges
4. `PetriDish` - 37 edges
5. `GooGraphBrain` - 36 edges
6. `PopulationTensor` - 34 edges
7. `MovementOutputMode` - 32 edges
8. `main()` - 32 edges
9. `SpatialGraphBrain` - 31 edges
10. `HexCoord` - 31 edges

## Surprising Connections (you probably didn't know these)
- `structural_mutation_can_grow_mlp()` --calls--> `reproduce_brain()`  [INFERRED]
  rust/src/individual.rs → rust/src/brain/mutation.rs
- `ExportedIndividualRecord` --references--> `BrainGenome`  [EXTRACTED]
  rust/src/bin/interactive.rs → rust/src/brain/genome.rs
- `landscape_output_from_runs()` --calls--> `build_heatmaps_for_live()`  [INFERRED]
  rust/src/bin/evo_fitness_landscape.rs → rust/src/fitness_landscape/skeleton.rs
- `main()` --calls--> `build_heatmaps_from_cells()`  [INFERRED]
  rust/src/bin/evo_fitness_landscape.rs → rust/src/fitness_landscape/skeleton.rs
- `main()` --calls--> `architecture_label()`  [INFERRED]
  rust/src/bin/evo_fitness_landscape.rs → rust/src/fitness_landscape/sweep_grid.rs

## Import Cycles
- 1-file cycle: `rust/src/simulation.rs -> rust/src/simulation.rs`

## Communities (106 total, 33 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.10
Nodes (53): aggregate_cells(), architecture_labels(), average_trajectories(), backfill_extinction_flags(), cell_diagrams_need_probe_backfill(), cell_matches_arch_filter(), CellDiagram, CellProbeMeta (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (45): bias_to_color(), BrainDiagram, ColorRgb, ColorRgba, diagram_from_brain(), DiagramEdge, DiagramNode, DiagramText (+37 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (22): DEFAULT_DISH_RADIUS, DEFAULT_HEX_SIZE, hexagon_points(), lab_default_config(), LabApp, main(), LabStructuralOp, .ALL (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (12): BufferSummary, BufferWriter, BufferWriter<'a>, CompactIndividual, CompactIndividualsFrame, ensure_nn_index_entry(), IndividualsEncoding, Cbor (+4 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (32): _atomic_write_json(), _bootstrap_repo_if_missing(), _db_mark_done(), _db_mark_started(), _db_upsert_job(), _deploy_remote_helper(), _ensure_remote_dir(), _init_db() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (20): Config, default_spatial_blob_layout(), EcologyProfile, Plateau80, Standard, MovementSpeed, Fast, Medium (+12 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (38): config_path(), dedupe_runs(), load_config(), load_config_named(), load_runs_from_state(), main(), parse_arg_value(), parse_stages() (+30 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (19): sample_individual(), copy_weights(), PopulationSnapshot, SNAPSHOT_VERSION, SNAPSHOT_VERSION_V1, SnapshotError, Codec, Io (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (29): AppOptions, assign_genotype_colors(), collect_snapshots_from_dir(), compute_headless_genotype_stats(), compute_session_id(), DEFAULT_DISH_RADIUS_CELLS, export_individual_brain_json(), ExportedIndividualRecord (+21 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (16): build_app(), recompute_base(), sync_nn(), update_env_plot(), update_k(), update_nn_plot(), carrying_capacity(), cauchy_mutation() (+8 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (24): CARDINAL_ANGLES_DEG, COMPASS_EPS, default_movement_output_mode(), direction_roundtrip_near_equator(), direction_signal_from_degrees(), hex_compass_cardinal_uniform_is_flat(), hex_compass_favors_east_activation(), hex_compass_magnitude() (+16 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (12): decide_action_gathers_when_move_signal_nonpositive(), founder_perceptron(), Individual, .MAINTENANCE_COST_PER_NODE, .MOVEMENT_COST_FACTOR, .MUTATION_RATE, .MUTATION_SIGMA, .REPRODUCTION_COST (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (29): brain_step_kernel(), build_dish_edges(), build_hex_edges(), build_perceptron_sensors(), build_ring_edges(), build_sensors(), decision_from_output(), DecisionRowGpu (+21 more)

### Community 13 - "Community 13"
Cohesion: 0.10
Nodes (16): blob_forward_runs(), cardinal_pool_inputs(), cardinal_pools_match_compass_recipe(), Compass8, hex_edges_from_sensors(), hex_to_compass8(), SECTOR_E, SECTOR_NE (+8 more)

### Community 14 - "Community 14"
Cohesion: 0.10
Nodes (15): anim_step(), draw_env_with_density(), on_next(), on_prev(), on_rand(), on_slider(), on_text_submit(), _safe_set_label() (+7 more)

### Community 15 - "Community 15"
Cohesion: 0.10
Nodes (24): discover_sims(), find_and_load_initial(), find_index_by_id(), _load_individual_frames(), _load_individuals_delta(), load_sim(), next_sim(), on_click() (+16 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (12): cached_cuda_module(), CachedCudaRuntime, DecisionRowGpu, device_ptr_or_null(), cuda_debug!, sync_device_buffer(), sync_device_buffer_opt(), tensor_kernel_ptx_path() (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.16
Nodes (12): BrainGenome, param_distance(), apply_goo_structural(), apply_lab_structural(), apply_mlp_structural(), apply_spatial_structural(), init_founder_weights(), init_founder_weights_with_mode() (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.09
Nodes (5): main(), _parse_last_summary_json(), iter_env_frames(), iter_individuals_frames(), fail()

### Community 19 - "Community 19"
Cohesion: 0.09
Nodes (15): cmd_builder(), deep_result_builder(), find_sim_folder_for_prefix(), _make_param_tuple(), make_progress_writer(), atomic_write(), writer(), _parse_float_list() (+7 more)

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (24): demo_goo_branching_json(), random_goo_json(), random_mlp_json(), random_perceptron_json(), random_spatial_json(), brain_to_json(), build_probe_sensors(), demo_goo_branching_brain() (+16 more)

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (26): backfill_probe_diagrams(), build_cell_diagrams(), cell_snapshot_path(), config_path(), fitness_score(), load_config(), load_config_named(), mean_complexity() (+18 more)

### Community 22 - "Community 22"
Cohesion: 0.19
Nodes (20): ActionDecision, ActionDecision, apply_decision(), apply_decisions_sequential(), apply_gather_at(), build_perceptron_sensor_inputs(), build_sensor_inputs(), build_sensor_inputs_for_kind() (+12 more)

### Community 23 - "Community 23"
Cohesion: 0.15
Nodes (4): EvolutionApp, GenotypeLineageRow, hexagon_points(), open_session_folder()

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (8): all_sectors_mask(), flip_random_sector(), random_sector_mask(), SPATIAL_SECTOR_COUNT, SpatialEdge, SpatialGraphBrain, EMBED_DIM, SpatialRuntime

### Community 25 - "Community 25"
Cohesion: 0.15
Nodes (12): classify_trajectory(), load_events(), main(), main(), read_final_population(), main(), tree_balance_factor(), main() (+4 more)

### Community 27 - "Community 27"
Cohesion: 0.16
Nodes (10): max_cell_occupancy(), HexCell, HexCoord, HexGrid, test_hex_coord_from_flat(), test_hex_coord_neighbors(), test_hex_coord_to_flat(), test_hex_dish_cell_count() (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.11
Nodes (9): diagram_json(), goo_add_node_json(), goo_cycle_op_json(), validate_brain_json(), wrap_json(), brain_to_diagram_json(), goo_add_structural_node(), goo_cycle_node_op() (+1 more)

### Community 29 - "Community 29"
Cohesion: 0.12
Nodes (7): build_index(), load_promising(), load_results(), main(), _parse_extinction(), run_param_sweep(), worker()

### Community 31 - "Community 31"
Cohesion: 0.21
Nodes (19): parse_sweep_profile(), replicate_seed(), build_cells_to_sweep(), build_goo_cells(), build_spatial_cells(), cells_to_grid_specs(), evolution_hunt_grand_total(), evolution_hunt_scope_is_108_cells_24_heatmaps() (+11 more)

### Community 33 - "Community 33"
Cohesion: 0.12
Nodes (9): main(), make_handler(), do_GET(), do_POST(), end_headers(), __init__(), _json(), log_message() (+1 more)

### Community 34 - "Community 34"
Cohesion: 0.12
Nodes (6): main(), write_empty_tensor_kernel_path(), main(), parse_arg_value(), print_help(), sanitize_basename()

### Community 35 - "Community 35"
Cohesion: 0.16
Nodes (6): Cladogram, LineageNode, test_cladogram_basic(), test_genetic_distance(), test_lineage_path_and_descendants(), test_mark_death_filters_living_descendants()

### Community 36 - "Community 36"
Cohesion: 0.27
Nodes (18): heatmaps_for_scope(), is_promising_cell(), select_promoted_cells(), axis_labels_sort_numeric_suffix(), build_expected_heatmaps(), build_heatmaps_for_live(), build_heatmaps_from_cells(), CellAggregate (+10 more)

### Community 37 - "Community 37"
Cohesion: 0.17
Nodes (11): Environment, scaled_resource(), edge_resources_hex(), edge_resources_hex_dish(), edge_resources_ring(), flat_to_hex(), hex_neighbor_wraps(), hex_to_flat() (+3 more)

### Community 38 - "Community 38"
Cohesion: 0.18
Nodes (16): RunRecordContext, collect_snapshots_from_dir(), find_repo_root(), import_existing_snapshot(), is_source_registered(), list_registry_entries(), list_registry_snapshot_paths(), manifest_label() (+8 more)

### Community 39 - "Community 39"
Cohesion: 0.19
Nodes (3): GooGraphBrain, GooNode, grow_to_size_can_produce_mixed_ops()

### Community 40 - "Community 40"
Cohesion: 0.17
Nodes (5): genotype_color(), genotype_signature(), GOLDEN_ANGLE, hsv_to_rgb(), LabDish

### Community 41 - "Community 41"
Cohesion: 0.13
Nodes (6): compute_nn_response_for(), compute_output_vector(), draw_info(), draw_nn_diagram(), draw_nn_graph(), forward_single_with_nn()

### Community 42 - "Community 42"
Cohesion: 0.11
Nodes (6): ArchitecturePolicy, EvolveGoo, EvolveMlp, EvolveSpatialGraph, PerceptronOnly, policy_slug()

### Community 43 - "Community 43"
Cohesion: 0.20
Nodes (6): directional_movement_maps_to_expected_neighbor_sectors(), DishCheckpoint, make_direction_test_dish(), movement_is_suppressed_when_below_threshold(), MoveTrail, PetriDish

### Community 44 - "Community 44"
Cohesion: 0.11
Nodes (18): StructuralMutation, GooAddAggregate, GooAddGenerate, GooAddKeepBest, GooAddScore, GooChangeOp, KindChange, MlpAddLayer (+10 more)

### Community 45 - "Community 45"
Cohesion: 0.24
Nodes (14): GridTopology, Hex2d, Ring1d, topology_label(), assert_forward_parity(), assert_sim_parity(), parity_config(), parity_goo() (+6 more)

### Community 46 - "Community 46"
Cohesion: 0.20
Nodes (6): forward_multi_simple(), forward_single_buf_matches(), NeuralNetwork, NNType, MultiInput, SingleInput

### Community 47 - "Community 47"
Cohesion: 0.27
Nodes (15): append_global_history(), BinaryInfo, build_sim_params(), build_worker_info(), current_binary_name(), persist_run_record(), policy_label(), repo_output_dir() (+7 more)

### Community 48 - "Community 48"
Cohesion: 0.24
Nodes (6): MlpBrainSerde, l2_slices(), PERCEPTRON_SENSOR_DIM, SENSOR_DIM, THOUGHT_DIM, ExampleConfig

### Community 49 - "Community 49"
Cohesion: 0.20
Nodes (5): Perlin, resource_contrast_spreads_around_midpoint(), sample_2d_time_is_deterministic(), sample_2d_time_varies_with_time(), sample_2d_time_varies_with_y_dimension()

### Community 50 - "Community 50"
Cohesion: 0.23
Nodes (7): brain_kind_id(), individual_fits_gpu(), pack_perceptron(), pack_perceptron_brain_kind(), pack_row(), pack_spatial(), PopulationBrainTensor

### Community 51 - "Community 51"
Cohesion: 0.19
Nodes (12): BUILD_ID, CRATE_VERSION, CUDA_ARCH, VERSION_LABEL, parse_architecture_policy(), parse_arg_value(), parse_grid_topology(), parse_movement_output_mode() (+4 more)

### Community 52 - "Community 52"
Cohesion: 0.27
Nodes (8): build_bundle(), _compress_frame(), _infer_grid_width(), main(), parse_args(), _quantize_resources(), _read_jsonl(), _select_evenly_spaced_indices()

### Community 53 - "Community 53"
Cohesion: 0.22
Nodes (7): init_response(), prepare_fitness_data(), setup_fitness_axis(), update_fitness_axis(), update_response_axis(), test_multi_input_vectorized(), test_single_input_vectorized()

### Community 54 - "Community 54"
Cohesion: 0.18
Nodes (3): ArchitectureLineageEvent, write_lineage_jsonl(), progress_update_interval()

### Community 56 - "Community 56"
Cohesion: 0.31
Nodes (7): build_map(), hash_bytes(), build_map(), diff_outputs(), hash_bytes(), main(), run_simulation()

### Community 57 - "Community 57"
Cohesion: 0.15
Nodes (10): args, cells, data, json_path, out_dir, p_heat, p_time, runs (+2 more)

### Community 58 - "Community 58"
Cohesion: 0.22
Nodes (7): BrainInner, GooGraph, Mlp, Perceptron, SpatialBlob, SpatialGraph, PerceptronBrain

### Community 59 - "Community 59"
Cohesion: 0.32
Nodes (7): apply_metabolism_and_cull(), BirthRecord, pop_cap_blocks_births(), pop_cap_reached(), record_births_in_cladogram(), record_births_in_lineage(), run_reproduction_phase()

### Community 60 - "Community 60"
Cohesion: 0.29
Nodes (11): canvas, clamp(), ctx, drawFrame(), fpsInput, frameCount(), frameInput, loadBundle() (+3 more)

### Community 61 - "Community 61"
Cohesion: 0.27
Nodes (6): grid_cell_count(), GpuForwardState, make_env_view(), run_brain_forward(), run_brain_forward_cpu(), run_brain_forward_gpu()

### Community 62 - "Community 62"
Cohesion: 0.35
Nodes (10): collect_legacy_snapshots(), find_repo_root(), import_one(), load_run_record(), main(), read_text_file(), resolve_config_path(), RunRecordImport (+2 more)

### Community 63 - "Community 63"
Cohesion: 0.27
Nodes (3): copy_dynamics_config(), interactive_default_config(), merge_run_config_from_snapshot()

### Community 64 - "Community 64"
Cohesion: 0.20
Nodes (8): GOO_OP_AGGREGATE, GOO_OP_GENERATE, goo_op_id(), GOO_OP_KEEP_BEST, GOO_OP_SCORE, MAX_GOO_PARENTS, pack_goo(), SPATIAL_SECTOR_COUNT

### Community 65 - "Community 65"
Cohesion: 0.29
Nodes (5): dish_cell_count_formula(), open_dish_edge_neighbor_is_self(), resolve_neighbor_index(), torus_dish_wraps_across_bounding_box(), wrap_coord_in_box()

### Community 68 - "Community 68"
Cohesion: 0.29
Nodes (4): CladogramNodeState, HexCellState, IndividualState, SimulationState

### Community 69 - "Community 69"
Cohesion: 0.44
Nodes (7): choose(), discover_runs(), discover_snapshots(), launch_viewers(), main(), read_build_id(), run_replay()

### Community 70 - "Community 70"
Cohesion: 0.22
Nodes (6): BrainKind, GooGraph, Mlp, Perceptron, SpatialBlob, SpatialGraph

### Community 71 - "Community 71"
Cohesion: 0.32
Nodes (3): start_static_server(), test_metric_prop_moved_in_overview(), write_test_progress()

### Community 73 - "Community 73"
Cohesion: 0.39
Nodes (4): inject_live_meta(), inject_live_meta_running_sweep(), slim_for_html_embed(), slim_for_html_embed_strips_heavy_tables()

### Community 74 - "Community 74"
Cohesion: 0.39
Nodes (4): HexDishLayout, direction_to_sector(), neighbor_flat_dish(), compute_new_position()

### Community 75 - "Community 75"
Cohesion: 0.38
Nodes (3): build_cmd(), main(), run_one()

### Community 76 - "Community 76"
Cohesion: 0.29
Nodes (6): dish_info_line(), dish_run_config_candidates(), dish_run_config_display(), draw_brain_json_export_ui(), draw_dish_run_config_panel(), reveal_in_file_manager()

### Community 78 - "Community 78"
Cohesion: 0.29
Nodes (3): draw_env_line(), setup_env_axis(), update_env()

### Community 80 - "Community 80"
Cohesion: 0.40
Nodes (5): competition_heatmap_color(), hex_background_color(), HexBackgroundMode, Competition, Resources

### Community 81 - "Community 81"
Cohesion: 0.33
Nodes (4): FounderWeightMode, Neutral, RandomSmall, RandomUniform

## Knowledge Gaps
- **83 isolated node(s):** `Goo`, `Mlp`, `Perceptron`, `Spatial`, `Aggregate` (+78 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 508 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BrainGenome` connect `Community 17` to `Community 0`, `Community 1`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 20`, `Community 21`, `Community 22`, `Community 28`, `Community 30`, `Community 35`, `Community 48`, `Community 50`, `Community 58`, `Community 59`, `Community 64`, `Community 70`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **What connects `Goo`, `Mlp`, `Perceptron` to the rest of the system?**
  _83 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.09877264757451783 - nodes in this community are weakly interconnected._
- **Why does `ArchitecturePolicy` connect `Community 42` to `Community 0`, `Community 2`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 17`, `Community 20`, `Community 31`, `Community 38`, `Community 40`, `Community 43`, `Community 45`, `Community 47`, `Community 48`, `Community 51`, `Community 59`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.10651629072681704 - nodes in this community are weakly interconnected._
- **Why does `WebviewerWriter` connect `Community 3` to `Community 54`, `Community 7`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.05714285714285714 - nodes in this community are weakly interconnected._