use evo_movement_rust::brain::goo_graph::GooGraphBrain;
use evo_movement_rust::brain::{BrainGenome, BrainInner, FounderWeightMode};
use evo_movement_rust::brain_diagram::{diagram_from_brain, BrainDiagram};
use evo_movement_rust::build_info;
use evo_movement_rust::config::{ArchitecturePolicy, Config};
use evo_movement_rust::fitness_landscape::{
    architecture_label, build_cells_to_sweep, build_heatmaps_for_live, build_heatmaps_from_cells,
    cells_to_grid_specs, escape_json_for_script_embed, expand_output_mode_cells, inject_live_meta,
    slim_for_html_embed, CellAggregate, HeatmapData, LiveMeta, SweepCell, SweepProfile,
    TrajectoryPoint,
};
use evo_movement_rust::simulation::Simulation;
use evo_movement_rust::snapshot::PopulationSnapshot;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunMetric {
    architecture: String,
    x_label: String,
    y_label: String,
    founder_mode: String,
    seed: u32,
    #[serde(default)]
    run_steps: i32,
    steps_completed: i32,
    final_population: usize,
    max_population: i32,
    pop_ratio: f64,
    avg_resources: f64,
    avg_complexity: f64,
    prop_moved_final: f64,
    fitness_score: f64,
    #[serde(default)]
    immediate_extinction: bool,
    trajectory: Vec<TrajectoryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CellProbeMeta {
    brain_kind: String,
    movement_output_mode: String,
    movement_cost_factor: f64,
    resource_seed: u64,
    resource_scale: f64,
    resource_contrast: f64,
    env_perlin_spatial_scale: f64,
    env_perlin_time_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellDiagram {
    architecture: String,
    x_label: String,
    y_label: String,
    structure_signature: String,
    diagram: BrainDiagram,
    #[serde(default)]
    brain_json: String,
    #[serde(default)]
    probe_meta: CellProbeMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComplexityHelp {
    tooltip: String,
    per_architecture: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FitnessHelp {
    tooltip: String,
    formula: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LandscapeOutput {
    version: String,
    build_id: String,
    steps: i32,
    replicates: usize,
    founder_modes: Vec<String>,
    complexity_help: ComplexityHelp,
    fitness_help: FitnessHelp,
    architecture_labels: BTreeMap<String, String>,
    runs: Vec<RunMetric>,
    cells: Vec<CellAggregate>,
    heatmaps: Vec<HeatmapData>,
    cell_diagrams: Vec<CellDiagram>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct PromotedCell {
    architecture: String,
    x_label: String,
    y_label: String,
    founder_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StagedSweepState {
    stages: Vec<i32>,
    #[serde(default)]
    current_replicate_idx: usize,
    current_stage_idx: usize,
    all_runs: Vec<RunMetric>,
}

const DEFAULT_SWEEP_STAGES: &[i32] = &[10_000, 50_000, 100_000, 200_000];

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn landscape_output_from_runs(
    runs: &[RunMetric],
    cell_diagrams: &[CellDiagram],
    steps: i32,
    replicates: usize,
    founder_modes: &[String],
    expected_grid: Option<&[SweepCell]>,
) -> LandscapeOutput {
    let cells = aggregate_cells(runs);
    let grid_specs = expected_grid.map(cells_to_grid_specs);
    let heatmaps = build_heatmaps_for_live(
        &cells,
        grid_specs.as_deref(),
        founder_modes,
    );
    LandscapeOutput {
        version: build_info::CRATE_VERSION.to_string(),
        build_id: build_info::BUILD_ID.to_string(),
        steps,
        replicates,
        founder_modes: founder_modes.to_vec(),
        complexity_help: complexity_help(),
        fitness_help: fitness_help(),
        architecture_labels: architecture_labels(),
        runs: runs.to_vec(),
        cells,
        heatmaps,
        cell_diagrams: cell_diagrams.to_vec(),
    }
}

fn heatmaps_for_scope(
    cells: &[CellAggregate],
    cells_to_sweep: &[SweepCell],
    founder_modes: &[String],
) -> Vec<HeatmapData> {
    let specs = cells_to_grid_specs(cells_to_sweep);
    build_heatmaps_for_live(cells, Some(&specs), founder_modes)
}

fn flush_live_snapshot(
    out_dir: &Path,
    runs: &[RunMetric],
    cell_diagrams: &[CellDiagram],
    steps: i32,
    replicates: usize,
    founder_mode_labels: &[String],
    expected_grid: Option<&[SweepCell]>,
    live: LiveMeta,
) -> io::Result<()> {
    let out = landscape_output_from_runs(
        runs,
        cell_diagrams,
        steps,
        replicates,
        founder_mode_labels,
        expected_grid,
    );
    let db_path = out_dir.join("fitness_landscape.sqlite");
    let html_path = out_dir.join("fitness_landscape_heatmap.html");
    write_sqlite_atomic(&db_path, &out, Some(live))?;
    write_html(&html_path, &out, Some(live))
}

fn flush_staged_live(
    out_dir: &Path,
    all_runs: &[RunMetric],
    cell_diagrams: &[CellDiagram],
    cells_to_sweep: &[SweepCell],
    stages: &[i32],
    replicates: usize,
    founder_mode_labels: &[String],
    grand_total: usize,
) -> io::Result<()> {
    let combined = combined_latest_runs(all_runs);
    let steps = *stages.last().unwrap_or(&10_000);
    flush_live_snapshot(
        out_dir,
        &combined,
        cell_diagrams,
        steps,
        replicates,
        founder_mode_labels,
        Some(cells_to_sweep),
        LiveMeta {
            status: "running",
            run_count: all_runs.len(),
            total_runs: grand_total,
        },
    )
}

fn parse_arg_value<T: std::str::FromStr>(arg: &str, flag: &str) -> Option<T> {
    if arg.starts_with(flag) && arg.len() > flag.len() && arg.as_bytes()[flag.len()] == b'=' {
        return arg[flag.len() + 1..].parse::<T>().ok();
    }
    None
}

fn print_help() {
    println!("evo-fitness-landscape {} — usage:", build_info::VERSION_LABEL);
    println!("  --steps=N                   Steps per run (default 30000)");
    println!("  --replicates=N              Replicates per cell (default 3)");
    println!("  --output-dir=PATH           Output directory (default output/fitness_landscape_medium)");
    println!("  --base-config-root=PATH     Config root (default configs/ecology_plateau80/medium_100k)");
    println!("  --cap-multiplier=F          Multiply max_population to add cap headroom (default 1.8)");
    println!("  --founder-modes=LIST        Comma list: neutral,random_uniform,random_small (default neutral,random_uniform)");
    println!("  --architectures=LIST        Run subset: perceptron,mlp,spatial,goo (default: all)");
    println!("  --sweep-output-modes        Duplicate spatial/goo cells across movement_output_mode variants");
    println!("  --sweep-profile=PROFILE     full (default) | evolution_hunt (focused grid, both spatial layouts)");
    println!("  --staged-sweep              Replicate-first ladder: each rep runs 10k->50k->100k->200k from snapshots");
    println!("  --sweep-stages=LIST         Override stage step counts (default 10000,50000,100000,200000)");
    println!("  --merge-json=PATH           Merge new runs into existing JSON (replaces matching architectures)");
    println!("  --resume-from=PATH          Continue sweep from checkpoint JSON (skips completed runs)");
    println!("  --resume                    Resume from OUTPUT_DIR/fitness_landscape_checkpoint.json");
    println!("  --html-from-json=PATH       Regenerate HTML dashboard from existing JSON (no sims)");
    println!("  --refresh-skeleton          Rewrite live sqlite/html with full pending grid (no sims)");
    println!("  Live viewer: HTML polls fitness_landscape.sqlite every 2s while status=running");
    println!("  --help, -h                  Show help");
}

fn load_config(path: &Path) -> io::Result<Config> {
    Config::from_file(path)
}

fn load_config_named(base_root: &Path, name: &str, fallback_root: &Path) -> io::Result<Config> {
    let primary = config_path(base_root, name);
    if primary.is_file() {
        load_config(&primary)
    } else {
        load_config(&config_path(fallback_root, name))
    }
}

fn config_path(base_root: &Path, name: &str) -> PathBuf {
    base_root.join(format!("{name}.json"))
}

fn run_resume_key(r: &RunMetric) -> String {
    run_resume_key_parts(
        &r.architecture,
        &r.x_label,
        &r.y_label,
        &r.founder_mode,
        r.seed,
        r.run_steps,
    )
}

fn run_resume_key_parts(
    architecture: &str,
    x_label: &str,
    y_label: &str,
    founder_mode: &str,
    seed: u32,
    run_steps: i32,
) -> String {
    format!("{architecture}|{x_label}|{y_label}|{founder_mode}|{seed}|{run_steps}")
}

fn parse_sweep_stages(raw: &str) -> io::Result<Vec<i32>> {
    let stages: Vec<i32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<i32>().ok())
        .map(|s| s.max(1000))
        .collect();
    if stages.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sweep stages list is empty",
        ));
    }
    Ok(stages)
}

fn promoted_cell_key(arch: &str, x: &str, y: &str, founder: &str) -> PromotedCell {
    PromotedCell {
        architecture: arch.to_string(),
        x_label: x.to_string(),
        y_label: y.to_string(),
        founder_mode: founder.to_string(),
    }
}

fn promoted_cell_from_run(r: &RunMetric) -> PromotedCell {
    promoted_cell_key(&r.architecture, &r.x_label, &r.y_label, &r.founder_mode)
}

fn promoted_cell_from_aggregate(c: &CellAggregate) -> PromotedCell {
    promoted_cell_key(&c.architecture, &c.x_label, &c.y_label, &c.founder_mode)
}

fn is_promising_cell(cell: &CellAggregate) -> bool {
    if cell.data_na {
        return false;
    }
    if cell.fitness_mean > 0.05 {
        return true;
    }
    if cell.pop_ratio_mean > 0.10 {
        return true;
    }
    if cell.trajectory_mean.len() > 1 {
        if let Some(last) = cell.trajectory_mean.last() {
            if last.population > 0 && last.fitness > 0.01 {
                return true;
            }
        }
    }
    false
}

fn select_promoted_cells(cells: &[CellAggregate]) -> Vec<PromotedCell> {
    cells
        .iter()
        .filter(|c| is_promising_cell(c))
        .map(promoted_cell_from_aggregate)
        .collect()
}

fn combined_latest_runs(runs: &[RunMetric]) -> Vec<RunMetric> {
    let mut latest: BTreeMap<(String, String, String, String, u32), RunMetric> = BTreeMap::new();
    for r in runs {
        let key = (
            r.architecture.clone(),
            r.x_label.clone(),
            r.y_label.clone(),
            r.founder_mode.clone(),
            r.seed,
        );
        latest
            .entry(key)
            .and_modify(|prev| {
                if r.run_steps >= prev.run_steps {
                    *prev = r.clone();
                }
            })
            .or_insert(r.clone());
    }
    latest.into_values().collect()
}

fn sanitize_filename_part(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '=' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn save_sim_snapshot(sim: &Simulation, cfg: &Config, timestep: i32, path: &Path) -> io::Result<()> {
    let env_height = if cfg.hex_dish_radius > 0 {
        sim.hex_dish
            .as_ref()
            .map(|d| d.bounding_size as u32)
            .unwrap_or(0)
    } else {
        cfg.grid_height.max(cfg.grid_width) as u32
    };
    let hex_dish_radius = if cfg.hex_dish_radius > 0 {
        cfg.hex_dish_radius.max(0) as u32
    } else {
        0
    };
    let snapshot = PopulationSnapshot::capture(
        &sim.env,
        &sim.individuals,
        sim.next_id,
        cfg.seed as u64,
        timestep,
        env_height,
        hex_dish_radius,
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    snapshot
        .write_to_path(path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(())
}

fn cell_snapshot_path(stage_dir: &Path, cell: &PromotedCell, seed: u32) -> PathBuf {
    stage_dir.join("snapshots").join(format!(
        "{}__{}__{}__{}__seed{}.cbor",
        sanitize_filename_part(&cell.architecture),
        sanitize_filename_part(&cell.x_label),
        sanitize_filename_part(&cell.y_label),
        sanitize_filename_part(&cell.founder_mode),
        seed,
    ))
}

fn write_stage_report(
    stage_dir: &Path,
    stage_steps: i32,
    cells: &[CellAggregate],
    promoted: &[PromotedCell],
) -> io::Result<()> {
    let report_path = stage_dir.join("promotion_report.txt");
    let mut out = format!(
        "Stage {stage_steps} complete\n{}\n\nPromoted {} / {} cells to next stage:\n\n",
        "=".repeat(60),
        promoted.len(),
        cells.len()
    );
    for c in cells {
        let p = promoted_cell_from_aggregate(c);
        let mark = if promoted.iter().any(|x| x == &p) {
            "PROMOTE"
        } else if c.data_na {
            "N/A"
        } else {
            "drop"
        };
        out.push_str(&format!(
            "[{mark}] {} | {} x {} | founder={} | fitness={:.3} pop={:.3}\n",
            c.architecture, c.x_label, c.y_label, c.founder_mode, c.fitness_mean, c.pop_ratio_mean
        ));
    }
    fs::write(report_path, out)?;
    let json_path = stage_dir.join("promoted_cells.json");
    fs::write(json_path, serde_json::to_string_pretty(promoted)?)?;
    Ok(())
}

fn write_landscape_outputs(
    out_dir: &Path,
    payload: &LandscapeOutput,
    live: Option<LiveMeta>,
) -> io::Result<()> {
    let json_path = out_dir.join("fitness_landscape.json");
    let csv_path = out_dir.join("fitness_landscape_cells.csv");
    let db_path = out_dir.join("fitness_landscape.sqlite");
    let html_path = out_dir.join("fitness_landscape_heatmap.html");
    fs::write(json_path, serde_json::to_string_pretty(payload)?)?;
    write_csv(&csv_path, &payload.cells)?;
    write_sqlite_atomic(&db_path, payload, live)?;
    write_html(&html_path, payload, live)?;
    Ok(())
}

fn is_spatial_architecture(arch: &str) -> bool {
    arch.contains("Spatial") || arch == "spatial_graph" || arch == "spatial_blob"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SweepArch {
    Perceptron,
    Mlp,
    Spatial,
    Goo,
}

fn parse_architectures(raw: &str) -> Vec<SweepArch> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "" => {}
            "perceptron" | "p" => out.push(SweepArch::Perceptron),
            "mlp" => out.push(SweepArch::Mlp),
            "spatial" | "spatial_graph" | "spatial_blob" | "sg" => out.push(SweepArch::Spatial),
            "goo" | "goo_graph" => out.push(SweepArch::Goo),
            other => eprintln!("warning: unknown architecture filter '{other}'"),
        }
    }
    out.sort();
    out.dedup();
    out
}

fn cell_matches_arch_filter(cell: &SweepCell, filter: &[SweepArch]) -> bool {
    if filter.is_empty() {
        return true;
    }
    let want = match cell.cfg.architecture_policy {
        ArchitecturePolicy::PerceptronOnly => SweepArch::Perceptron,
        ArchitecturePolicy::EvolveMlp => SweepArch::Mlp,
        ArchitecturePolicy::EvolveSpatialGraph => SweepArch::Spatial,
        ArchitecturePolicy::EvolveGoo => SweepArch::Goo,
    };
    filter.contains(&want)
}

fn prep_config_for_run(mut cfg: Config, steps: i32, seed: u32, cap_multiplier: f64) -> Config {
    cfg.seed = seed;
    cfg.steps = steps;
    cfg.report_interval = (steps / 20).max(250);
    cfg.write_outputs = false;
    cfg.write_snapshot = false;
    cfg.write_config = false;
    cfg.show_progress = false;
    cfg.output_run_dir.clear();
    // Keep architecture_mutation_rate from base config so GoO can evolve Agg/Score/KeepBest.
    if cfg.max_population > 0 && cap_multiplier > 1.0 {
        cfg.max_population = ((cfg.max_population as f64) * cap_multiplier).round() as i32;
    }
    if cfg.architecture_policy == ArchitecturePolicy::EvolveSpatialGraph
        && !cfg.uses_legacy_spatial_graph()
    {
        cfg.use_gpu_step = false;
    }
    cfg.finalize();
    cfg
}

fn seed_goo_on_brain(brain: &mut BrainGenome, cfg: &Config, goo_target_nodes: usize, seed: u64) {
    if cfg.architecture_policy != ArchitecturePolicy::EvolveGoo {
        return;
    }
    if let BrainInner::GooGraph(ref mut goo) = brain.inner {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FFEE_u64);
        *goo = GooGraphBrain::new_initial();
        goo.grow_to_size(&mut rng, goo_target_nodes.max(1));
    }
}

fn seed_goo_founders(sim: &mut Simulation, cfg: &Config, goo_target_nodes: usize, seed: u64) {
    if cfg.architecture_policy != ArchitecturePolicy::EvolveGoo {
        return;
    }
    let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FFEE_u64);
    for ind in sim.individuals.iter_mut() {
        if let BrainInner::GooGraph(ref mut goo) = ind.brain.inner {
            *goo = GooGraphBrain::new_initial();
            goo.grow_to_size(&mut rng, goo_target_nodes.max(1));
        }
        ind.spatial_runtime = ind.brain.init_spatial_runtime();
    }
}

fn representative_brain(cfg: &Config, goo_generate_nodes: usize, seed: u64) -> BrainGenome {
    let mut brain = BrainGenome::new_founder(cfg);
    seed_goo_on_brain(&mut brain, cfg, goo_generate_nodes, seed);
    brain
}

fn prop_moved(sim: &Simulation) -> f64 {
    if sim.last_delta.is_empty() || sim.individuals.is_empty() {
        return 0.0;
    }
    let seen = std::cmp::min(sim.last_delta.len(), sim.individuals.len());
    if seen == 0 {
        return 0.0;
    }
    sim.last_delta.iter().take(seen).filter(|&&d| d != 0).count() as f64 / seen as f64
}

fn mean_resources(sim: &Simulation) -> f64 {
    if sim.individuals.is_empty() {
        return 0.0;
    }
    sim.individuals.iter().map(|i| i.resources).sum::<f64>() / sim.individuals.len() as f64
}

fn mean_complexity(sim: &Simulation) -> f64 {
    if sim.individuals.is_empty() {
        return 0.0;
    }
    sim.individuals
        .iter()
        .map(|i| i.complexity() as f64)
        .sum::<f64>()
        / sim.individuals.len() as f64
}

fn fitness_score(pop_ratio: f64, avg_resources: f64, avg_complexity: f64) -> f64 {
    let cap_penalty = if pop_ratio > 0.95 {
        (1.0 - ((pop_ratio - 0.95) / 0.05) * 0.4).clamp(0.6, 1.0)
    } else {
        1.0
    };
    let resource_term = (avg_resources / 20.0).clamp(0.0, 1.5);
    let complexity_term = (avg_complexity.ln_1p() / 4.0).clamp(0.0, 1.0);
    pop_ratio * (0.55 + 0.25 * resource_term + 0.20 * complexity_term) * cap_penalty
}

fn sample_point(sim: &Simulation, step: i32, max_population: i32) -> TrajectoryPoint {
    let final_population = sim.individuals.len();
    let pop_ratio = if max_population > 0 {
        (final_population as f64 / max_population as f64).clamp(0.0, 1.5)
    } else {
        0.0
    };
    let avg_resources = mean_resources(sim);
    let avg_complexity = mean_complexity(sim);
    TrajectoryPoint {
        step,
        fitness: fitness_score(pop_ratio, avg_resources, avg_complexity),
        pop_ratio,
        avg_complexity,
        population: final_population,
    }
}

fn run_cell(
    mut cfg: Config,
    steps: i32,
    seed: u32,
    cap_multiplier: f64,
    x_label: &str,
    y_label: &str,
    goo_generate_nodes: usize,
    founder_mode: FounderWeightMode,
) -> (RunMetric, Simulation) {
    cfg = prep_config_for_run(cfg, steps, seed, cap_multiplier);
    let arch = architecture_label(&cfg);
    let max_population = cfg.max_population;
    let sample_every = (steps / 60).max(250).max(1);

    let mut sim = Simulation::new(&cfg);
    sim.init_population_with_founder_mode(cfg.initial_population.max(1) as usize, Some(founder_mode));
    seed_goo_founders(&mut sim, &cfg, goo_generate_nodes, seed as u64);

    let mut trajectory = vec![sample_point(&sim, 0, max_population)];
    let mut steps_completed = 0i32;
    for step in 1..=steps {
        if sim.individuals.is_empty() {
            break;
        }
        sim.step();
        steps_completed = step;
        if step % sample_every == 0 || step == steps {
            trajectory.push(sample_point(&sim, step, max_population));
        }
    }

    let final_population = sim.individuals.len();
    let pop_ratio = if max_population > 0 {
        (final_population as f64 / max_population as f64).clamp(0.0, 1.5)
    } else {
        0.0
    };
    let avg_resources = mean_resources(&sim);
    let avg_complexity = mean_complexity(&sim);
    let prop_moved_final = prop_moved(&sim);
    let fitness = fitness_score(pop_ratio, avg_resources, avg_complexity);
    let immediate_extinction = final_population == 0 && trajectory.len() <= 1;

    (
        RunMetric {
            architecture: arch,
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
            founder_mode: founder_mode.label().to_string(),
            seed,
            run_steps: steps,
            steps_completed,
            final_population,
            max_population,
            pop_ratio,
            avg_resources,
            avg_complexity,
            prop_moved_final,
            fitness_score: fitness,
            immediate_extinction,
            trajectory,
        },
        sim,
    )
}

fn run_cell_continue(
    mut cfg: Config,
    total_steps: i32,
    steps_already: i32,
    seed: u32,
    cap_multiplier: f64,
    x_label: &str,
    y_label: &str,
    snapshot_path: &Path,
) -> io::Result<(RunMetric, Simulation)> {
    cfg = prep_config_for_run(cfg, total_steps, seed, cap_multiplier);
    let arch = architecture_label(&cfg);
    let max_population = cfg.max_population;
    let sample_every = (total_steps / 60).max(250).max(1);

    let mut sim = Simulation::new(&cfg);
    sim.load_snapshot_from_path(snapshot_path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut trajectory = vec![sample_point(&sim, steps_already, max_population)];
    let mut steps_completed = steps_already;
    for step in (steps_already + 1)..=total_steps {
        if sim.individuals.is_empty() {
            break;
        }
        sim.step();
        steps_completed = step;
        if step % sample_every == 0 || step == total_steps {
            trajectory.push(sample_point(&sim, step, max_population));
        }
    }

    let final_population = sim.individuals.len();
    let pop_ratio = if max_population > 0 {
        (final_population as f64 / max_population as f64).clamp(0.0, 1.5)
    } else {
        0.0
    };
    let avg_resources = mean_resources(&sim);
    let avg_complexity = mean_complexity(&sim);
    let prop_moved_final = prop_moved(&sim);
    let fitness = fitness_score(pop_ratio, avg_resources, avg_complexity);
    let immediate_extinction = final_population == 0 && trajectory.len() <= 1;

    Ok((
        RunMetric {
            architecture: arch,
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
            founder_mode: String::new(), // filled by caller
            seed,
            run_steps: total_steps,
            steps_completed,
            final_population,
            max_population,
            pop_ratio,
            avg_resources,
            avg_complexity,
            prop_moved_final,
            fitness_score: fitness,
            immediate_extinction,
            trajectory,
        },
        sim,
    ))
}

fn replicate_seed(cell: &SweepCell, cell_idx: usize, founder_idx: usize, replicate: usize) -> u32 {
    cell.cfg
        .seed
        .wrapping_add((cell_idx * 101 + founder_idx * 1009 + replicate) as u32)
}

fn save_stage_snapshot(
    sim: &Simulation,
    cfg: &Config,
    metric: &RunMetric,
    stage_dir: &Path,
) {
    if metric.final_population == 0 {
        return;
    }
    let key = promoted_cell_from_run(metric);
    let path = cell_snapshot_path(stage_dir, &key, metric.seed);
    let _ = save_sim_snapshot(sim, cfg, metric.steps_completed, &path);
}

fn push_staged_run(
    all_runs: &mut Vec<RunMetric>,
    cell: &SweepCell,
    total_steps: i32,
    prev_stage_steps: Option<i32>,
    prev_stage_dir: Option<&Path>,
    seed: u32,
    cap_multiplier: f64,
    founder_mode: FounderWeightMode,
    stage_dir: &Path,
) -> bool {
    let arch = architecture_label(&cell.cfg);
    let founder_label = founder_mode.label().to_string();
    let snapshot_key = promoted_cell_key(&arch, &cell.x_label, &cell.y_label, &founder_label);

    let (mut metric, sim) = if let (Some(_prev_steps), Some(prev_dir)) =
        (prev_stage_steps, prev_stage_dir)
    {
        let snap_path = cell_snapshot_path(prev_dir, &snapshot_key, seed);
        if !snap_path.is_file() {
            println!(
                "  skip continue (no snapshot): {arch} {} {} founder={founder_label} seed={seed}",
                cell.x_label, cell.y_label
            );
            return false;
        }
        match run_cell_continue(
            cell.cfg.clone(),
            total_steps,
            prev_stage_steps.unwrap_or(0),
            seed,
            cap_multiplier,
            &cell.x_label,
            &cell.y_label,
            &snap_path,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                println!(
                    "  skip continue ({e}): {arch} {} {} founder={founder_label} seed={seed}",
                    cell.x_label, cell.y_label
                );
                return false;
            }
        }
    } else {
        run_cell(
            cell.cfg.clone(),
            total_steps,
            seed,
            cap_multiplier,
            &cell.x_label,
            &cell.y_label,
            cell.goo_generate_nodes,
            founder_mode,
        )
    };

    metric.founder_mode = founder_label;
    save_stage_snapshot(&sim, &cell.cfg, &metric, stage_dir);
    all_runs.retain(|x| run_resume_key(x) != run_resume_key(&metric));
    all_runs.push(metric);
    true
}

fn runs_at_stage(runs: &[RunMetric], stage_steps: i32) -> Vec<RunMetric> {
    runs.iter()
        .filter(|r| r.run_steps == stage_steps)
        .cloned()
        .collect()
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

fn stddev(vals: &[f64], m: f64) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64;
    var.sqrt()
}

fn average_trajectories(runs: &[&RunMetric]) -> Vec<TrajectoryPoint> {
    if runs.is_empty() {
        return Vec::new();
    }
    let mut by_step: BTreeMap<i32, Vec<&TrajectoryPoint>> = BTreeMap::new();
    for r in runs {
        for p in &r.trajectory {
            by_step.entry(p.step).or_default().push(p);
        }
    }
    by_step
        .into_iter()
        .map(|(step, pts)| TrajectoryPoint {
            step,
            fitness: mean(&pts.iter().map(|p| p.fitness).collect::<Vec<_>>()),
            pop_ratio: mean(&pts.iter().map(|p| p.pop_ratio).collect::<Vec<_>>()),
            avg_complexity: mean(&pts.iter().map(|p| p.avg_complexity).collect::<Vec<_>>()),
            population: (mean(
                &pts.iter()
                    .map(|p| p.population as f64)
                    .collect::<Vec<_>>(),
            )) as usize,
        })
        .collect()
}

fn aggregate_cells(runs: &[RunMetric]) -> Vec<CellAggregate> {
    let mut by_key: BTreeMap<(String, String, String, String), Vec<&RunMetric>> = BTreeMap::new();
    for r in runs {
        by_key
            .entry((
                r.architecture.clone(),
                r.x_label.clone(),
                r.y_label.clone(),
                r.founder_mode.clone(),
            ))
            .or_default()
            .push(r);
    }
    by_key
        .into_iter()
        .map(|((arch, x, y, founder), vals)| {
            let fitness: Vec<f64> = vals.iter().map(|v| v.fitness_score).collect();
            let pop_ratio: Vec<f64> = vals.iter().map(|v| v.pop_ratio).collect();
            let complexity: Vec<f64> = vals.iter().map(|v| v.avg_complexity).collect();
            let fm = mean(&fitness);
            CellAggregate {
                architecture: arch,
                x_label: x,
                y_label: y,
                founder_mode: founder,
                replicates: vals.len(),
                fitness_mean: fm,
                fitness_std: stddev(&fitness, fm),
                pop_ratio_mean: mean(&pop_ratio),
                complexity_mean: mean(&complexity),
                data_na: vals.iter().all(|v| v.immediate_extinction),
                trajectory_mean: average_trajectories(&vals),
            }
        })
        .collect()
}

fn build_cell_diagrams(
    specs: &[(Config, String, String, usize, u64)],
) -> Vec<CellDiagram> {
    let mut out = Vec::new();
    for (cfg, x_label, y_label, goo_gen, seed) in specs {
        let brain = representative_brain(cfg, *goo_gen, *seed);
        let arch = architecture_label(cfg);
        let brain_json = serde_json::to_string(&brain).unwrap_or_else(|_| "{}".to_string());
        out.push(CellDiagram {
            architecture: arch,
            x_label: x_label.clone(),
            y_label: y_label.clone(),
            structure_signature: brain.structure_signature(),
            diagram: diagram_from_brain(&brain, 360.0, 200.0),
            brain_json,
            probe_meta: CellProbeMeta {
                brain_kind: brain.kind().as_str().to_string(),
                movement_output_mode: cfg.resolved_movement_output_mode().label().to_string(),
                movement_cost_factor: cfg.movement_cost_factor,
                resource_seed: cfg.seed as u64,
                resource_scale: cfg.resource_scale,
                resource_contrast: cfg.resource_contrast,
                env_perlin_spatial_scale: cfg.env_perlin_spatial_scale,
                env_perlin_time_scale: cfg.env_perlin_time_scale,
            },
        });
    }
    out
}

fn complexity_help() -> ComplexityHelp {
    let mut per_architecture = BTreeMap::new();
    per_architecture.insert(
        "perceptron".to_string(),
        "Per hidden node: nodes + (weights÷inputs). Counts decision units and their input fan-in.".to_string(),
    );
    per_architecture.insert(
        "mlp".to_string(),
        "Layer sizes: hidden nodes (L1) + hidden nodes (L2) + 1 readout group.".to_string(),
    );
    per_architecture.insert(
        "spatial_graph".to_string(),
        "Landmarks + graph edges + edge-weight slots (message-passing topology).".to_string(),
    );
    per_architecture.insert(
        "spatial_blob".to_string(),
        "Hex-edge or cardinal spatial layers + recurrent mixing; max_brain_nodes caps layer count.".to_string(),
    );
    per_architecture.insert(
        "goo".to_string(),
        "GoO nodes + parent links + parameter groups (thought_dim-sized blocks).".to_string(),
    );
    ComplexityHelp {
        tooltip: "Mean complexity is the population average of each individual's brain.complexity(): a coarse size/capacity score (not fitness by itself). Structural mutations can raise it over time; the heatmap uses the final-step population mean.".to_string(),
        per_architecture,
    }
}

fn fitness_help() -> FitnessHelp {
    FitnessHelp {
        tooltip: "Composite landscape score at each sample step. Rewards sustaining population near (but not pinned at) carrying capacity, moderate resource stores, and non-trivial brain complexity. Extinction yields fitness ≈ 0.".to_string(),
        formula: "fitness = pop_ratio × (0.55 + 0.25×resource_term + 0.20×complexity_term) × cap_penalty\n\npop_ratio = population ÷ max_population (clamped 0–1.5)\nresource_term = clamp(mean internal stores ÷ 20, 0, 1.5)\ncomplexity_term = clamp(ln(1 + mean brain complexity) ÷ 4, 0, 1)\ncap_penalty = 1.0 if pop_ratio ≤ 0.95; else linear down to 0.6 at pop_ratio ≥ 1.0 (discourages pinning at cap)".to_string(),
    }
}

fn architecture_labels() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for policy in [
        ArchitecturePolicy::PerceptronOnly,
        ArchitecturePolicy::EvolveMlp,
        ArchitecturePolicy::EvolveSpatialGraph,
        ArchitecturePolicy::EvolveGoo,
    ] {
        let label = policy.label().to_string();
        m.insert(label.clone(), label);
    }
    // Short keys used in older exports / complexity help.
    m.insert("perceptron".to_string(), "Perceptron".to_string());
    m.insert("mlp".to_string(), "Multilayer perceptron (MLP)".to_string());
    m.insert("spatial_graph".to_string(), "Spatial graph (SG)".to_string());
    m.insert("spatial_blob".to_string(), "Spatial macromolecule blob".to_string());
    m.insert("goo".to_string(), "Graph of operations (GoO)".to_string());
    m.insert("GoO".to_string(), "Graph of operations (GoO)".to_string());
    m.insert("MLP".to_string(), "Multilayer perceptron (MLP)".to_string());
    m
}

fn write_csv(path: &Path, rows: &[CellAggregate]) -> io::Result<()> {
    let mut out = String::from(
        "architecture,x_label,y_label,founder_mode,replicates,fitness_mean,fitness_std,pop_ratio_mean,complexity_mean,data_na\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{}\n",
            r.architecture,
            r.x_label,
            r.y_label,
            r.founder_mode,
            r.replicates,
            r.fitness_mean,
            r.fitness_std,
            r.pop_ratio_mean,
            r.complexity_mean,
            r.data_na
        ));
    }
    fs::write(path, out)
}

fn write_sqlite(path: &Path, out: &LandscapeOutput, live: Option<LiveMeta>) -> io::Result<()> {
    if path.exists() {
        fs::remove_file(path)?;
    }
    let conn = Connection::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    conn.execute_batch(
        "
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY,
            architecture TEXT NOT NULL,
            x_label TEXT NOT NULL,
            y_label TEXT NOT NULL,
            founder_mode TEXT NOT NULL,
            seed INTEGER NOT NULL,
            steps_completed INTEGER NOT NULL,
            final_population INTEGER NOT NULL,
            max_population INTEGER NOT NULL,
            pop_ratio REAL NOT NULL,
            avg_resources REAL NOT NULL,
            avg_complexity REAL NOT NULL,
            prop_moved_final REAL NOT NULL,
            fitness_score REAL NOT NULL,
            immediate_extinction INTEGER NOT NULL,
            trajectory_json TEXT NOT NULL
        );
        CREATE TABLE cells (
            id INTEGER PRIMARY KEY,
            architecture TEXT NOT NULL,
            x_label TEXT NOT NULL,
            y_label TEXT NOT NULL,
            founder_mode TEXT NOT NULL,
            replicates INTEGER NOT NULL,
            fitness_mean REAL NOT NULL,
            fitness_std REAL NOT NULL,
            pop_ratio_mean REAL NOT NULL,
            complexity_mean REAL NOT NULL,
            data_na INTEGER NOT NULL,
            trajectory_mean_json TEXT NOT NULL
        );
        CREATE TABLE heatmaps (
            id INTEGER PRIMARY KEY,
            architecture TEXT NOT NULL,
            founder_mode TEXT NOT NULL,
            x_labels_json TEXT NOT NULL,
            y_labels_json TEXT NOT NULL,
            fitness_matrix_json TEXT NOT NULL,
            fitness_na_matrix_json TEXT NOT NULL,
            pop_ratio_matrix_json TEXT NOT NULL,
            complexity_matrix_json TEXT NOT NULL
        );
        CREATE TABLE cell_diagrams (
            id INTEGER PRIMARY KEY,
            architecture TEXT NOT NULL,
            x_label TEXT NOT NULL,
            y_label TEXT NOT NULL,
            structure_signature TEXT NOT NULL,
            diagram_json TEXT NOT NULL,
            brain_json TEXT NOT NULL DEFAULT '',
            probe_meta_json TEXT NOT NULL DEFAULT '{}'
        );
        ",
    )
    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let meta_pairs = [
        ("version", out.version.as_str()),
        ("build_id", out.build_id.as_str()),
        ("steps", &out.steps.to_string()),
        ("replicates", &out.replicates.to_string()),
        (
            "founder_modes",
            &serde_json::to_string(&out.founder_modes).unwrap_or_else(|_| "[]".to_string()),
        ),
        (
            "complexity_help",
            &serde_json::to_string(&out.complexity_help).unwrap_or_else(|_| "{}".to_string()),
        ),
        (
            "fitness_help",
            &serde_json::to_string(&out.fitness_help).unwrap_or_else(|_| "{}".to_string()),
        ),
        (
            "architecture_labels",
            &serde_json::to_string(&out.architecture_labels).unwrap_or_else(|_| "{}".to_string()),
        ),
    ];
    let mut meta_pairs: Vec<(&str, String)> = meta_pairs
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
    if let Some(l) = live {
        meta_pairs.push(("status", l.status.to_string()));
        meta_pairs.push(("run_count", l.run_count.to_string()));
        meta_pairs.push(("total_runs", l.total_runs.to_string()));
        meta_pairs.push(("updated_at_ms", now_ms().to_string()));
    } else {
        meta_pairs.push(("status", "complete".to_string()));
        meta_pairs.push(("run_count", out.runs.len().to_string()));
        meta_pairs.push(("total_runs", out.runs.len().to_string()));
        meta_pairs.push(("updated_at_ms", now_ms().to_string()));
    }
    for (k, v) in meta_pairs {
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)",
            params![k, v],
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }

    for r in &out.runs {
        conn.execute(
            "INSERT INTO runs (
                architecture, x_label, y_label, founder_mode, seed, steps_completed,
                final_population, max_population, pop_ratio, avg_resources, avg_complexity,
                prop_moved_final, fitness_score, immediate_extinction, trajectory_json
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            params![
                r.architecture,
                r.x_label,
                r.y_label,
                r.founder_mode,
                r.seed,
                r.steps_completed,
                r.final_population,
                r.max_population,
                r.pop_ratio,
                r.avg_resources,
                r.avg_complexity,
                r.prop_moved_final,
                r.fitness_score,
                r.immediate_extinction as i32,
                serde_json::to_string(&r.trajectory).unwrap_or_else(|_| "[]".to_string()),
            ],
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }

    for c in &out.cells {
        conn.execute(
            "INSERT INTO cells (
                architecture, x_label, y_label, founder_mode, replicates,
                fitness_mean, fitness_std, pop_ratio_mean, complexity_mean, data_na,
                trajectory_mean_json
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                c.architecture,
                c.x_label,
                c.y_label,
                c.founder_mode,
                c.replicates,
                c.fitness_mean,
                c.fitness_std,
                c.pop_ratio_mean,
                c.complexity_mean,
                c.data_na as i32,
                serde_json::to_string(&c.trajectory_mean).unwrap_or_else(|_| "[]".to_string()),
            ],
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }

    for h in &out.heatmaps {
        let fit_na = if h.fitness_na_matrix.is_empty() {
            vec![vec![false; h.x_labels.len()]; h.y_labels.len()]
        } else {
            h.fitness_na_matrix.clone()
        };
        conn.execute(
            "INSERT INTO heatmaps (
                architecture, founder_mode, x_labels_json, y_labels_json,
                fitness_matrix_json, fitness_na_matrix_json,
                pop_ratio_matrix_json, complexity_matrix_json
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                h.architecture,
                h.founder_mode,
                serde_json::to_string(&h.x_labels).unwrap_or_else(|_| "[]".to_string()),
                serde_json::to_string(&h.y_labels).unwrap_or_else(|_| "[]".to_string()),
                serde_json::to_string(&h.fitness_matrix).unwrap_or_else(|_| "[]".to_string()),
                serde_json::to_string(&fit_na).unwrap_or_else(|_| "[]".to_string()),
                serde_json::to_string(&h.pop_ratio_matrix).unwrap_or_else(|_| "[]".to_string()),
                serde_json::to_string(&h.complexity_matrix).unwrap_or_else(|_| "[]".to_string()),
            ],
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }

    for d in &out.cell_diagrams {
        conn.execute(
            "INSERT INTO cell_diagrams (
                architecture, x_label, y_label, structure_signature, diagram_json,
                brain_json, probe_meta_json
            ) VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![
                d.architecture,
                d.x_label,
                d.y_label,
                d.structure_signature,
                serde_json::to_string(&d.diagram).unwrap_or_else(|_| "{}".to_string()),
                d.brain_json,
                serde_json::to_string(&d.probe_meta).unwrap_or_else(|_| "{}".to_string()),
            ],
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }

    Ok(())
}

fn is_file_lock_error(err: &io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(32) | Some(33) // Windows sharing violation / lock
    )
}

fn write_sqlite_atomic(path: &Path, out: &LandscapeOutput, live: Option<LiveMeta>) -> io::Result<()> {
    let tmp = path.with_extension("sqlite.tmp");
    write_sqlite(&tmp, out, live)?;
    const MAX_TRIES: u32 = 10;
    for attempt in 0..MAX_TRIES {
        let result = (|| -> io::Result<()> {
            if path.exists() {
                fs::remove_file(path)?;
            }
            fs::rename(&tmp, path)?;
            Ok(())
        })();
        match result {
            Ok(()) => return Ok(()),
            Err(e) if attempt + 1 < MAX_TRIES && is_file_lock_error(&e) => {
                std::thread::sleep(std::time::Duration::from_millis(40 * (attempt as u64 + 1)));
            }
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::Other,
        format!("failed to replace {} after retries", path.display()),
    ))
}

const VIEWER_TEMPLATE: &str = include_str!("../../assets/fitness_landscape_viewer.html");

fn load_viewer_template() -> io::Result<String> {
    let mut tried: Vec<PathBuf> = Vec::new();
    let mut push_candidate = |p: PathBuf| {
        if p.is_file() {
            if let Ok(s) = fs::read_to_string(&p) {
                return Some(s);
            }
        }
        tried.push(p);
        None
    };

    if let Ok(cwd) = env::current_dir() {
        if let Some(s) = push_candidate(cwd.join("assets/fitness_landscape_viewer.html")) {
            return Ok(s);
        }
        if let Some(s) = push_candidate(cwd.join("rust/assets/fitness_landscape_viewer.html")) {
            return Ok(s);
        }
        let mut dir = cwd.as_path();
        while let Some(parent) = dir.parent() {
            if let Some(s) =
                push_candidate(parent.join("rust/assets/fitness_landscape_viewer.html"))
            {
                return Ok(s);
            }
            dir = parent;
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            for rel in [
                "assets/fitness_landscape_viewer.html",
                "../assets/fitness_landscape_viewer.html",
                "../../assets/fitness_landscape_viewer.html",
                "../../../rust/assets/fitness_landscape_viewer.html",
            ] {
                if let Some(s) = push_candidate(dir.join(rel)) {
                    return Ok(s);
                }
            }
        }
    }

    Ok(VIEWER_TEMPLATE.to_string())
}

fn viewer_json_payload(out: &LandscapeOutput, live: Option<LiveMeta>) -> io::Result<String> {
    let val = serde_json::to_value(out).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let val = slim_for_html_embed(val);
    let val = inject_live_meta(val, live, out.runs.len());
    let mut val = val.as_object().cloned().unwrap_or_default();
    val.insert("updated_at_ms".into(), serde_json::json!(now_ms()));
    let json = serde_json::to_string(&val).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(escape_json_for_script_embed(&json))
}

fn write_html(path: &Path, out: &LandscapeOutput, live: Option<LiveMeta>) -> io::Result<()> {
    let template = load_viewer_template()?;
    let embed = viewer_json_payload(out, live)?;
    // Replace only the <script id="landscape-data"> slot — not the JS sentinel in parseEmbeddedLandscape.
    let html = template.replacen("__LANDSCAPE_JSON__", &embed, 1);
    fs::write(path, html)?;
    if let Some(out_dir) = path.parent() {
        copy_probe_wasm_bundle(out_dir);
    }
    Ok(())
}

fn copy_probe_wasm_bundle(out_dir: &Path) {
    let candidates = [
        PathBuf::from("assets/brain_playground_wasm"),
        PathBuf::from("rust/assets/brain_playground_wasm"),
        PathBuf::from("../docs/brain_playground_wasm"),
        PathBuf::from("docs/brain_playground_wasm"),
    ];
    let mut src: Option<PathBuf> = None;
    if let Ok(cwd) = env::current_dir() {
        for rel in candidates {
            let p = cwd.join(&rel);
            if p.join("brain_playground_wasm.js").is_file() {
                src = Some(p);
                break;
            }
        }
    }
    let Some(src) = src else {
        return;
    };
    let dst = out_dir.join("brain_playground_wasm");
    let _ = fs::create_dir_all(&dst);
    for name in [
        "brain_playground_wasm.js",
        "brain_playground_wasm_bg.wasm",
        "brain_playground_wasm.d.ts",
        "package.json",
    ] {
        let from = src.join(name);
        if from.is_file() {
            let _ = fs::copy(&from, dst.join(name));
        }
    }
}

fn backfill_extinction_flags(out: &mut LandscapeOutput) {
    for r in &mut out.runs {
        if !r.immediate_extinction {
            r.immediate_extinction = r.final_population == 0 && r.trajectory.len() <= 1;
        }
    }
}

fn cell_diagrams_need_probe_backfill(diagrams: &[CellDiagram]) -> bool {
    diagrams.is_empty() || diagrams.iter().any(|d| d.brain_json.is_empty())
}

fn backfill_probe_diagrams(
    out: &mut LandscapeOutput,
    base_root: &Path,
    profile: SweepProfile,
    sweep_output_modes: bool,
) -> io::Result<()> {
    let fallback_root = PathBuf::from("configs/ecology_plateau80/medium_100k");
    let base_perceptron = load_config_named(base_root, "perceptron", &fallback_root)?;
    let base_mlp = load_config_named(base_root, "mlp", &fallback_root)?;
    let base_spatial = load_config_named(base_root, "spatial_graph", &fallback_root)?;
    let base_goo = load_config_named(base_root, "goo", &fallback_root)?;
    let mut spatial_bases = vec![base_spatial.clone()];
    if profile == SweepProfile::EvolutionHunt {
        let cardinal = config_path(base_root, "spatial_cardinal");
        if cardinal.is_file() {
            spatial_bases.push(load_config(&cardinal)?);
        }
    }
    let mut cells_to_sweep = build_cells_to_sweep(
        &base_perceptron,
        &base_mlp,
        &spatial_bases,
        &base_goo,
        profile,
    );
    if sweep_output_modes {
        cells_to_sweep = expand_output_mode_cells(cells_to_sweep);
    }
    let present: HashSet<(String, String, String)> = out
        .runs
        .iter()
        .map(|r| (r.architecture.clone(), r.x_label.clone(), r.y_label.clone()))
        .collect();
    cells_to_sweep.retain(|c| {
        present.contains(&(
            architecture_label(&c.cfg),
            c.x_label.clone(),
            c.y_label.clone(),
        ))
    });
    let diagram_specs: Vec<(Config, String, String, usize, u64)> = cells_to_sweep
        .iter()
        .map(|c| {
            (
                c.cfg.clone(),
                c.x_label.clone(),
                c.y_label.clone(),
                c.goo_generate_nodes,
                c.diagram_seed,
            )
        })
        .collect();
    out.cell_diagrams = build_cell_diagrams(&diagram_specs);
    println!(
        "backfilled probe data for {} cell diagram(s)",
        out.cell_diagrams.len()
    );
    Ok(())
}

fn regenerate_html_from_json(json_path: &Path, probe_opts: Option<(PathBuf, SweepProfile, bool)>) -> io::Result<()> {
    let raw = fs::read_to_string(json_path)?;
    let mut out: LandscapeOutput = serde_json::from_str(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    backfill_extinction_flags(&mut out);
    if !out.runs.is_empty() {
        out.cells = aggregate_cells(&out.runs);
    } else if out.cells.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "landscape JSON has no runs or cells",
        ));
    }
    if let Some((base_root, profile, sweep_output_modes)) = probe_opts {
        if cell_diagrams_need_probe_backfill(&out.cell_diagrams) {
            backfill_probe_diagrams(&mut out, &base_root, profile, sweep_output_modes)?;
        }
    }
    out.heatmaps = build_heatmaps_from_cells(&out.cells);
    out.complexity_help = complexity_help();
    out.fitness_help = fitness_help();
    out.architecture_labels = architecture_labels();
    let out_dir = json_path.parent().unwrap_or_else(|| Path::new("."));
    let db_path = out_dir.join("fitness_landscape.sqlite");
    write_sqlite_atomic(&db_path, &out, None)?;
    let html_path = out_dir.join("fitness_landscape_heatmap.html");
    write_html(&html_path, &out, None)?;
    println!("sqlite: {}", db_path.display());
    println!("html: {}", html_path.display());
    Ok(())
}

fn parse_sweep_profile(raw: &str) -> Option<SweepProfile> {
    match raw.to_ascii_lowercase().as_str() {
        "full" | "" => Some(SweepProfile::Full),
        "evolution_hunt" | "evolution-hunt" | "hunt" => Some(SweepProfile::EvolutionHunt),
        _ => None,
    }
}

fn load_landscape_for_merge(path: &Path) -> io::Result<LandscapeOutput> {
    let raw = fs::read_to_string(path)?;
    let mut val: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if val.get("fitness_help").is_none() {
        val["fitness_help"] = serde_json::to_value(fitness_help())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    if val.get("complexity_help").is_none() {
        val["complexity_help"] = serde_json::to_value(complexity_help())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    if val.get("architecture_labels").is_none() {
        val["architecture_labels"] = serde_json::to_value(architecture_labels())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    serde_json::from_value(val).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn merge_spatial_into_landscape(
    mut base: LandscapeOutput,
    spatial_runs: Vec<RunMetric>,
    spatial_diagrams: Vec<CellDiagram>,
    steps: i32,
    replicates: usize,
    founder_modes: &[FounderWeightMode],
) -> LandscapeOutput {
    base.runs.retain(|r| !is_spatial_architecture(&r.architecture));
    base.runs.extend(spatial_runs);
    base.cell_diagrams
        .retain(|d| !is_spatial_architecture(&d.architecture));
    base.cell_diagrams.extend(spatial_diagrams);
    base.cells = aggregate_cells(&base.runs);
    base.heatmaps = build_heatmaps_from_cells(&base.cells);
    base.steps = steps;
    base.replicates = replicates;
    base.founder_modes = founder_modes
        .iter()
        .map(|m| m.label().to_string())
        .collect();
    base.build_id = build_info::BUILD_ID.to_string();
    base.complexity_help = complexity_help();
    base.fitness_help = fitness_help();
    base.architecture_labels = architecture_labels();
    base
}

fn push_run(
    runs: &mut Vec<RunMetric>,
    cell: &SweepCell,
    steps: i32,
    seed: u32,
    cap_multiplier: f64,
    founder_mode: FounderWeightMode,
) {
    let (metric, _sim) = run_cell(
        cell.cfg.clone(),
        steps,
        seed,
        cap_multiplier,
        &cell.x_label,
        &cell.y_label,
        cell.goo_generate_nodes,
        founder_mode,
    );
    runs.push(metric);
}

#[allow(clippy::too_many_arguments)]
fn run_staged_sweep(
    out_dir: &Path,
    stages: &[i32],
    cells_to_sweep: &[SweepCell],
    cell_diagrams: &[CellDiagram],
    replicates: usize,
    cap_multiplier: f64,
    founder_modes: &[FounderWeightMode],
    founder_mode_labels: &[String],
    mut all_runs: Vec<RunMetric>,
    mut completed: HashSet<String>,
    start_replicate_idx: usize,
    start_stage_idx: usize,
) -> io::Result<()> {
    let total_cells = cells_to_sweep.len();
    let founder_count = founder_modes.len();
    let grand_total = total_cells * founder_count * replicates * stages.len();

    for rep in start_replicate_idx..replicates {
        for (stage_idx, &stage_steps) in stages.iter().enumerate() {
            if rep == start_replicate_idx && stage_idx < start_stage_idx {
                continue;
            }

            let prev_stage_steps = stage_idx.checked_sub(1).map(|i| stages[i]);
            let prev_stage_dir = prev_stage_steps
                .map(|s| out_dir.join(format!("stage_{s}")));
            let stage_dir = out_dir.join(format!("stage_{stage_steps}"));
            fs::create_dir_all(&stage_dir)?;

            let batch_total = total_cells * founder_count;
            let mut run_idx = 0usize;

            println!(
                "\n=== Replicate {}/{} · Stage {}/{}: {stage_steps} steps ({batch_total} runs) ===",
                rep + 1,
                replicates,
                stage_idx + 1,
                stages.len(),
            );

            let mut cell_idx = 0usize;
            for cell in cells_to_sweep {
                for (fi, founder_mode) in founder_modes.iter().enumerate() {
                    let arch = architecture_label(&cell.cfg);
                    let seed = replicate_seed(cell, cell_idx, fi, rep);
                    run_idx += 1;
                    let resume_key = run_resume_key_parts(
                        &arch,
                        &cell.x_label,
                        &cell.y_label,
                        founder_mode.label(),
                        seed,
                        stage_steps,
                    );
                    if completed.contains(&resume_key) {
                        println!(
                            "[{run_idx}/{batch_total}] skip (resume) rep={rep} {arch} {} {} founder={} seed={seed} @ {stage_steps}",
                            cell.x_label, cell.y_label, founder_mode.label()
                        );
                        continue;
                    }
                    let cont = if prev_stage_steps.is_some() {
                        "continue"
                    } else {
                        "fresh"
                    };
                    println!(
                        "[{run_idx}/{batch_total}] rep={rep} {cont} {arch} {} {} founder={} seed={seed} @ {stage_steps}",
                        cell.x_label, cell.y_label, founder_mode.label()
                    );
                    if push_staged_run(
                        &mut all_runs,
                        cell,
                        stage_steps,
                        prev_stage_steps,
                        prev_stage_dir.as_deref(),
                        seed,
                        cap_multiplier,
                        *founder_mode,
                        &stage_dir,
                    ) {
                        completed.insert(resume_key);
                        flush_staged_live(
                            out_dir,
                            &all_runs,
                            cell_diagrams,
                            cells_to_sweep,
                            stages,
                            replicates,
                            founder_mode_labels,
                            grand_total,
                        )?;
                        let checkpoint = StagedSweepState {
                            stages: stages.to_vec(),
                            current_replicate_idx: rep,
                            current_stage_idx: stage_idx,
                            all_runs: all_runs.clone(),
                        };
                        fs::write(
                            out_dir.join("staged_sweep_state.json"),
                            serde_json::to_string_pretty(&checkpoint)?,
                        )?;
                    }
                }
                cell_idx += 1;
            }

            let stage_runs = runs_at_stage(&all_runs, stage_steps);
            let stage_cells = aggregate_cells(&stage_runs);
            let promoted = select_promoted_cells(&stage_cells);
            write_stage_report(&stage_dir, stage_steps, &stage_cells, &promoted)?;

            let sweep_done = rep + 1 >= replicates && stage_idx + 1 >= stages.len();
            let stage_payload = LandscapeOutput {
                version: build_info::CRATE_VERSION.to_string(),
                build_id: build_info::BUILD_ID.to_string(),
                steps: stage_steps,
                replicates,
                founder_modes: founder_mode_labels.to_vec(),
                complexity_help: complexity_help(),
                fitness_help: fitness_help(),
                architecture_labels: architecture_labels(),
                runs: stage_runs.clone(),
                cells: stage_cells.clone(),
                heatmaps: heatmaps_for_scope(&stage_cells, cells_to_sweep, founder_mode_labels),
                cell_diagrams: cell_diagrams.to_vec(),
            };
            write_landscape_outputs(
                &stage_dir,
                &stage_payload,
                Some(LiveMeta {
                    status: if sweep_done { "complete" } else { "running" },
                    run_count: stage_runs.len(),
                    total_runs: stage_runs.len(),
                }),
            )?;
            println!(
                "stage {stage_steps} (rep {}): {} total runs at this stage, {} promising cells",
                rep + 1,
                stage_runs.len(),
                promoted.len()
            );

            let combined_runs = combined_latest_runs(&all_runs);
            let combined_cells = aggregate_cells(&combined_runs);
            let combined_payload = LandscapeOutput {
                version: build_info::CRATE_VERSION.to_string(),
                build_id: build_info::BUILD_ID.to_string(),
                steps: *stages.last().unwrap_or(&stage_steps),
                replicates,
                founder_modes: founder_mode_labels.to_vec(),
                complexity_help: complexity_help(),
                fitness_help: fitness_help(),
                architecture_labels: architecture_labels(),
                runs: combined_runs,
                cells: combined_cells.clone(),
                heatmaps: heatmaps_for_scope(&combined_cells, cells_to_sweep, founder_mode_labels),
                cell_diagrams: cell_diagrams.to_vec(),
            };
            write_landscape_outputs(
                out_dir,
                &combined_payload,
                Some(LiveMeta {
                    status: if sweep_done { "complete" } else { "running" },
                    run_count: all_runs.len(),
                    total_runs: grand_total,
                }),
            )?;

            let next_stage_idx = if stage_idx + 1 < stages.len() {
                stage_idx + 1
            } else {
                0
            };
            let next_replicate_idx = if stage_idx + 1 < stages.len() {
                rep
            } else {
                rep + 1
            };
            let state = StagedSweepState {
                stages: stages.to_vec(),
                current_replicate_idx: next_replicate_idx,
                current_stage_idx: next_stage_idx,
                all_runs: all_runs.clone(),
            };
            fs::write(
                out_dir.join("staged_sweep_state.json"),
                serde_json::to_string_pretty(&state)?,
            )?;

            if stage_idx + 1 < stages.len() {
                println!(
                    "Next: replicate {} @ {} steps (continuing from snapshots)",
                    rep + 1,
                    stages[stage_idx + 1]
                );
            } else if rep + 1 < replicates {
                println!(
                    "Replicate {} complete — starting replicate {} @ {} steps (fresh)",
                    rep + 1,
                    rep + 2,
                    stages[0]
                );
            }
        }
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let mut steps: i32 = 30_000;
    let mut replicates: usize = 3;
    let mut out_dir = PathBuf::from("output/fitness_landscape_medium");
    let mut base_root = PathBuf::from("configs/ecology_plateau80/medium_100k");
    let mut cap_multiplier = 1.8f64;
    let mut founder_modes = vec![FounderWeightMode::Neutral, FounderWeightMode::RandomUniform];
    let mut html_from_json: Option<PathBuf> = None;
    let mut refresh_skeleton = false;
    let mut arch_filter: Vec<SweepArch> = Vec::new();
    let mut sweep_output_modes = false;
    let mut sweep_profile = SweepProfile::Full;
    let mut staged_sweep = false;
    let mut sweep_stages: Option<Vec<i32>> = None;
    let mut merge_json: Option<PathBuf> = None;
    let mut resume_from: Option<PathBuf> = None;
    let mut resume_auto = false;

    for arg in env::args().skip(1) {
        if arg == "--help" || arg == "-h" {
            print_help();
            return Ok(());
        }
        if arg == "--resume" {
            resume_auto = true;
            continue;
        }
        if let Some(v) = parse_arg_value::<i32>(&arg, "--steps") {
            steps = v.max(1000);
            continue;
        }
        if let Some(v) = parse_arg_value::<usize>(&arg, "--replicates") {
            replicates = v.max(1);
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--output-dir") {
            out_dir = PathBuf::from(v);
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--base-config-root") {
            base_root = PathBuf::from(v);
            continue;
        }
        if let Some(v) = parse_arg_value::<f64>(&arg, "--cap-multiplier") {
            cap_multiplier = v.max(1.0);
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--founder-modes") {
            founder_modes = FounderWeightMode::parse_list(&v);
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--architectures") {
            arch_filter = parse_architectures(&v);
            continue;
        }
        if arg == "--sweep-output-modes" {
            sweep_output_modes = true;
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--sweep-profile") {
            sweep_profile = parse_sweep_profile(&v).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown --sweep-profile={v} (use full or evolution_hunt)"),
                )
            })?;
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--sweep-stages") {
            sweep_stages = Some(parse_sweep_stages(&v)?);
            continue;
        }
        if arg == "--staged-sweep" {
            staged_sweep = true;
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--merge-json") {
            merge_json = Some(PathBuf::from(v));
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--resume-from") {
            resume_from = Some(PathBuf::from(v));
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--html-from-json") {
            html_from_json = Some(PathBuf::from(v));
            continue;
        }
        if arg == "--refresh-skeleton" {
            refresh_skeleton = true;
            continue;
        }
    }

    if let Some(json_path) = html_from_json {
        let probe_opts = Some((base_root.clone(), sweep_profile, sweep_output_modes));
        return regenerate_html_from_json(&json_path, probe_opts);
    }

    fs::create_dir_all(&out_dir)?;

    let fallback_root = PathBuf::from("configs/ecology_plateau80/medium_100k");
    let base_perceptron = load_config_named(&base_root, "perceptron", &fallback_root)?;
    let base_mlp = load_config_named(&base_root, "mlp", &fallback_root)?;
    let base_spatial = load_config_named(&base_root, "spatial_graph", &fallback_root)?;
    let base_goo = load_config_named(&base_root, "goo", &fallback_root)?;

    let mut spatial_bases = vec![base_spatial.clone()];
    if sweep_profile == SweepProfile::EvolutionHunt {
        let cardinal = config_path(&base_root, "spatial_cardinal");
        if cardinal.is_file() {
            spatial_bases.push(load_config(&cardinal)?);
            println!("evolution_hunt: including spatial_cardinal layout");
        }
    }

    let mut cells_to_sweep = build_cells_to_sweep(
        &base_perceptron,
        &base_mlp,
        &spatial_bases,
        &base_goo,
        sweep_profile,
    );
    if !arch_filter.is_empty() {
        cells_to_sweep.retain(|c| cell_matches_arch_filter(c, &arch_filter));
    }
    if sweep_output_modes {
        cells_to_sweep = expand_output_mode_cells(cells_to_sweep);
    }
    println!(
        "sweep profile: {} · {} cells (after filters)",
        match sweep_profile {
            SweepProfile::Full => "full",
            SweepProfile::EvolutionHunt => "evolution_hunt",
        },
        cells_to_sweep.len()
    );
    if cells_to_sweep.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no sweep cells matched --architectures filter",
        ));
    }

    let diagram_specs: Vec<(Config, String, String, usize, u64)> = cells_to_sweep
        .iter()
        .map(|c| {
            (
                c.cfg.clone(),
                c.x_label.clone(),
                c.y_label.clone(),
                c.goo_generate_nodes,
                c.diagram_seed,
            )
        })
        .collect();
    let mut cell_diagrams = build_cell_diagrams(&diagram_specs);

    let mut runs: Vec<RunMetric> = Vec::new();
    let mut completed: HashSet<String> = HashSet::new();
    let ckpt_path = resume_from.or_else(|| {
        if resume_auto {
            let p = out_dir.join("fitness_landscape_checkpoint.json");
            if p.exists() {
                Some(p)
            } else {
                eprintln!("warning: --resume but no checkpoint at {}", p.display());
                None
            }
        } else {
            None
        }
    });
    if let Some(ref path) = ckpt_path {
        let ckpt = load_landscape_for_merge(path)?;
        if ckpt.steps != steps {
            eprintln!(
                "warning: checkpoint steps={} differs from --steps={steps}",
                ckpt.steps
            );
        }
        if ckpt.replicates != replicates {
            eprintln!(
                "warning: checkpoint replicates={} differs from --replicates={replicates}",
                ckpt.replicates
            );
        }
        println!(
            "resume: {} completed runs from {}",
            ckpt.runs.len(),
            path.display()
        );
        for r in &ckpt.runs {
            completed.insert(run_resume_key(r));
        }
        runs = ckpt.runs;
        if !ckpt.cell_diagrams.is_empty() {
            cell_diagrams = ckpt.cell_diagrams;
        }
    }

    let founder_mode_labels: Vec<String> = founder_modes.iter().map(|m| m.label().to_string()).collect();

    if refresh_skeleton {
        let stages = sweep_stages.clone().unwrap_or_else(|| DEFAULT_SWEEP_STAGES.to_vec());
        let grand_total = cells_to_sweep.len() * founder_modes.len() * replicates * stages.len();
        let mut all_runs = runs;
        if let Ok(raw) = fs::read_to_string(out_dir.join("staged_sweep_state.json")) {
            if let Ok(state) = serde_json::from_str::<StagedSweepState>(&raw) {
                all_runs = state.all_runs;
                println!(
                    "refresh skeleton: loaded {} runs from staged_sweep_state.json",
                    all_runs.len()
                );
            }
        }
        flush_staged_live(
            &out_dir,
            &all_runs,
            &cell_diagrams,
            &cells_to_sweep,
            &stages,
            replicates,
            &founder_mode_labels,
            grand_total,
        )?;
        let combined = combined_latest_runs(&all_runs);
        let cells = aggregate_cells(&combined);
        let hm_count =
            heatmaps_for_scope(&cells, &cells_to_sweep, &founder_mode_labels).len();
        println!(
            "refreshed live viewer: {} architecture tables · {} sweep cells · {} runs in state",
            hm_count,
            cells_to_sweep.len(),
            all_runs.len()
        );
        return Ok(());
    }

    if staged_sweep {
        let stages = sweep_stages.unwrap_or_else(|| DEFAULT_SWEEP_STAGES.to_vec());
        let mut start_replicate_idx = 0usize;
        let mut start_stage_idx = 0usize;
        let mut all_runs = runs;
        for r in &mut all_runs {
            if r.run_steps == 0 {
                r.run_steps = steps;
            }
        }

        let state_path = out_dir.join("staged_sweep_state.json");
        if resume_auto && state_path.exists() {
            let state: StagedSweepState =
                serde_json::from_str(&fs::read_to_string(&state_path)?).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, e)
                })?;
            all_runs = state.all_runs;
            start_replicate_idx = state.current_replicate_idx.min(replicates.saturating_sub(1));
            start_stage_idx = state.current_stage_idx.min(stages.len().saturating_sub(1));
            completed.clear();
            for r in &all_runs {
                completed.insert(run_resume_key(r));
            }
            println!(
                "resume staged sweep: {} runs, starting replicate {} stage {} ({} steps)",
                all_runs.len(),
                start_replicate_idx + 1,
                start_stage_idx + 1,
                stages.get(start_stage_idx).copied().unwrap_or(0)
            );
        } else if resume_auto {
            for &stage_steps in &stages {
                let partial_path = out_dir
                    .join(format!("stage_{stage_steps}"))
                    .join("fitness_landscape.json");
                if !partial_path.is_file() {
                    continue;
                }
                let partial: LandscapeOutput =
                    serde_json::from_str(&fs::read_to_string(&partial_path)?).map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, e)
                    })?;
                for r in partial.runs {
                    completed.insert(run_resume_key(&r));
                    all_runs.retain(|x| run_resume_key(x) != run_resume_key(&r));
                    all_runs.push(r);
                }
                println!(
                    "resume partial stage {stage_steps}: {} runs loaded",
                    partial_path.display()
                );
            }
        }

        println!(
            "staged sweep: {:?} · replicate-first (each rep: full ladder from snapshots)",
            stages
        );
        let grand_total = cells_to_sweep.len() * founder_modes.len() * replicates * stages.len();
        flush_staged_live(
            &out_dir,
            &all_runs,
            &cell_diagrams,
            &cells_to_sweep,
            &stages,
            replicates,
            &founder_mode_labels,
            grand_total,
        )?;
        let html_path = out_dir.join("fitness_landscape_heatmap.html");
        println!("html (live): {}", html_path.display());
        println!(
            "sqlite (live): {} · {} / {} runs in DB so far",
            out_dir.join("fitness_landscape.sqlite").display(),
            all_runs.len(),
            grand_total
        );
        run_staged_sweep(
            &out_dir,
            &stages,
            &cells_to_sweep,
            &cell_diagrams,
            replicates,
            cap_multiplier,
            &founder_modes,
            &founder_mode_labels,
            all_runs,
            completed,
            start_replicate_idx,
            start_stage_idx,
        )?;
        println!("staged sweep complete");
        println!("combined outputs: {}", out_dir.display());
        return Ok(());
    }

    let total_runs = cells_to_sweep.len() * founder_modes.len() * replicates;
    println!(
        "sweep: {} cells × {} founder modes × {} replicates = {} runs @ {steps} steps",
        cells_to_sweep.len(),
        founder_modes.len(),
        replicates,
        total_runs
    );

    let html_path = out_dir.join("fitness_landscape_heatmap.html");
    flush_live_snapshot(
        &out_dir,
        &runs,
        &cell_diagrams,
        steps,
        replicates,
        &founder_mode_labels,
        Some(&cells_to_sweep),
        LiveMeta {
            status: "running",
            run_count: runs.len(),
            total_runs,
        },
    )?;
    println!("html (live): {}", html_path.display());
    println!("sqlite (live): {}", out_dir.join("fitness_landscape.sqlite").display());

    let mut runs = runs;
    let mut cell_idx = 0usize;
    let mut run_idx = 0usize;
    for cell in &cells_to_sweep {
        for (fi, founder_mode) in founder_modes.iter().enumerate() {
            for r in 0..replicates {
                let seed = cell
                    .cfg
                    .seed
                    .wrapping_add((cell_idx * 101 + fi * 1009 + r) as u32);
                run_idx += 1;
                let arch = architecture_label(&cell.cfg);
                let resume_key = run_resume_key_parts(
                    &arch,
                    &cell.x_label,
                    &cell.y_label,
                    founder_mode.label(),
                    seed,
                    steps,
                );
                if completed.contains(&resume_key) {
                    println!(
                        "[{run_idx}/{total_runs}] skip (resume) {arch} {} {} founder={} seed={seed}",
                        cell.x_label,
                        cell.y_label,
                        founder_mode.label()
                    );
                    continue;
                }
                println!(
                    "[{run_idx}/{total_runs}] {arch} {} {} founder={} seed={seed}",
                    cell.x_label,
                    cell.y_label,
                    founder_mode.label()
                );
                push_run(
                    &mut runs,
                    cell,
                    steps,
                    seed,
                    cap_multiplier,
                    *founder_mode,
                );
                flush_live_snapshot(
                    &out_dir,
                    &runs,
                    &cell_diagrams,
                    steps,
                    replicates,
                    &founder_mode_labels,
                    Some(&cells_to_sweep),
                    LiveMeta {
                        status: "running",
                        run_count: runs.len(),
                        total_runs,
                    },
                )?;
                if run_idx % 10 == 0 || run_idx == total_runs {
                    let cells_partial = aggregate_cells(&runs);
                    let checkpoint = LandscapeOutput {
                        version: build_info::CRATE_VERSION.to_string(),
                        build_id: build_info::BUILD_ID.to_string(),
                        steps,
                        replicates,
                        founder_modes: founder_modes.iter().map(|m| m.label().to_string()).collect(),
                        complexity_help: complexity_help(),
                        fitness_help: fitness_help(),
                        architecture_labels: architecture_labels(),
                        runs: runs.clone(),
                        cells: cells_partial,
                        heatmaps: Vec::new(),
                        cell_diagrams: cell_diagrams.clone(),
                    };
                    let ckpt_path = out_dir.join("fitness_landscape_checkpoint.json");
                    if let Ok(json) = serde_json::to_string_pretty(&checkpoint) {
                        let _ = fs::write(&ckpt_path, json);
                        println!(
                            "checkpoint: {} runs saved -> {}",
                            runs.len(),
                            ckpt_path.display()
                        );
                    }
                }
            }
        }
        cell_idx += 1;
    }

    let cells = aggregate_cells(&runs);
    let heatmaps = build_heatmaps_from_cells(&cells);
    let payload = if let Some(merge_path) = merge_json {
        let base = load_landscape_for_merge(&merge_path)?;
        println!("merging {} new runs into {}", runs.len(), merge_path.display());
        merge_spatial_into_landscape(
            base,
            runs,
            cell_diagrams,
            steps,
            replicates,
            &founder_modes,
        )
    } else {
        LandscapeOutput {
            version: build_info::CRATE_VERSION.to_string(),
            build_id: build_info::BUILD_ID.to_string(),
            steps,
            replicates,
            founder_modes: founder_mode_labels.clone(),
            complexity_help: complexity_help(),
            fitness_help: fitness_help(),
            architecture_labels: architecture_labels(),
            runs,
            cells: cells.clone(),
            heatmaps,
            cell_diagrams,
        }
    };
    let cells = payload.cells.clone();

    let json_path = out_dir.join("fitness_landscape.json");
    let csv_path = out_dir.join("fitness_landscape_cells.csv");
    let db_path = out_dir.join("fitness_landscape.sqlite");
    let html_path = out_dir.join("fitness_landscape_heatmap.html");

    fs::write(&json_path, serde_json::to_string_pretty(&payload).unwrap())?;
    write_csv(&csv_path, &cells)?;
    write_sqlite_atomic(
        &db_path,
        &payload,
        Some(LiveMeta {
            status: "complete",
            run_count: payload.runs.len(),
            total_runs,
        }),
    )?;
    write_html(
        &html_path,
        &payload,
        Some(LiveMeta {
            status: "complete",
            run_count: payload.runs.len(),
            total_runs,
        }),
    )?;

    println!("evo-fitness-landscape {}", build_info::VERSION_LABEL);
    println!("runs: {}", payload.runs.len());
    println!("founder modes: {}", payload.founder_modes.join(", "));
    println!("json: {}", json_path.display());
    println!("csv: {}", csv_path.display());
    println!("sqlite: {}", db_path.display());
    println!("html: {}", html_path.display());
    Ok(())
}
