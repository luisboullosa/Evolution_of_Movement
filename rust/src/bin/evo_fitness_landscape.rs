use evo_movement_rust::brain::LabStructuralOp;
use evo_movement_rust::brain::{BrainGenome, FounderWeightMode};
use evo_movement_rust::brain_diagram::{diagram_from_brain, BrainDiagram};
use evo_movement_rust::build_info;
use evo_movement_rust::config::{ArchitecturePolicy, Config};
use evo_movement_rust::simulation::Simulation;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrajectoryPoint {
    step: i32,
    fitness: f64,
    pop_ratio: f64,
    avg_complexity: f64,
    population: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunMetric {
    architecture: String,
    x_label: String,
    y_label: String,
    founder_mode: String,
    seed: u32,
    steps_completed: i32,
    final_population: usize,
    max_population: i32,
    pop_ratio: f64,
    avg_resources: f64,
    avg_complexity: f64,
    prop_moved_final: f64,
    fitness_score: f64,
    trajectory: Vec<TrajectoryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellAggregate {
    architecture: String,
    x_label: String,
    y_label: String,
    founder_mode: String,
    replicates: usize,
    fitness_mean: f64,
    fitness_std: f64,
    pop_ratio_mean: f64,
    complexity_mean: f64,
    trajectory_mean: Vec<TrajectoryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HeatmapData {
    architecture: String,
    founder_mode: String,
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    fitness_matrix: Vec<Vec<Option<f64>>>,
    pop_ratio_matrix: Vec<Vec<Option<f64>>>,
    complexity_matrix: Vec<Vec<Option<f64>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellDiagram {
    architecture: String,
    x_label: String,
    y_label: String,
    structure_signature: String,
    diagram: BrainDiagram,
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
    println!("  --merge-json=PATH           Merge new runs into existing JSON (replaces matching architectures)");
    println!("  --resume-from=PATH          Continue sweep from checkpoint JSON (skips completed runs)");
    println!("  --resume                    Resume from OUTPUT_DIR/fitness_landscape_checkpoint.json");
    println!("  --html-from-json=PATH       Regenerate HTML dashboard from existing JSON (no sims)");
    println!("  --help, -h                  Show help");
}

fn load_config(path: &Path) -> io::Result<Config> {
    Config::from_file(path)
}

fn config_path(base_root: &Path, name: &str) -> PathBuf {
    base_root.join(format!("{name}.json"))
}

fn run_resume_key(r: &RunMetric) -> String {
    format!(
        "{}|{}|{}|{}|{}",
        r.architecture, r.x_label, r.y_label, r.founder_mode, r.seed
    )
}

fn run_resume_key_parts(
    architecture: &str,
    x_label: &str,
    y_label: &str,
    founder_mode: &str,
    seed: u32,
) -> String {
    format!("{architecture}|{x_label}|{y_label}|{founder_mode}|{seed}")
}

fn architecture_label(cfg: &Config) -> String {
    match cfg.architecture_policy {
        ArchitecturePolicy::EvolveSpatialGraph if !cfg.uses_legacy_spatial_graph() => {
            "Spatial macromolecule blob".to_string()
        }
        other => other.label().to_string(),
    }
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
    cfg.architecture_mutation_rate = 0.0;
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

fn seed_goo_on_brain(brain: &mut BrainGenome, cfg: &Config, generate_nodes: usize, seed: u64) {
    if cfg.architecture_policy != ArchitecturePolicy::EvolveGoo || generate_nodes <= 1 {
        return;
    }
    let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FFEE_u64);
    for _ in 1..generate_nodes {
        let _ = evo_movement_rust::brain::apply_lab_structural(
            brain,
            LabStructuralOp::GooAddGenerate,
            cfg,
            &mut rng,
        );
    }
}

fn seed_goo_founders(sim: &mut Simulation, cfg: &Config, generate_nodes: usize, seed: u64) {
    if cfg.architecture_policy != ArchitecturePolicy::EvolveGoo || generate_nodes <= 1 {
        return;
    }
    let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FFEE_u64);
    for ind in sim.individuals.iter_mut() {
        for _ in 1..generate_nodes {
            let _ = evo_movement_rust::brain::apply_lab_structural(
                &mut ind.brain,
                LabStructuralOp::GooAddGenerate,
                cfg,
                &mut rng,
            );
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
) -> RunMetric {
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

    RunMetric {
        architecture: arch,
        x_label: x_label.to_string(),
        y_label: y_label.to_string(),
        founder_mode: founder_mode.label().to_string(),
        seed,
        steps_completed,
        final_population,
        max_population,
        pop_ratio,
        avg_resources,
        avg_complexity,
        prop_moved_final,
        fitness_score: fitness,
        trajectory,
    }
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
                trajectory_mean: average_trajectories(&vals),
            }
        })
        .collect()
}

fn build_heatmaps(cells: &[CellAggregate]) -> Vec<HeatmapData> {
    let mut by_key: BTreeMap<(String, String), Vec<&CellAggregate>> = BTreeMap::new();
    for c in cells {
        by_key
            .entry((c.architecture.clone(), c.founder_mode.clone()))
            .or_default()
            .push(c);
    }
    let mut out = Vec::new();
    for ((arch, founder), vals) in by_key {
        let mut x_labels: Vec<String> = vals.iter().map(|v| v.x_label.clone()).collect();
        x_labels.sort();
        x_labels.dedup();
        let mut y_labels: Vec<String> = vals.iter().map(|v| v.y_label.clone()).collect();
        y_labels.sort();
        y_labels.dedup();

        let mut fit = vec![vec![None; x_labels.len()]; y_labels.len()];
        let mut pop = vec![vec![None; x_labels.len()]; y_labels.len()];
        let mut cplx = vec![vec![None; x_labels.len()]; y_labels.len()];

        for v in vals {
            let xi = x_labels.iter().position(|x| x == &v.x_label).unwrap();
            let yi = y_labels.iter().position(|y| y == &v.y_label).unwrap();
            fit[yi][xi] = Some(v.fitness_mean);
            pop[yi][xi] = Some(v.pop_ratio_mean);
            cplx[yi][xi] = Some(v.complexity_mean);
        }

        out.push(HeatmapData {
            architecture: arch,
            founder_mode: founder,
            x_labels,
            y_labels,
            fitness_matrix: fit,
            pop_ratio_matrix: pop,
            complexity_matrix: cplx,
        });
    }
    out
}

fn build_cell_diagrams(
    specs: &[(Config, String, String, usize, u64)],
) -> Vec<CellDiagram> {
    let mut out = Vec::new();
    for (cfg, x_label, y_label, goo_gen, seed) in specs {
        let brain = representative_brain(cfg, *goo_gen, *seed);
        let arch = architecture_label(cfg);
        out.push(CellDiagram {
            architecture: arch,
            x_label: x_label.clone(),
            y_label: y_label.clone(),
            structure_signature: brain.structure_signature(),
            diagram: diagram_from_brain(&brain, 360.0, 200.0),
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
        "architecture,x_label,y_label,founder_mode,replicates,fitness_mean,fitness_std,pop_ratio_mean,complexity_mean\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}\n",
            r.architecture,
            r.x_label,
            r.y_label,
            r.founder_mode,
            r.replicates,
            r.fitness_mean,
            r.fitness_std,
            r.pop_ratio_mean,
            r.complexity_mean
        ));
    }
    fs::write(path, out)
}

const VIEWER_TEMPLATE: &str = include_str!("../../assets/fitness_landscape_viewer.html");

fn write_html(path: &Path, out: &LandscapeOutput) -> io::Result<()> {
    let mut json = serde_json::to_string(out)?;
    json = json.replace("</", "<\\/");
    let html = VIEWER_TEMPLATE.replace("__LANDSCAPE_JSON__", &json);
    fs::write(path, html)
}

fn regenerate_html_from_json(json_path: &Path) -> io::Result<()> {
    let raw = fs::read_to_string(json_path)?;
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
    let mut json = serde_json::to_string(&val)?;
    json = json.replace("</", "<\\/");
    let html_path = json_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("fitness_landscape_heatmap.html");
    let html = VIEWER_TEMPLATE.replace("__LANDSCAPE_JSON__", &json);
    fs::write(&html_path, html)?;
    println!("html: {}", html_path.display());
    Ok(())
}

struct SweepCell {
    cfg: Config,
    x_label: String,
    y_label: String,
    goo_generate_nodes: usize,
    diagram_seed: u64,
}

fn build_cells_to_sweep(
    base_perceptron: &Config,
    base_mlp: &Config,
    base_spatial: &Config,
    base_goo: &Config,
) -> Vec<SweepCell> {
    let mut cells_to_sweep: Vec<SweepCell> = Vec::new();

    let perceptron_nodes = [1, 2, 4, 8, 12, 16, 24, 32];
    for (xi, n) in perceptron_nodes.iter().enumerate() {
        let mut cfg = base_perceptron.clone();
        cfg.nn_nodes = *n;
        cells_to_sweep.push(SweepCell {
            cfg,
            x_label: format!("nodes={n}"),
            y_label: "baseline".to_string(),
            goo_generate_nodes: 1,
            diagram_seed: xi as u64,
        });
    }

    let mlp_nodes = [1, 2, 4, 8, 12];
    let mlp_nodes2 = [0, 2, 4, 8];
    for (xi, n1) in mlp_nodes.iter().enumerate() {
        for (yi, n2) in mlp_nodes2.iter().enumerate() {
            let mut cfg = base_mlp.clone();
            cfg.nn_nodes = *n1;
            if *n2 > 0 {
                cfg.nn_layers = 2;
                cfg.nn_nodes2 = *n2;
            } else {
                cfg.nn_layers = 1;
                cfg.nn_nodes2 = 0;
            }
            cells_to_sweep.push(SweepCell {
                cfg,
                x_label: format!("L1={n1}"),
                y_label: format!("L2={n2}"),
                goo_generate_nodes: 1,
                diagram_seed: (xi * 17 + yi) as u64,
            });
        }
    }

    let spatial_layers = [1, 2, 4, 8, 12, 16];
    let spatial_caps = [16, 24, 32, 48];
    for (xi, layers) in spatial_layers.iter().enumerate() {
        for (yi, cap) in spatial_caps.iter().enumerate() {
            let mut cfg = base_spatial.clone();
            cfg.nn_nodes = *layers;
            cfg.max_brain_nodes = *cap;
            let x_label = if cfg.uses_legacy_spatial_graph() {
                format!("landmarks={layers}")
            } else {
                format!("layers={layers}")
            };
            cells_to_sweep.push(SweepCell {
                cfg,
                x_label,
                y_label: format!("max_brain_nodes={cap}"),
                goo_generate_nodes: 1,
                diagram_seed: (xi * 23 + yi) as u64,
            });
        }
    }

    let goo_gen_nodes = [1, 2, 4, 8, 12];
    let goo_caps = [16, 24, 32, 48];
    for (xi, gn) in goo_gen_nodes.iter().enumerate() {
        for (yi, cap) in goo_caps.iter().enumerate() {
            let mut cfg = base_goo.clone();
            cfg.max_brain_nodes = *cap;
            cells_to_sweep.push(SweepCell {
                cfg,
                x_label: format!("generate_nodes={gn}"),
                y_label: format!("max_brain_nodes={cap}"),
                goo_generate_nodes: *gn as usize,
                diagram_seed: (xi * 29 + yi) as u64,
            });
        }
    }

    cells_to_sweep
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
    base.heatmaps = build_heatmaps(&base.cells);
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
    runs.push(run_cell(
        cell.cfg.clone(),
        steps,
        seed,
        cap_multiplier,
        &cell.x_label,
        &cell.y_label,
        cell.goo_generate_nodes,
        founder_mode,
    ));
}

fn main() -> io::Result<()> {
    let mut steps: i32 = 30_000;
    let mut replicates: usize = 3;
    let mut out_dir = PathBuf::from("output/fitness_landscape_medium");
    let mut base_root = PathBuf::from("configs/ecology_plateau80/medium_100k");
    let mut cap_multiplier = 1.8f64;
    let mut founder_modes = vec![FounderWeightMode::Neutral, FounderWeightMode::RandomUniform];
    let mut html_from_json: Option<PathBuf> = None;
    let mut arch_filter: Vec<SweepArch> = Vec::new();
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
    }

    if let Some(json_path) = html_from_json {
        return regenerate_html_from_json(&json_path);
    }

    fs::create_dir_all(&out_dir)?;

    let base_perceptron = load_config(&config_path(&base_root, "perceptron"))?;
    let base_mlp = load_config(&config_path(&base_root, "mlp"))?;
    let base_spatial = load_config(&config_path(&base_root, "spatial_graph"))?;
    let base_goo = load_config(&config_path(&base_root, "goo"))?;

    let mut cells_to_sweep =
        build_cells_to_sweep(&base_perceptron, &base_mlp, &base_spatial, &base_goo);
    if !arch_filter.is_empty() {
        cells_to_sweep.retain(|c| cell_matches_arch_filter(c, &arch_filter));
    }
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

    let total_runs = cells_to_sweep.len() * founder_modes.len() * replicates;
    println!(
        "sweep: {} cells × {} founder modes × {} replicates = {} runs",
        cells_to_sweep.len(),
        founder_modes.len(),
        replicates,
        total_runs
    );

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
    let heatmaps = build_heatmaps(&cells);
    let founder_mode_labels: Vec<String> = founder_modes.iter().map(|m| m.label().to_string()).collect();
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
            founder_modes: founder_mode_labels,
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
    let html_path = out_dir.join("fitness_landscape_heatmap.html");

    fs::write(&json_path, serde_json::to_string_pretty(&payload).unwrap())?;
    write_csv(&csv_path, &cells)?;
    write_html(&html_path, &payload)?;

    println!("evo-fitness-landscape {}", build_info::VERSION_LABEL);
    println!("runs: {}", payload.runs.len());
    println!("founder modes: {}", payload.founder_modes.join(", "));
    println!("json: {}", json_path.display());
    println!("csv: {}", csv_path.display());
    println!("html: {}", html_path.display());
    Ok(())
}
