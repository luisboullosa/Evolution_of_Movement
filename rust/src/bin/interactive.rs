/// Interactive evolution simulation viewer with GUI
use eframe::egui;
#[cfg(feature = "cuda")]
use evo_movement_rust::gpu_resource::GpuResourceUpdater;
use evo_movement_rust::{
    brain::{BrainGenome, PERCEPTRON_SENSOR_DIM},
    build_info,
    cladogram::Cladogram,
  config::{ArchitecturePolicy, Config, GridTopology, MovementSpeed},
    ecology_presets,
    hex_grid::{HexCell, HexCoord, HexGrid},
    individual::Individual,
    neural_network::NeuralNetwork,
    nn_viz,
    perlin::Perlin,
    population_dynamics::{apply_metabolism_and_cull, run_reproduction_phase},
    save_state::SimulationState,
    simulation::Simulation,
    snapshot::{PopulationSnapshot, SnapshotError, SnapshotIndividual},
    snapshot_registry::{self, SnapshotManifest},
    step_engine::{
        forward_decision, gather_amount_for_decision, move_allowed, start_move_cooldown,
        tick_move_wait,
    },
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Default axial radius for new interactive dishes (1 + 3r(r+1) cells; r=30 → 931 cells).
const DEFAULT_DISH_RADIUS_CELLS: i32 = 30;
/// Batch configs with larger dish radii are not used for fresh interactive sessions.
const INTERACTIVE_MAX_DISH_RADIUS: i32 = 100;
const REWIND_CAP: usize = 60;
const TRAIL_MAX_AGE: u8 = 10;
const TARGET_FPS: f32 = 24.0;

const INTERACTIVE_SESSIONS_DIR: &str = "output/interactive_sessions";

fn compute_session_id(seed: u32, snapshot_paths: &[PathBuf]) -> String {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    build_info::BUILD_ID.hash(&mut hasher);
    unix_now_secs().hash(&mut hasher);
    for path in snapshot_paths {
        path.to_string_lossy().hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Serialize)]
struct InteractiveSessionManifest {
    session_id: String,
    started_at_unix_secs: u64,
    build_id: String,
    version_label: String,
    seed: u32,
    architecture_policy: String,
    source_snapshots: Vec<String>,
    export_dir: String,
}

#[derive(Serialize)]
struct ExportedIndividualRecord {
    session_id: String,
    exported_at_unix_secs: u64,
    individual_id: i32,
    resources: f64,
    age: i32,
    position: usize,
    genotype: String,
    brain_kind: String,
    brain: BrainGenome,
}

fn init_interactive_session(
    seed: u32,
    config: &Config,
    snapshot_paths: &[PathBuf],
) -> (String, PathBuf) {
    let session_id = compute_session_id(seed, snapshot_paths);
    let session_dir = PathBuf::from(INTERACTIVE_SESSIONS_DIR).join(&session_id);
    let individuals_dir = session_dir.join("individuals");
    let _ = fs::create_dir_all(&individuals_dir);

    let manifest = InteractiveSessionManifest {
        session_id: session_id.clone(),
        started_at_unix_secs: unix_now_secs(),
        build_id: build_info::BUILD_ID.to_string(),
        version_label: build_info::VERSION_LABEL.to_string(),
        seed,
        architecture_policy: config.architecture_policy.label().to_string(),
        source_snapshots: snapshot_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
        export_dir: session_dir.to_string_lossy().into_owned(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&manifest) {
        let _ = fs::write(session_dir.join("session.json"), json);
    }

    println!(
        "[interactive] session {} -> {}",
        session_id,
        session_dir.display()
    );
    (session_id, session_dir)
}

fn export_individual_brain_json(
    session_id: &str,
    session_dir: &Path,
    ind: &Individual,
) -> Result<PathBuf, String> {
    let genotype = genotype_signature(ind);
    let record = ExportedIndividualRecord {
        session_id: session_id.to_string(),
        exported_at_unix_secs: unix_now_secs(),
        individual_id: ind.id,
        resources: ind.resources,
        age: ind.age,
        position: ind.position,
        genotype,
        brain_kind: ind.brain.kind().as_str().to_string(),
        brain: ind.brain.clone(),
    };
    let path = session_dir
        .join("individuals")
        .join(format!("individual_{}.json", ind.id));
    let json = serde_json::to_string_pretty(&record)
        .map_err(|e| format!("serialize brain json: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

fn reveal_in_file_manager(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer")
            .arg("/select,")
            .arg(path)
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(parent) = path.parent() {
            let _ = std::process::Command::new("xdg-open").arg(parent).spawn();
        }
    }
}

fn open_session_folder(session_dir: &Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer").arg(session_dir).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(session_dir).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(session_dir).spawn();
    }
}

fn draw_brain_json_export_ui(ui: &mut egui::Ui, session_id: &str, path: Option<&PathBuf>) {
    ui.label(egui::RichText::new("Session").strong());
    ui.label(
        egui::RichText::new(session_id)
            .monospace()
            .weak(),
    );
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Brain JSON export").strong());
    if let Some(path) = path {
        let display = path.to_string_lossy();
        ui.label(
            egui::RichText::new(display.as_ref())
                .monospace()
                .small(),
        );
        ui.horizontal(|ui| {
            if ui.button("Reveal in Explorer").clicked() {
                reveal_in_file_manager(path);
            }
            if ui.button("Copy path").clicked() {
                ui.ctx().copy_text(display.into_owned());
            }
        });
    } else {
        ui.label(
            egui::RichText::new("Select an individual to export BrainGenome JSON.")
                .weak()
                .italics(),
        );
    }
}

fn viewer_window_title() -> String {
    format!(
        "Evolution of Movement - Interactive Viewer {}",
        build_info::VERSION_LABEL
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HexBackgroundMode {
    Resources,
    Competition,
}

fn max_cell_occupancy(cells: &std::collections::HashMap<HexCoord, HexCell>) -> usize {
    cells
        .values()
        .map(|c| c.individual_indices.len())
        .max()
        .unwrap_or(0)
}

fn competition_heatmap_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.01 {
        return egui::Color32::from_rgb(26, 30, 38);
    }
    // Cool (empty) → warm (crowded) competition map
    let r = (40.0 + t * 215.0) as u8;
    let g = (50.0 + (1.0 - t) * 120.0 + t * 40.0) as u8;
    let b = (70.0 + (1.0 - t) * 100.0) as u8;
    egui::Color32::from_rgb(r, g, b)
}

fn hex_background_color(
    mode: HexBackgroundMode,
    resource_level: f64,
    occupant_count: usize,
    max_occupants: usize,
) -> egui::Color32 {
    match mode {
        HexBackgroundMode::Resources => {
            let brightness = (resource_level * 255.0).clamp(0.0, 255.0) as u8;
            egui::Color32::from_rgb(brightness, brightness, 100)
        }
        HexBackgroundMode::Competition => {
            let t = if max_occupants > 0 {
                occupant_count as f32 / max_occupants as f32
            } else {
                0.0
            };
            competition_heatmap_color(t)
        }
    }
}

#[derive(Debug, Clone)]
struct HeadlessGenotypeStats {
    pop_rank: usize,
    pop_share: f64,
    resource_rank: usize,
    resource_share: f64,
    total_genotypes: usize,
}

#[derive(Debug, Clone)]
struct GenotypeLineageRow {
    signature: String,
    color: [u8; 3],
    pop: usize,
    movers_pct: Option<f64>,
    member_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct MoveTrail {
    from_pos: usize,
    to_pos: usize,
    color: [u8; 3],
    age: u8,
}

#[derive(Debug, Clone)]
struct DishCheckpoint {
    timestep: i32,
    perlin_time: f64,
    env_resources: Vec<f64>,
    individuals: Vec<Individual>,
    next_id: i32,
    cladogram: Cladogram,
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

fn genotype_signature(ind: &Individual) -> String {
    ind.brain.structure_signature()
}

fn genotype_signature_from_snapshot(rec: &SnapshotIndividual) -> String {
    if let Some(ref json) = rec.brain_json {
        if let Some(sig) = BrainGenome::structure_signature_from_brain_json(json) {
            return sig;
        }
    }
    if rec.nn_layers >= 2 || rec.weights2.iter().any(|&w| w != 0.0) {
        format!("m:{}:{}:{}:{}", rec.nn_nodes, rec.nn_inputs, rec.nn_nodes2, rec.nn_layers)
    } else {
        format!("p:{}:{}", rec.nn_inputs, rec.nn_nodes.max(1))
    }
}

fn genotype_color(sig: &str) -> [u8; 3] {
    let mut hasher = DefaultHasher::new();
    sig.hash(&mut hasher);
    let hue = (hasher.finish() % 360) as f32;
    hsv_to_rgb(hue, 0.7, 0.95)
}

fn assign_genotype_colors(individuals: &mut [Individual]) {
    for ind in individuals.iter_mut() {
        let sig = genotype_signature(ind);
        ind.lineage_color = genotype_color(&sig);
    }
}

fn rank_descending(values: &[(String, f64)]) -> HashMap<String, usize> {
    let mut sorted: Vec<(String, f64)> = values.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = HashMap::new();
    let mut rank = 0usize;
    let mut prev: Option<f64> = None;
    for (sig, val) in sorted {
        if prev.map(|p| (p - val).abs() < f64::EPSILON) != Some(true) {
            rank += 1;
        }
        ranks.insert(sig, rank);
        prev = Some(val);
    }
    ranks
}

fn compute_headless_genotype_stats(records: &[SnapshotIndividual]) -> HashMap<String, HeadlessGenotypeStats> {
    let mut pop_counts: HashMap<String, usize> = HashMap::new();
    let mut resource_sums: HashMap<String, f64> = HashMap::new();
    let total_pop = records.len().max(1);
    let mut total_resources = 0.0f64;

    for rec in records {
        let sig = genotype_signature_from_snapshot(rec);
        *pop_counts.entry(sig.clone()).or_insert(0) += 1;
        let res = rec.resources as f64;
        *resource_sums.entry(sig).or_insert(0.0) += res;
        total_resources += res;
    }

    let total_genotypes = pop_counts.len().max(1);
    let total_resources = total_resources.max(f64::EPSILON);

    let pop_values: Vec<(String, f64)> = pop_counts
        .iter()
        .map(|(k, &v)| (k.clone(), v as f64))
        .collect();
    let res_values: Vec<(String, f64)> = resource_sums
        .iter()
        .map(|(k, &v)| (k.clone(), v))
        .collect();
    let pop_ranks = rank_descending(&pop_values);
    let res_ranks = rank_descending(&res_values);

    pop_counts
        .keys()
        .map(|sig| {
            let pop_count = *pop_counts.get(sig).unwrap_or(&0);
            let pop_share = pop_count as f64 / total_pop as f64 * 100.0;
            let resource_sum = *resource_sums.get(sig).unwrap_or(&0.0);
            let resource_share = resource_sum / total_resources * 100.0;
            (
                sig.clone(),
                HeadlessGenotypeStats {
                    pop_rank: *pop_ranks.get(sig).unwrap_or(&total_genotypes),
                    pop_share,
                    resource_rank: *res_ranks.get(sig).unwrap_or(&total_genotypes),
                    resource_share,
                    total_genotypes,
                },
            )
        })
        .collect()
}

fn interactive_default_config() -> Config {
    let mut cfg = Config::default();
    cfg.grid_topology = GridTopology::Hex2d;
    cfg.initial_population = 30;
    cfg.max_population = 500;
    cfg.steps = 10_000;
    cfg.hex_dish_radius = 0;
    cfg
}

/// Strip batch-scale grid settings so fresh interactive sessions stay responsive.
fn prepare_interactive_session_config(cfg: &mut Config) {
    if cfg.hex_dish_radius > INTERACTIVE_MAX_DISH_RADIUS {
        eprintln!(
            "[interactive] ignoring config hex_dish_radius {} (max {} for new sessions); using dish radius {}",
            cfg.hex_dish_radius, INTERACTIVE_MAX_DISH_RADIUS, DEFAULT_DISH_RADIUS_CELLS
        );
    }
    cfg.grid_topology = GridTopology::Hex2d;
    cfg.hex_dish_radius = 0;
    if cfg.max_population <= 0 {
        cfg.max_population = 500;
    }
    cfg.initial_population = cfg.initial_population.min(200);
}

struct AppOptions {
    config: Option<Config>,
    snapshot_paths: Vec<PathBuf>,
    arena_dir: Option<PathBuf>,
    add_sandbox: bool,
    /// Center-crop hex dish snapshots to this axial radius when loading.
    interactive_radius: Option<i32>,
}

fn collect_snapshots_from_dir(dir: &Path) -> Vec<PathBuf> {
    snapshot_registry::collect_snapshots_from_dir(dir)
}

fn print_snapshot_load_hint(failed_path: &Path) {
    let reg = snapshot_registry::resolve_registry_dir();
    let entries = snapshot_registry::list_registry_entries(&reg);
    if entries.is_empty() {
        return;
    }
    eprintln!(
        "[hint] registered snapshots in {} (use --arena or --load-snapshot=PATH):",
        reg.display()
    );
    for (path, label) in entries.iter().take(12) {
        if let Some(label) = label {
            eprintln!("  {}  ({})", path.display(), label);
        } else {
            eprintln!("  {}", path.display());
        }
    }
    if failed_path.components().any(|c| c.as_os_str() == "pipeline_goo_evolved") {
        let goo = reg
            .parent()
            .map(|p| p.join("pipeline_goo_evolved").join("goo").join("snapshot.cbor"));
        if let Some(goo) = goo {
            if goo.is_file() {
                eprintln!(
                    "[hint] GoO evolved snapshot is at: {}",
                    goo.display()
                );
            }
        }
    }
}

fn copy_dynamics_config(dst: &mut Config, src: &Config) {
    dst.reproduction_threshold = src.reproduction_threshold;
    dst.reproduction_cost = src.reproduction_cost;
    dst.max_population = src.max_population;
    dst.basal_metabolic_cost = src.basal_metabolic_cost;
    dst.maintenance_cost_per_node = src.maintenance_cost_per_node;
    dst.death_risk = src.death_risk;
    dst.initial_population = src.initial_population;
}

fn merge_run_config_from_snapshot(config: &mut Config, file_cfg: &Config) {
    config.architecture_policy = file_cfg.architecture_policy;
    config.architecture_mutation_rate = file_cfg.architecture_mutation_rate;
    config.neutral_founder_weights = file_cfg.neutral_founder_weights;
    copy_dynamics_config(config, file_cfg);
}

fn dish_run_config_candidates(dish: &PetriDish) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(ref snap) = dish.source_snapshot {
        let reg_dir = snapshot_registry::resolve_registry_dir();
        if let Some(stem) = snap.file_stem().and_then(|s| s.to_str()) {
            let manifest_path = reg_dir.join(format!("{stem}.json"));
            if let Ok(text) = fs::read_to_string(&manifest_path) {
                if let Ok(manifest) = serde_json::from_str::<SnapshotManifest>(&text) {
                    if !manifest.source_run_dir.is_empty() {
                        let run_dir = PathBuf::from(&manifest.source_run_dir);
                        paths.push(run_dir.join("config.json"));
                        paths.push(run_dir.join("run_config.json"));
                    }
                    if let Some(cf) = manifest.config_file.filter(|s| !s.is_empty()) {
                        paths.push(PathBuf::from(cf));
                    }
                }
            }
        }
        if let Some(parent) = snap.parent() {
            paths.push(parent.join("config.json"));
            paths.push(parent.join("run_config.json"));
        }
    }
    let out = dish.simulation.config.output_run_dir.trim();
    if !out.is_empty() {
        let run_dir = PathBuf::from(out);
        paths.push(run_dir.join("config.json"));
        paths.push(run_dir.join("run_config.json"));
    }
    paths
}

fn dish_run_config_display(dish: &PetriDish) -> (String, String) {
    for path in dish_run_config_candidates(dish) {
        if path.is_file() {
            if let Ok(text) = fs::read_to_string(&path) {
                return (text, format!("{}", path.display()));
            }
        }
    }
    match serde_json::to_string_pretty(&dish.simulation.config) {
        Ok(json) => (
            json,
            "merged dish config (no run config.json found on disk)".to_string(),
        ),
        Err(e) => (format!("{{\"error\": \"{e}\"}}"), "unavailable".to_string()),
    }
}

fn draw_dish_run_config_panel(ui: &mut egui::Ui, dish: &PetriDish) {
    ui.heading("Run Configuration");
    ui.label(format!("Dish: {}", dish.name));
    ui.label(dish_info_line(dish));
    if let Some(ref path) = dish.source_snapshot {
        ui.label(format!("Snapshot: {}", path.display()));
    }
    ui.separator();
    let (json, source) = dish_run_config_display(dish);
    ui.label(format!("Source: {source}"));
    ui.add_space(4.0);
    egui::ScrollArea::vertical()
        .max_height(480.0)
        .show(ui, |ui| {
            let mut text = json;
            ui.add(
                egui::TextEdit::multiline(&mut text)
                    .font(egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY)
                    .interactive(false),
            );
        });
}

fn dish_info_line(dish: &PetriDish) -> String {
    if dish.is_sandbox {
        return "Empty sandbox — drag individuals here to experiment".to_string();
    }
    let arch = dish.architecture_policy.label();
    let pop = dish.simulation.individuals.len();
    let t = dish.headless_timestep.unwrap_or(dish.timestep);
    let max_pop = dish.simulation.config.max_population;
    let mut line = if max_pop > 0 {
        format!("{arch} · t={t} · n={pop}/{max_pop}")
    } else {
        format!("{arch} · t={t} · n={pop}")
    };
    if let Some(ref path) = dish.source_snapshot {
        if let Some(name) = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) {
            line.push_str(&format!(" · from {name}"));
        }
    }
    line
}

fn parse_interactive_radius(value: &str) -> Option<i32> {
    let trimmed = value.trim().strip_prefix('r').unwrap_or(value.trim());
    trimmed.parse::<i32>().ok().map(|r| r.max(0))
}

fn print_interactive_help() {
    eprintln!(
        "evo_movement_interactive {} — options:",
        build_info::VERSION_LABEL
    );
    eprintln!("  --config-file=PATH         Load simulation config JSON");
    eprintln!("  --load-snapshot=PATH       Load one population snapshot (repeatable)");
    eprintln!(
        "  --arena[=DIR]              Load snapshots from DIR (default: ../output/snapshots flat registry, or */snapshot.cbor)"
    );
    eprintln!("  --sandbox                  Add an empty dish for manual drag-and-drop experiments");
    eprintln!(
        "  --interactive-radius=N     Hex dish radius when loading snapshot genetics (default {DEFAULT_DISH_RADIUS_CELLS})"
    );
    eprintln!("  --r=N                      Alias for --interactive-radius=N");
    eprintln!("  --help, -h                 Show this help message");
    eprintln!("  --version, -V              Show build version and exit");
}

fn parse_cli_options() -> AppOptions {
    let mut config = None;
    let mut snapshot_paths = Vec::new();
    let mut arena_dir = None;
    let mut add_sandbox = false;
    let mut interactive_radius = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            print_interactive_help();
            std::process::exit(0);
        }
        if arg == "--version" || arg == "-V" {
            eprintln!(
                "evo_movement_interactive {}",
                build_info::VERSION_LABEL
            );
            std::process::exit(0);
        }
        if let Some(path) = arg.strip_prefix("--config-file=") {
            config = Config::from_file(path).ok();
            continue;
        }
        if arg == "--config-file" {
            if let Some(path) = args.next() {
                config = Config::from_file(&path).ok();
            }
            continue;
        }
        if let Some(path) = arg.strip_prefix("--load-snapshot=") {
            snapshot_paths.push(PathBuf::from(path));
            continue;
        }
        if arg == "--load-snapshot" {
            if let Some(path) = args.next() {
                snapshot_paths.push(PathBuf::from(path));
            }
            continue;
        }
        if arg == "--arena" {
            arena_dir = Some(PathBuf::from("../output/snapshots"));
            continue;
        }
        if let Some(path) = arg.strip_prefix("--arena=") {
            arena_dir = Some(PathBuf::from(path));
            continue;
        }
        if arg == "--arena-dir" {
            if let Some(path) = args.next() {
                arena_dir = Some(PathBuf::from(path));
            }
            continue;
        }
        if let Some(path) = arg.strip_prefix("--arena-dir=") {
            arena_dir = Some(PathBuf::from(path));
            continue;
        }
        if arg == "--sandbox" {
            add_sandbox = true;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--interactive-radius=") {
            interactive_radius = parse_interactive_radius(value);
            continue;
        }
        if arg == "--interactive-radius" || arg == "-r" {
            if let Some(value) = args.next() {
                interactive_radius = parse_interactive_radius(&value);
            }
            continue;
        }
        if let Some(value) = arg.strip_prefix("--r=") {
            interactive_radius = parse_interactive_radius(value);
            continue;
        }
        if let Some(value) = arg.strip_prefix("r=") {
            interactive_radius = parse_interactive_radius(value);
        }
    }
    AppOptions {
        config,
        snapshot_paths,
        arena_dir,
        add_sandbox,
        interactive_radius,
    }
}

fn main() -> Result<(), eframe::Error> {
    for arg in env::args().skip(1) {
        if arg == "--help" || arg == "-h" {
            print_interactive_help();
            return Ok(());
        }
        if arg == "--version" || arg == "-V" {
            eprintln!(
                "evo_movement_interactive {}",
                build_info::VERSION_LABEL
            );
            return Ok(());
        }
    }

    println!(
        "Evolution interactive viewer {}",
        build_info::VERSION_LABEL
    );
    let options = parse_cli_options();
    let options = std::sync::Arc::new(options);
    let options_for_app = options.clone();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title(viewer_window_title()),
        ..Default::default()
    };

    eframe::run_native(
        &viewer_window_title(),
        options,
        Box::new(move |cc| Box::new(EvolutionApp::new(cc, options_for_app))),
    )
}

struct PetriDish {
    hex_grid: HexGrid,
    simulation: Simulation,
    cladogram: Cladogram,
    name: String,
    architecture_policy: ArchitecturePolicy,
    source_snapshot: Option<PathBuf>,
    is_sandbox: bool,
    timestep: i32,
    radius_cells: i32,
    active_coords: Vec<HexCoord>,
    coord_to_index: HashMap<HexCoord, usize>,
    resource_base: Vec<f64>,
    #[cfg(feature = "cuda")]
    gpu_resource_updater: Option<GpuResourceUpdater>,
    last_avg_move_signal: f64,
    headless_genotype_stats: HashMap<String, HeadlessGenotypeStats>,
    headless_timestep: Option<i32>,
    move_trails: Vec<MoveTrail>,
    rewind_history: VecDeque<DishCheckpoint>,
    rewind_index: usize,
}

impl PetriDish {
    fn new_hex_dish(id: usize, radius_cells: i32, hex_size: f32, config: Config) -> Self {
        let hex_grid = HexGrid::new_hex_dish(radius_cells, hex_size);
        Self::from_grid(id, hex_grid, config, radius_cells)
    }

    fn from_grid(id: usize, hex_grid: HexGrid, mut config: Config, radius_cells: i32) -> Self {
        let mut active_coords: Vec<HexCoord> = hex_grid.cells.keys().copied().collect();
        active_coords.sort_by_key(|c| (c.r, c.q));
        let mut coord_to_index = HashMap::with_capacity(active_coords.len());
        for (idx, coord) in active_coords.iter().enumerate() {
            coord_to_index.insert(*coord, idx);
        }

        // Perceptron: current-cell sensors; other policies need six edge + occupancy + stores.
        config.grid_topology = GridTopology::Hex2d;
        if config.architecture_policy == ArchitecturePolicy::PerceptronOnly {
            config.nn_inputs = PERCEPTRON_SENSOR_DIM as i32;
        } else {
            config.nn_inputs = config.nn_inputs.max(8);
        }
        config.grid_width = active_coords.len().max(1);
        let simulation = Simulation::new(&config);
        let cladogram = Cladogram::new();

        let seed = config.seed as u64;
        let spatial_scale = config.env_perlin_spatial_scale;
        let time_scale = config.env_perlin_time_scale;
        let mut resource_base = Vec::with_capacity(active_coords.len());
        for coord in &active_coords {
            let v = Perlin::sample_2d_time(
                coord.q as f64,
                coord.r as f64,
                0.0,
                seed,
                spatial_scale,
                time_scale,
            );
            resource_base.push(v);
        }

        #[cfg(feature = "cuda")]
        let gpu_resource_updater = GpuResourceUpdater::new(&active_coords, &resource_base).ok();

        let mut dish = Self {
            hex_grid,
            simulation,
            cladogram,
            name: format!("Dish {}", id),
            architecture_policy: config.architecture_policy,
            source_snapshot: None,
            is_sandbox: false,
            timestep: 0,
            radius_cells,
            active_coords,
            coord_to_index,
            resource_base,
            #[cfg(feature = "cuda")]
            gpu_resource_updater,
            last_avg_move_signal: 0.0,
            headless_genotype_stats: HashMap::new(),
            headless_timestep: None,
            move_trails: Vec::new(),
            rewind_history: VecDeque::new(),
            rewind_index: 0,
        };
        dish.update_resource_grid();
        dish.seed_rewind_history();
        dish
    }

    fn seed_rewind_history(&mut self) {
        self.rewind_history.clear();
        self.rewind_history.push_back(self.capture_checkpoint());
        self.rewind_index = 0;
    }

    fn capture_checkpoint(&self) -> DishCheckpoint {
        DishCheckpoint {
            timestep: self.timestep,
            perlin_time: self.simulation.env.perlin_time,
            env_resources: self.simulation.env.resources.clone(),
            individuals: self.simulation.individuals.clone(),
            next_id: self.simulation.next_id,
            cladogram: self.cladogram.clone(),
        }
    }

    fn restore_checkpoint(&mut self, cp: DishCheckpoint) {
        self.timestep = cp.timestep;
        self.simulation.env.perlin_time = cp.perlin_time;
        self.simulation.env.resources = cp.env_resources;
        self.simulation.individuals = cp.individuals;
        self.simulation.next_id = cp.next_id;
        self.cladogram = cp.cladogram;
        self.move_trails.clear();
        self.rebuild_hex_occupancy();
        self.update_resource_grid();
    }

    fn truncate_rewind_future(&mut self) {
        if self.rewind_index + 1 < self.rewind_history.len() {
            self.rewind_history.truncate(self.rewind_index + 1);
        }
    }

    fn push_rewind_checkpoint(&mut self) {
        self.rewind_history.push_back(self.capture_checkpoint());
        self.rewind_index = self.rewind_history.len().saturating_sub(1);
        while self.rewind_history.len() > REWIND_CAP {
            self.rewind_history.pop_front();
            self.rewind_index = self.rewind_index.saturating_sub(1);
        }
    }

    fn step_back(&mut self) -> bool {
        if self.rewind_index == 0 {
            return false;
        }
        self.rewind_index -= 1;
        let cp = self
            .rewind_history
            .get(self.rewind_index)
            .cloned()
            .expect("rewind checkpoint");
        self.restore_checkpoint(cp);
        true
    }

    fn age_move_trails(&mut self) {
        for trail in &mut self.move_trails {
            trail.age = trail.age.saturating_add(1);
        }
        self.move_trails.retain(|t| t.age < TRAIL_MAX_AGE);
    }

    fn new_sandbox(id: usize, radius_cells: i32, hex_size: f32, config: Config) -> Self {
        let mut dish = Self::new_hex_dish(id, radius_cells, hex_size, config);
        dish.is_sandbox = true;
        dish.name = "Sandbox".to_string();
        dish.simulation.individuals.clear();
        dish.cladogram = Cladogram::new();
        dish.timestep = 0;
        dish.headless_timestep = None;
        dish.headless_genotype_stats.clear();
        dish.seed_rewind_history();
        dish
    }

    /// Load snapshot genetics onto a fresh interactive dish (no spatial replay).
    fn from_population_snapshot(
        id: usize,
        snapshot: PopulationSnapshot,
        mut config: Config,
        hex_size: f32,
        interactive_radius: Option<i32>,
        source_snapshot: Option<PathBuf>,
    ) -> Result<Self, SnapshotError> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use rand::seq::SliceRandom;

        let view_radius = interactive_radius
            .filter(|&r| r > 0)
            .unwrap_or(DEFAULT_DISH_RADIUS_CELLS);
        let source_pop = snapshot.individuals.len();
        let source_dish = snapshot.hex_dish_radius;
        let headless_timestep = snapshot.timestep;
        let headless_genotype_stats =
            compute_headless_genotype_stats(&snapshot.individuals);

        config.grid_topology = GridTopology::Hex2d;
        if config.architecture_policy == ArchitecturePolicy::PerceptronOnly {
            config.nn_inputs = PERCEPTRON_SENSOR_DIM as i32;
        } else {
            config.nn_inputs = config.nn_inputs.max(8);
        }
        config.hex_dish_radius = 0;

        if let Some(ref src) = source_snapshot {
            if let Some(parent) = src.parent() {
                for name in ["config.json", "resolved_config.json"] {
                    let cfg_path = parent.join(name);
                    if let Ok(file_cfg) = Config::from_file(&cfg_path) {
                        merge_run_config_from_snapshot(&mut config, &file_cfg);
                        break;
                    }
                }
            }
        }

        let arch_policy = config.architecture_policy;
        let mut dish = Self::new_hex_dish(id, view_radius, hex_size, config);
        let cell_count = dish.active_coords.len().max(1);

        let display_cap = dish
            .simulation
            .config
            .max_population
            .max(1) as usize;
        let mut rng = StdRng::seed_from_u64(snapshot.rng_seed_hint);
        let mut records = snapshot.individuals;
        if records.len() > display_cap {
            records.shuffle(&mut rng);
            records.truncate(display_cap);
            eprintln!(
                "[interactive] subsampled {display_cap} / {source_pop} snapshot genotypes for display"
            );
        }

        let mut individuals: Vec<Individual> = records
            .into_iter()
            .map(|rec| rec.into_individual())
            .collect();

        assign_genotype_colors(&mut individuals);

        let mut place_rng = StdRng::seed_from_u64(snapshot.rng_seed_hint.wrapping_add(1));
        for ind in &mut individuals {
            ind.position = place_rng.gen_range(0..cell_count);
        }

        dish.simulation.next_id = snapshot.next_id;
        dish.simulation.individuals = individuals;
        dish.simulation.env.perlin_time = snapshot.env_perlin_time;
        dish.timestep = snapshot.timestep;
        dish.headless_genotype_stats = headless_genotype_stats;
        dish.headless_timestep = Some(headless_timestep);
        dish.architecture_policy = arch_policy;
        dish.source_snapshot = source_snapshot;

        dish.cladogram = Cladogram::new();
        for ind in &dish.simulation.individuals {
            dish.cladogram.add_founder(ind.id, ind.lineage_color);
        }

        dish.name = format!(
            "{} · t={} · n={}",
            arch_policy.label(),
            headless_timestep,
            dish.simulation.individuals.len(),
        );

        eprintln!(
            "[interactive] seeded visualization dish r={view_radius} from snapshot \
             (source dish r={source_dish}, t={headless_timestep}, genetics only — env/positions not restored)",
        );

        dish.rebuild_hex_occupancy();
        dish.update_resource_grid();
        dish.seed_rewind_history();
        Ok(dish)
    }

    fn init_population(&mut self, count: usize) {
        let start = self.simulation.individuals.len();
        self.simulation.init_population(count);

        for i in start..self.simulation.individuals.len() {
            if let Some(ind) = self.simulation.individuals.get_mut(i) {
                ind.parent_id = None;
                let sig = genotype_signature(ind);
                ind.lineage_color = genotype_color(&sig);
                self.cladogram.add_founder(ind.id, ind.lineage_color);
            }
        }

        self.rebuild_hex_occupancy();

        // Initialize environment resources
        self.update_resource_grid();
    }

    fn update_resource_grid(&mut self) {
        let seed = self.simulation.config.seed as u64;
        let spatial_scale = self.simulation.config.env_perlin_spatial_scale;
        let time_scale = self.simulation.config.env_perlin_time_scale;
        let time_amplitude = self.simulation.config.env_time_amplitude;
        let resource_scale = self.simulation.config.resource_scale;

        #[cfg(feature = "cuda")]
        if let Some(gpu) = self.gpu_resource_updater.as_mut() {
            if let Some(resources) = gpu.compute(
                seed as u32,
                self.simulation.env.perlin_time as f32,
                spatial_scale as f32,
                time_scale as f32,
                time_amplitude as f32,
                resource_scale as f32,
            ) {
                for (idx, coord) in self.active_coords.iter().enumerate() {
                    let resource = resources.get(idx).copied().unwrap_or(0.0).max(0.0);
                    if let Some(slot) = self.simulation.env.resources.get_mut(idx) {
                        *slot = resource;
                    }
                    if let Some(cell) = self.hex_grid.get_cell_mut(*coord) {
                        cell.resource_level = resource;
                    }
                }
                return;
            }
        }

        for (idx, coord) in self.active_coords.iter().enumerate() {
            let temporal = Perlin::sample_2d_time(
                coord.q as f64,
                coord.r as f64,
                self.simulation.env.perlin_time,
                seed,
                spatial_scale,
                time_scale,
            );
            let centered_temporal = temporal - 0.5;
            let base = self.resource_base.get(idx).copied().unwrap_or(0.0);
            let resource = ((base + centered_temporal * time_amplitude) * resource_scale).max(0.0);

            if let Some(slot) = self.simulation.env.resources.get_mut(idx) {
                *slot = resource;
            }
            if let Some(cell) = self.hex_grid.get_cell_mut(*coord) {
                cell.resource_level = resource;
            }
        }
    }

    fn rebuild_hex_occupancy(&mut self) {
        for cell in self.hex_grid.cells.values_mut() {
            cell.individual_indices.clear();
        }
        let width = self.active_coords.len().max(1);
        for (idx, ind) in self.simulation.individuals.iter_mut().enumerate() {
            let pos = ind.position % width;
            ind.position = pos;
            let coord = self.active_coords[pos];
            if let Some(cell) = self.hex_grid.get_cell_mut(coord) {
                cell.individual_indices.push(idx);
            }
        }
    }

    fn step(&mut self, move_threshold: f64, env_speed_multiplier: f32) {
        if self.active_coords.is_empty() {
            return;
        }

        let before_ids: HashSet<i32> = self
            .simulation
            .individuals
            .iter()
            .map(|ind| ind.id)
            .collect();

        // Advance time
        self.timestep += 1;
        self.cladogram.advance_time();
        let env_delta = self.simulation.config.env_time_speed * env_speed_multiplier as f64;
        self.simulation.env.perlin_time += env_delta;

        // Refresh 2D spatial + temporal resources for this dish
        self.update_resource_grid();

        // Run full simulation dynamics with one-hex directional movement.
        let width = self.active_coords.len();
        let mut occupancy = vec![0i32; width];
        for ind in &self.simulation.individuals {
            let pos = ind.position % width;
            occupancy[pos] = occupancy[pos].saturating_add(1);
        }

        self.simulation.last_spend.resize(self.simulation.individuals.len(), 0.0);
        self.simulation.last_delta.resize(self.simulation.individuals.len(), 0);
        self.simulation.last_movers = 0;
        let mut move_signal_sum = 0.0f64;
        let mut move_signal_count = 0usize;

        for (i, ind) in self.simulation.individuals.iter_mut().enumerate() {
            let pos = ind.position % width;
            let coord = self.active_coords[pos];
            let neighbors = coord.neighbors();

            let center = self.simulation.env.resources.get(pos).copied().unwrap_or(0.0);
            let mut edge_resources = [center; 6];
            for (edge, ncoord) in neighbors.iter().enumerate() {
                edge_resources[edge] = self
                    .coord_to_index
                    .get(ncoord)
                    .and_then(|idx| self.simulation.env.resources.get(*idx))
                    .copied()
                    .unwrap_or(center);
            }
            let local_population = if self.simulation.config.nn_use_occupancy {
                occupancy[pos]
            } else {
                0
            };

            let decision = forward_decision(
                ind,
                edge_resources,
                center,
                local_population,
                self.simulation.config.movement_cost_factor,
                self.simulation.config.resolved_movement_output_mode(),
                &mut self.simulation.nn_hidden_buf,
                &mut self.simulation.nn_out2_buf,
            );
            move_signal_sum += decision.move_signal;
            move_signal_count = move_signal_count.saturating_add(1);

            let wants_move = decision.move_one_step && decision.move_signal > move_threshold;
            if wants_move && !move_allowed(ind, true, &self.simulation.config) {
                tick_move_wait(ind);
                let gathered = gather_amount_for_decision(
                    decision,
                    &self.simulation.env,
                    pos,
                    local_population.max(1),
                );
                if gathered > 0.0 {
                    ind.resources += gathered;
                    if let Some(resource_slot) = self.simulation.env.resources.get_mut(pos) {
                        *resource_slot = (*resource_slot - gathered).max(0.0);
                    }
                }
            } else if wants_move {
                self.simulation.last_spend[i] = decision.movement_spend;
                self.simulation.last_movers = self.simulation.last_movers.saturating_add(1);
                ind.resources -= decision.movement_spend;

                let mut sector = ((decision.direction_degrees.rem_euclid(360.0) / 60.0).floor() as usize) % 6;
                if sector >= 6 {
                    sector = 0;
                }
                let target_coord = neighbors[sector];
                if let Some(target_idx) = self.coord_to_index.get(&target_coord).copied() {
                    let from_pos = pos;
                    let color = ind.lineage_color;
                    ind.position = target_idx;
                    self.simulation.last_delta[i] = 1;
                    occupancy[pos] = occupancy[pos].saturating_sub(1);
                    occupancy[target_idx] = occupancy[target_idx].saturating_add(1);
                    start_move_cooldown(ind, &self.simulation.config);
                    self.move_trails.push(MoveTrail {
                        from_pos,
                        to_pos: target_idx,
                        color,
                        age: 0,
                    });
                }
            } else if decision.gathered > 0.0 {
                ind.resources += decision.gathered;
                if let Some(resource_slot) = self.simulation.env.resources.get_mut(pos) {
                    *resource_slot = (*resource_slot - decision.gathered).max(0.0);
                }
            }
        }

        self.last_avg_move_signal = if move_signal_count > 0 {
            move_signal_sum / move_signal_count as f64
        } else {
            0.0
        };

        for ind in &mut self.simulation.individuals {
            ind.age += 1;
        }

        let births = run_reproduction_phase(
            &mut self.simulation.individuals,
            &self.simulation.config,
            &mut self.simulation.rng,
            &mut self.simulation.next_id,
            |child| {
                let sig = genotype_signature(child);
                child.lineage_color = genotype_color(&sig);
            },
        );
        for birth in &births {
            if self.cladogram.nodes.contains_key(&birth.parent.id) {
                self.cladogram.add_offspring(
                    birth.parent.id,
                    birth.child.id,
                    birth.child.lineage_color,
                    birth.genetic_distance,
                );
            } else {
                self.cladogram.add_founder(birth.child.id, birth.child.lineage_color);
            }
        }

        let dead_ids =
            apply_metabolism_and_cull(&mut self.simulation.individuals, &self.simulation.config, &mut self.simulation.rng);
        for dead_id in dead_ids {
            self.cladogram.mark_death(dead_id);
        }

        let after_ids: HashSet<i32> = self
            .simulation
            .individuals
            .iter()
            .map(|ind| ind.id)
            .collect();

        for ind in &self.simulation.individuals {
            if !before_ids.contains(&ind.id) {
                if ind.parent_id.is_none() && !births.iter().any(|b| b.child.id == ind.id) {
                    self.cladogram.add_founder(ind.id, ind.lineage_color);
                }
            }
        }
        let _ = after_ids;

        self.rebuild_hex_occupancy();
    }

}

struct EvolutionApp {
    petri_dishes: Vec<PetriDish>,
    selected_dish: usize,
    show_all_dishes: bool,
    simulate_all_dishes: bool,
    selected_individual_id: Option<i32>,
    selected_hex: Option<HexCoord>,
    paused: bool,
    paused_before_drag: bool,
    is_dragging_individual: bool,
    env_playback_speed: MovementSpeed,
    zoom: f32,
    timesteps_per_frame: usize,
    move_threshold: f64,
    show_nn_diagram: bool,
    show_cladogram: bool,
    show_run_config: bool,
    hex_background_mode: HexBackgroundMode,
    phylogeny_zoom: f32,
    phylogeny_height: f32,
    phylogeny_max_render_nodes: usize,
    phylogeny_last_draw_ids: Vec<i32>,
    phylogeny_last_y_rank: HashMap<i32, f32>,
    show_config_editor: bool,
    config_radius_cells: i32,
    config_hex_size: f32,
    config: Config,
    interactive_radius: Option<i32>,
    extant_lineages_expanded: bool,
    last_frame: Option<Instant>,
    session_id: String,
    session_dir: PathBuf,
    last_brain_export_id: Option<i32>,
    last_brain_export_path: Option<PathBuf>,
}

impl EvolutionApp {
    fn new(_cc: &eframe::CreationContext<'_>, options: std::sync::Arc<AppOptions>) -> Self {
        let mut config = options.config.clone().unwrap_or_else(interactive_default_config);

        let mut snapshot_paths = options.snapshot_paths.clone();
        if let Some(ref dir) = options.arena_dir {
            snapshot_paths.extend(collect_snapshots_from_dir(dir));
        }
        snapshot_paths.sort();
        snapshot_paths.dedup();

        let loading_snapshot = !snapshot_paths.is_empty();
        if !loading_snapshot {
            prepare_interactive_session_config(&mut config);
        }

        let mut dishes = Vec::new();
        for (i, snap_path) in snapshot_paths.iter().enumerate() {
            match PopulationSnapshot::read_from_path(snap_path) {
                Ok(snapshot) => {
                    let dish_config = config.clone();
                    match PetriDish::from_population_snapshot(
                        i,
                        snapshot,
                        dish_config,
                        10.0,
                        options.interactive_radius,
                        Some(snap_path.clone()),
                    ) {
                        Ok(dish) => {
                            config = dish.simulation.config.clone();
                            dishes.push(dish);
                        }
                        Err(e) => eprintln!("Failed to build dish from {}: {}", snap_path.display(), e),
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load snapshot {:?}: {}", snap_path, e);
                    print_snapshot_load_hint(snap_path);
                }
            }
        }

        if options.add_sandbox || options.arena_dir.is_some() {
            let id = dishes.len();
            dishes.push(PetriDish::new_sandbox(
                id,
                DEFAULT_DISH_RADIUS_CELLS,
                10.0,
                config.clone(),
            ));
        } else if dishes.is_empty() {
            let dish = PetriDish::new_hex_dish(0, DEFAULT_DISH_RADIUS_CELLS, 10.0, config.clone());
            dishes.push(dish);
        }

        let config_radius_cells = dishes
            .first()
            .map(|d| d.radius_cells)
            .unwrap_or(DEFAULT_DISH_RADIUS_CELLS);

        let seed = config.seed;
        let (session_id, session_dir) =
            init_interactive_session(seed, &config, &snapshot_paths);

        Self {
            petri_dishes: dishes,
            selected_dish: 0,
            show_all_dishes: false,
            simulate_all_dishes: false,
            selected_individual_id: None,
            selected_hex: None,
            paused: true,
            paused_before_drag: true,
            is_dragging_individual: false,
            env_playback_speed: MovementSpeed::Medium,
            zoom: 1.0,
            timesteps_per_frame: 1,
            move_threshold: 0.0,
            show_nn_diagram: true,
            show_cladogram: false,
            show_run_config: false,
            hex_background_mode: HexBackgroundMode::Resources,
            phylogeny_zoom: 1.0,
            phylogeny_height: 300.0,
            phylogeny_max_render_nodes: 1500,
            phylogeny_last_draw_ids: Vec::new(),
            phylogeny_last_y_rank: HashMap::new(),
            show_config_editor: false,
            config_radius_cells,
            config_hex_size: 10.0,
            config,
            interactive_radius: options.interactive_radius,
            extant_lineages_expanded: true,
            last_frame: None,
            session_id,
            session_dir,
            last_brain_export_id: None,
            last_brain_export_path: None,
        }
    }

    fn maybe_export_selected_brain(&mut self) {
        let dish = match self.petri_dishes.get(self.selected_dish) {
            Some(d) => d,
            None => return,
        };
        let idx = match self.selected_individual_index(dish) {
            Some(i) => i,
            None => return,
        };
        let ind = match dish.simulation.individuals.get(idx) {
            Some(i) => i,
            None => return,
        };
        if self.last_brain_export_id == Some(ind.id) {
            return;
        }
        match export_individual_brain_json(&self.session_id, &self.session_dir, ind) {
            Ok(path) => {
                self.last_brain_export_id = Some(ind.id);
                self.last_brain_export_path = Some(path);
            }
            Err(e) => eprintln!("[interactive] brain json export failed: {e}"),
        }
    }

    fn sync_dynamics_to_selected_dish(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            copy_dynamics_config(&mut dish.simulation.config, &self.config);
        }
    }

    fn pull_dynamics_from_selected_dish(&mut self) {
        if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
            copy_dynamics_config(&mut self.config, &dish.simulation.config);
        }
    }

    fn add_petri_dish(&mut self) {
        let id = self.petri_dishes.len();
        let dish = PetriDish::new_hex_dish(id, DEFAULT_DISH_RADIUS_CELLS, 10.0, self.config.clone());
        self.petri_dishes.push(dish);
        self.selected_dish = id;
    }

    fn initialize_population(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            dish.init_population(self.config.initial_population as usize);
        }
    }

    fn start_new_simulation_from_config(&mut self) {
        let radius = self.config_radius_cells.max(1);
        let hex_size = self.config_hex_size.max(2.0);
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            *dish = PetriDish::new_hex_dish(self.selected_dish, radius, hex_size, self.config.clone());
            dish.init_population(self.config.initial_population as usize);
            self.selected_individual_id = None;
        }
    }

    fn add_dish_from_config(&mut self) {
        let radius = self.config_radius_cells.max(1);
        let hex_size = self.config_hex_size.max(2.0);
        let id = self.petri_dishes.len();
        let mut dish = PetriDish::new_hex_dish(id, radius, hex_size, self.config.clone());
        dish.init_population(self.config.initial_population as usize);
        self.petri_dishes.push(dish);
        self.selected_dish = id;
        self.selected_individual_id = None;
    }

    fn selected_individual_index(&self, dish: &PetriDish) -> Option<usize> {
        let selected_id = self.selected_individual_id?;
        dish.simulation
            .individuals
            .iter()
            .position(|ind| ind.id == selected_id)
    }

    fn step_simulation(&mut self) {
        let env_speed = ecology_presets::env_time_playback_multiplier(self.env_playback_speed);
        let dishes_to_step: Vec<usize> = if self.simulate_all_dishes {
            (0..self.petri_dishes.len()).collect()
        } else {
            vec![self.selected_dish]
        };
        for dish_idx in dishes_to_step {
            let Some(dish) = self.petri_dishes.get_mut(dish_idx) else {
                continue;
            };
            dish.truncate_rewind_future();
            for _ in 0..self.timesteps_per_frame {
                dish.step(self.move_threshold, env_speed);
            }
            dish.push_rewind_checkpoint();
        }
    }

    fn step_back_simulation(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            if dish.step_back() {
                self.paused = true;
            }
        }
    }

    fn sorted_lineages_in_cell(dish: &PetriDish, cell: &HexCell) -> Vec<GenotypeLineageRow> {
        let mut by_sig: HashMap<String, Vec<usize>> = HashMap::new();
        let mut colors: HashMap<String, [u8; 3]> = HashMap::new();
        for &ind_idx in &cell.individual_indices {
            if let Some(ind) = dish.simulation.individuals.get(ind_idx) {
                let sig = genotype_signature(ind);
                colors.entry(sig.clone()).or_insert(ind.lineage_color);
                by_sig.entry(sig).or_default().push(ind_idx);
            }
        }

        let mut rows: Vec<GenotypeLineageRow> = by_sig
            .into_iter()
            .map(|(signature, member_indices)| GenotypeLineageRow {
                color: *colors.get(&signature).unwrap_or(&[128, 128, 128]),
                signature,
                pop: member_indices.len(),
                movers_pct: None,
                member_indices,
            })
            .collect();
        rows.sort_unstable_by(|a, b| {
            b.pop
                .cmp(&a.pop)
                .then_with(|| a.signature.cmp(&b.signature))
        });
        rows
    }

    fn dominant_lineage_color(dish: &PetriDish, cell: &HexCell) -> Option<[u8; 3]> {
        Self::sorted_lineages_in_cell(dish, cell)
            .into_iter()
            .next()
            .map(|row| row.color)
    }

    fn sorted_extant_lineages(dish: &PetriDish) -> Vec<GenotypeLineageRow> {
        let total_pop = dish.simulation.individuals.len().max(1);
        let has_mover_data = dish.timestep > 0
            && dish.simulation.last_delta.len() == dish.simulation.individuals.len();
        let mut by_sig: HashMap<String, (usize, usize, [u8; 3], Vec<usize>)> = HashMap::new();

        for (idx, ind) in dish.simulation.individuals.iter().enumerate() {
            let sig = genotype_signature(ind);
            let entry = by_sig
                .entry(sig)
                .or_insert((0, 0, ind.lineage_color, Vec::new()));
            entry.0 += 1;
            entry.2 = ind.lineage_color;
            entry.3.push(idx);
            if has_mover_data
                && dish.simulation.last_delta.get(idx).copied().unwrap_or(0) != 0
            {
                entry.1 += 1;
            }
        }

        let mut rows: Vec<GenotypeLineageRow> = by_sig
            .into_iter()
            .map(|(signature, (pop, movers, color, member_indices))| {
                let movers_pct = if has_mover_data {
                    Some(movers as f64 / total_pop as f64 * 100.0)
                } else {
                    None
                };
                GenotypeLineageRow {
                    signature,
                    color,
                    pop,
                    movers_pct,
                    member_indices,
                }
            })
            .collect();

        rows.sort_unstable_by(|a, b| {
            b.pop
                .cmp(&a.pop)
                .then_with(|| {
                    b.movers_pct
                        .unwrap_or(0.0)
                        .partial_cmp(&a.movers_pct.unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.signature.cmp(&b.signature))
        });
        rows
    }

    fn draw_extant_genotypes_panel(
        extant_lineages_expanded: &mut bool,
        selected_individual_id: &mut Option<i32>,
        ui: &mut egui::Ui,
        dish: &PetriDish,
    ) {
        let lineages = Self::sorted_extant_lineages(dish);
        let total_pop: usize = lineages.iter().map(|r| r.pop).sum();

        ui.horizontal(|ui| {
            let arrow = if *extant_lineages_expanded { "▼" } else { "▶" };
            if ui
                .button(format!("{arrow} Extant Genotypes"))
                .clicked()
            {
                *extant_lineages_expanded = !*extant_lineages_expanded;
            }
        });

        if *extant_lineages_expanded {
            if lineages.is_empty() {
                ui.label("No extant genotypes yet.");
            } else {
                ui.label(format!(
                    "{} genotypes · {} individuals",
                    lineages.len(),
                    total_pop
                ));
                egui::ScrollArea::vertical()
                    .max_height(260.0)
                    .show(ui, |ui| {
                        for row in &lineages {
                            let movers_text = match row.movers_pct {
                                Some(pct) => format!("movers: {:.2}%", pct),
                                None => "movers: n/a".to_string(),
                            };
                            let batch_text = if let Some(batch_t) = dish.headless_timestep {
                                if let Some(stats) = dish.headless_genotype_stats.get(&row.signature)
                                {
                                    format!(
                                        "batch t={batch_t}: pop #{}/{} ({:.2}%) · res #{}/{} ({:.2}%)",
                                        stats.pop_rank,
                                        stats.total_genotypes,
                                        stats.pop_share,
                                        stats.resource_rank,
                                        stats.total_genotypes,
                                        stats.resource_share,
                                    )
                                } else {
                                    String::new()
                                }
                            } else {
                                String::new()
                            };

                            let label = if batch_text.is_empty() {
                                format!("pop: {} | {}", row.pop, movers_text)
                            } else {
                                format!("pop: {} | {} | {}", row.pop, movers_text, batch_text)
                            };

                            ui.horizontal(|ui| {
                                let (swatch_resp, swatch_painter) = ui.allocate_painter(
                                    egui::vec2(16.0, 16.0),
                                    egui::Sense::hover(),
                                );
                                let center = swatch_resp.rect.center();
                                let hex = hexagon_points(center, 6.5);
                                swatch_painter.add(egui::Shape::convex_polygon(
                                    hex,
                                    egui::Color32::from_rgb(
                                        row.color[0], row.color[1], row.color[2],
                                    ),
                                    egui::Stroke::new(0.8, egui::Color32::BLACK),
                                ));

                                let response = ui.selectable_label(false, label);
                                if response.clicked() {
                                    use rand::seq::SliceRandom;
                                    if let Some(&random_idx) = row
                                        .member_indices
                                        .as_slice()
                                        .choose(&mut rand::thread_rng())
                                    {
                                        if let Some(ind) =
                                            dish.simulation.individuals.get(random_idx)
                                        {
                                            *selected_individual_id = Some(ind.id);
                                        }
                                    }
                                }
                            });
                        }
                    });
            }
        } else if !lineages.is_empty() {
            ui.label(format!(
                "{} genotypes · {} individuals (collapsed)",
                lineages.len(),
                total_pop
            ));
        }
    }

    fn generation_metrics(dish: &PetriDish) -> (usize, Option<f64>) {
        if dish.simulation.individuals.is_empty() {
            return (0, None);
        }

        let extant_ids: Vec<i32> = dish.simulation.individuals.iter().map(|ind| ind.id).collect();

        let mut max_depth = 0usize;
        for id in &extant_ids {
            let mut cur = *id;
            let mut depth = 0usize;
            let mut guard = 0usize;
            while let Some(node) = dish.cladogram.nodes.get(&cur) {
                if let Some(parent_id) = node.parent_id {
                    depth = depth.saturating_add(1);
                    cur = parent_id;
                    guard = guard.saturating_add(1);
                    if guard > dish.cladogram.nodes.len() {
                        break;
                    }
                } else {
                    break;
                }
            }
            max_depth = max_depth.max(depth);
        }

        let generations = max_depth.saturating_add(1);

        let mut interval_sum = 0.0f64;
        let mut interval_count = 0usize;
        for node in dish.cladogram.nodes.values() {
            if let Some(parent_id) = node.parent_id {
                if let Some(parent) = dish.cladogram.nodes.get(&parent_id) {
                    let dt = (node.birth_time - parent.birth_time).max(0) as f64;
                    if dt > 0.0 {
                        interval_sum += dt;
                        interval_count = interval_count.saturating_add(1);
                    }
                }
            }
        }

        let generation_time = if interval_count > 0 {
            Some(interval_sum / interval_count as f64)
        } else if max_depth > 0 {
            Some(dish.timestep.max(0) as f64 / max_depth as f64)
        } else {
            None
        };

        (generations, generation_time)
    }

    fn draw_dish_thumbnail(&mut self, ui: &mut egui::Ui, dish_idx: usize, tile_size: egui::Vec2) {
        let Some(dish) = self.petri_dishes.get(dish_idx) else {
            return;
        };

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for coord in dish.hex_grid.cells.keys() {
            let (x, y) = coord.to_pixel(dish.hex_grid.hex_size);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let (min_x, max_x, min_y, max_y) = if min_x.is_finite() {
            (min_x, max_x, min_y, max_y)
        } else {
            (0.0, 1.0, 0.0, 1.0)
        };

        let width_px = (max_x - min_x) + 2.0 * dish.hex_grid.hex_size;
        let height_px = (max_y - min_y) + 2.0 * dish.hex_grid.hex_size;
        let scale = ((tile_size.x - 10.0) / width_px)
            .min((tile_size.y - 10.0) / height_px)
            .max(0.05);
        let hex_size = dish.hex_grid.hex_size;
        let padding = hex_size;

        let canvas_size = tile_size;
        let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click());

        let drawn_w = width_px * scale;
        let drawn_h = height_px * scale;
        let offset = egui::vec2(
            (canvas_size.x - drawn_w).max(0.0) * 0.5,
            (canvas_size.y - drawn_h).max(0.0) * 0.5,
        );

        let max_occupants = max_cell_occupancy(&dish.hex_grid.cells);
        let bg_mode = self.hex_background_mode;

        for (coord, cell) in &dish.hex_grid.cells {
            let (x, y) = coord.to_pixel(hex_size);
            let local_x = (x - min_x + padding) * scale;
            let local_y = (y - min_y + padding) * scale;
            let screen_pos = response.rect.min + offset + egui::vec2(local_x, local_y);

            let hex_points = hexagon_points(screen_pos, hex_size * scale);
            let color = hex_background_color(
                bg_mode,
                cell.resource_level,
                cell.individual_indices.len(),
                max_occupants,
            );
            painter.add(egui::Shape::convex_polygon(
                hex_points,
                color,
                egui::Stroke::new(0.4, egui::Color32::GRAY),
            ));

            if let Some(dom) = Self::dominant_lineage_color(dish, cell) {
                painter.circle_filled(
                    screen_pos,
                    (2.2 * scale).max(1.8),
                    egui::Color32::from_rgb(dom[0], dom[1], dom[2]),
                );
            }
        }

        if response.clicked() {
            self.selected_dish = dish_idx;
        }
    }

    fn restart_simulation(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            let radius = dish.radius_cells;
            let hex_size = dish.hex_grid.hex_size;
            *dish = PetriDish::new_hex_dish(self.selected_dish, radius, hex_size, self.config.clone());
            dish.init_population(self.config.initial_population as usize);
        }
        self.selected_individual_id = None;
        self.selected_hex = None;
    }

    fn save_state(&self, filename: &str) {
        if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
            let state = SimulationState::from_individuals_and_grid(
                &dish.simulation.individuals,
                &dish.hex_grid,
                &dish.cladogram,
                &dish.simulation.config,
                dish.timestep,
                dish.simulation.next_id,
            );

            match state.save_to_file(filename) {
                Ok(_) => println!("State saved to {}", filename),
                Err(e) => eprintln!("Failed to save state: {}", e),
            }
        }
    }

    fn load_state(&mut self, filename: &str) {
        match SimulationState::load_from_file(filename) {
            Ok(state) => {
                // Reconstruct the petri dish from the saved state
                let dish_id = self.selected_dish;
                let mut grid = HexGrid {
                    width: state.grid_width,
                    height: state.grid_height,
                    hex_size: state.hex_size,
                    cells: HashMap::new(),
                };
                for &(q, r) in state.hex_cells.keys() {
                    grid.cells.insert(HexCoord::new(q, r), HexCell::new());
                }
                let mut dish = PetriDish::from_grid(dish_id, grid, state.config.clone(), DEFAULT_DISH_RADIUS_CELLS);

                dish.timestep = state.timestep;
                dish.simulation.next_id = state.next_id;

                // Reconstruct individuals
                for ind_state in state.individuals {
                    let mut nn = NeuralNetwork::new(
                        ind_state.nn_nodes,
                        ind_state.nn_inputs,
                        ind_state.nn_layers,
                        ind_state.nn_nodes2,
                    );
                    nn.weights = ind_state.nn_weights;
                    nn.weights_multi = ind_state.nn_weights_multi;
                    nn.biases = ind_state.nn_biases;
                    nn.weights2 = ind_state.nn_weights2;
                    nn.biases2 = ind_state.nn_biases2;

                    let mut ind = Individual::from_legacy_nn_with_color(
                        nn,
                        ind_state.position,
                        ind_state.id,
                        ind_state.lineage_color,
                    );
                    ind.resources = ind_state.resources;
                    ind.age = ind_state.age;
                    ind.parent_id = ind_state.parent_id;

                    dish.simulation.individuals.push(ind);
                }

                // Reconstruct cladogram
                for node_state in state.cladogram_nodes {
                    use evo_movement_rust::cladogram::LineageNode;
                    dish.cladogram.nodes.insert(
                        node_state.id,
                        LineageNode {
                            id: node_state.id,
                            parent_id: node_state.parent_id,
                            children: node_state.children,
                            birth_time: node_state.birth_time,
                            death_time: node_state.death_time,
                            color: node_state.color,
                            genetic_distance_to_parent: node_state.genetic_distance_to_parent,
                        },
                    );
                }
                dish.cladogram.current_time = state.timestep;

                // Reconstruct hex cells
                for ((q, r), cell_state) in state.hex_cells {
                    let coord = HexCoord::new(q, r);
                    if let Some(cell) = dish.hex_grid.get_cell_mut(coord) {
                        cell.resource_level = cell_state.resource_level;
                        cell.individual_indices = cell_state.individual_indices;
                    }
                }

                dish.seed_rewind_history();
                self.petri_dishes[self.selected_dish] = dish;
                println!("State loaded from {}", filename);
            }
            Err(e) => eprintln!("Failed to load state: {}", e),
        }
    }

    fn load_population_snapshot(&mut self, filename: &str) {
        match PopulationSnapshot::read_from_path(filename) {
            Ok(snapshot) => {
                let dish_id = self.selected_dish;
                match PetriDish::from_population_snapshot(
                    dish_id,
                    snapshot,
                    self.config.clone(),
                    self.config_hex_size,
                    self.interactive_radius,
                    Some(PathBuf::from(filename)),
                ) {
                    Ok(dish) => {
                        self.config = dish.simulation.config.clone();
                        if let Some(slot) = self.petri_dishes.get_mut(dish_id) {
                            *slot = dish;
                        }
                        self.selected_individual_id = None;
                        self.selected_hex = None;
                        self.paused = true;
                        println!("Loaded population snapshot from {}", filename);
                    }
                    Err(e) => eprintln!("Failed to build dish from snapshot: {}", e),
                }
            }
            Err(e) => eprintln!("Failed to load snapshot {}: {}", filename, e),
        }
    }
}

impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.maybe_export_selected_brain();

        // Top menu bar
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New Simulation").clicked() {
                        self.restart_simulation();
                    }
                    if ui.button("➕ Add Petri Dish").clicked() {
                        self.add_petri_dish();
                    }
                    if ui.button("🧬 Initialize Population").clicked() {
                        self.initialize_population();
                    }
                    ui.separator();
                    if ui.button("Config Editor").clicked() {
                        self.show_config_editor = true;
                    }
                    if ui.button("Save State").clicked() {
                        self.save_state("simulation_state.json");
                    }
                    if ui.button("Open Session Folder").clicked() {
                        open_session_folder(&self.session_dir);
                    }
                    if ui.button("Load State").clicked() {
                        self.load_state("simulation_state.json");
                    }
                    if ui.button("Load Snapshot (goo_gpu_demo)").clicked() {
                        self.load_population_snapshot("../output/goo_gpu_demo/snapshot.cbor");
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_nn_diagram, "Show Neural Network");
                    ui.checkbox(&mut self.show_cladogram, "Show Cladogram");
                    ui.checkbox(&mut self.show_run_config, "Show Run Configuration")
                        .on_hover_text("Headless run config for the active dish (right panel)");
                    ui.separator();
                    ui.checkbox(&mut self.show_all_dishes, "Show All Dishes")
                        .on_hover_text("Arena grid of dish previews (heavier to draw)");
                    ui.checkbox(&mut self.simulate_all_dishes, "Simulate All Dishes")
                        .on_hover_text(
                            "Step every loaded dish each frame. Off = only the selected dish (faster in arena mode).",
                        );
                    ui.separator();
                    ui.label("Hex background");
                    ui.selectable_value(
                        &mut self.hex_background_mode,
                        HexBackgroundMode::Resources,
                        "Resource distribution",
                    );
                    ui.selectable_value(
                        &mut self.hex_background_mode,
                        HexBackgroundMode::Competition,
                        "Competition heatmap (population)",
                    );
                });

                ui.separator();

                // Simulation controls
                if self.paused {
                    if ui.button("▶ Play").clicked() {
                        self.paused = false;
                    }
                } else {
                    if ui.button("⏸ Pause").clicked() {
                        self.paused = true;
                    }
                }

                if ui.button("⏭ Step").clicked() {
                    self.step_simulation();
                }

                if ui.button("⏮ Step Back").clicked() {
                    self.step_back_simulation();
                }

                if ui.button("🔄 Restart").clicked() {
                    self.restart_simulation();
                }

                ui.separator();

                ui.label("Env:");
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.env_playback_speed,
                        MovementSpeed::Slow,
                        "Slow",
                    )
                    .on_hover_text("Ecology slow preset (0.2× medium env advance)");
                    ui.selectable_value(
                        &mut self.env_playback_speed,
                        MovementSpeed::Medium,
                        "Medium",
                    )
                    .on_hover_text("Ecology medium preset (1× env advance)");
                    ui.selectable_value(
                        &mut self.env_playback_speed,
                        MovementSpeed::Fast,
                        "Fast",
                    )
                    .on_hover_text("Ecology fast preset (5× medium env advance)");
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("session {}", &self.session_id[..8.min(self.session_id.len())]))
                            .weak()
                            .monospace(),
                    )
                    .on_hover_text(format!(
                        "Cite this interactive run as session {}\n{}",
                        self.session_id,
                        self.session_dir.display()
                    ));
                    ui.label(
                        egui::RichText::new(build_info::VERSION_LABEL)
                            .weak()
                            .monospace(),
                    )
                    .on_hover_text("Viewer build — window title and Help also show this");
                    ui.menu_button("Help", |ui| {
                        ui.heading("About");
                        ui.label(format!(
                            "Interactive Viewer {}",
                            build_info::VERSION_LABEL
                        ));
                        ui.label("Rebuild after code changes to pick up a new build id.");
                    });
                });
            });
        });

        if self.show_config_editor {
            let mut editor_open = self.show_config_editor;
            let mut start_new = false;
            let mut add_new_dish = false;
            egui::Window::new("Simulation Config Editor")
                .open(&mut editor_open)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.label("Core");
                    ui.add(egui::Slider::new(&mut self.config.initial_population, 1..=1000).text("Initial Population"));
                    ui.add(egui::Slider::new(&mut self.config.seed, 0..=u32::MAX).text("Seed"));
                    ui.add(egui::Slider::new(&mut self.config_radius_cells, 5..=INTERACTIVE_MAX_DISH_RADIUS).text("Dish Radius (cells)"));
                    ui.add(egui::Slider::new(&mut self.config_hex_size, 4.0..=24.0).text("Hex Size"));

                    ui.label("Dynamics");
                    let mut dynamics_changed = false;
                    dynamics_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.max_population, 10..=50000)
                                .text("Max Population"),
                        )
                        .changed();
                    dynamics_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.reproduction_threshold, 1.0..=100.0)
                                .text("Reproduction Threshold"),
                        )
                        .changed();
                    dynamics_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.reproduction_cost, 0.0..=50.0)
                                .text("Reproduction Cost"),
                        )
                        .changed();
                    dynamics_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.basal_metabolic_cost, 0.0..=0.5)
                                .text("Basal Metabolic Cost"),
                        )
                        .changed();
                    dynamics_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.maintenance_cost_per_node, 0.0..=0.2)
                                .text("Maintenance / Node"),
                        )
                        .changed();
                    dynamics_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.death_risk, 0.0..=0.01)
                                .text("Death Risk"),
                        )
                        .changed();
                    if dynamics_changed {
                        self.sync_dynamics_to_selected_dish();
                    }

                    ui.separator();
                    ui.label("Neural Network");
                    ui.add(egui::Slider::new(&mut self.config.nn_nodes, 1..=128).text("Hidden Nodes"));
                    ui.add(egui::Slider::new(&mut self.config.nn_layers, 1..=2).text("Hidden Layers"));
                    ui.add(egui::Slider::new(&mut self.config.nn_nodes2, 0..=128).text("Second Hidden Nodes"));
                    ui.label("Sensors: 8 channels (6 edge resources + occupancy + internal resources) feed each individual's evolved brain (MLP, GoO graph, spatial graph, etc.).");

                    ui.separator();
                    ui.label("Evolution");
                    ui.add(egui::Slider::new(&mut self.config.mutation_rate, 0.0..=1.0).text("Mutation Rate"));
                    ui.add(egui::Slider::new(&mut self.config.mutation_sigma, 0.0..=2.0).text("Mutation Sigma"));
                    ui.add(egui::Slider::new(&mut self.config.movement_cost_factor, 0.0..=10.0).text("Movement Cost"));

                    ui.separator();
                    ui.label("Environment");
                    ui.add(egui::Slider::new(&mut self.config.resource_scale, 0.0..=5.0).text("Resource Scale"));
                    ui.add(egui::Slider::new(&mut self.config.env_time_speed, 0.0..=0.2).text("Env Time Speed"));
                    ui.add(egui::Slider::new(&mut self.config.env_time_amplitude, 0.0..=3.0).text("Env Time Amplitude"));
                    ui.add(egui::Slider::new(&mut self.config.env_perlin_spatial_scale, 0.001..=1.0).text("Spatial Scale"));
                    ui.add(egui::Slider::new(&mut self.config.env_perlin_time_scale, 0.001..=1.0).text("Time Scale"));

                    ui.separator();
                    if ui.button("Start New Simulation (Selected Dish)").clicked() {
                        start_new = true;
                    }
                    if ui.button("Add New Dish From Config").clicked() {
                        add_new_dish = true;
                    }
                });
            self.show_config_editor = editor_open;
            if start_new {
                self.start_new_simulation_from_config();
            }
            if add_new_dish {
                self.add_dish_from_config();
            }
        }

        // Left panel - Controls and info
        egui::SidePanel::left("control_panel").min_width(250.0).show(ctx, |ui| {
            ui.heading("Controls");

            ui.separator();

            let prev_dish = self.selected_dish;
            egui::ComboBox::from_label("Active Dish")
                .selected_text(format!("Dish {}", self.selected_dish))
                .show_ui(ui, |ui| {
                    for (i, dish) in self.petri_dishes.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_dish, i, &dish.name);
                    }
                });
            if self.selected_dish != prev_dish {
                self.pull_dynamics_from_selected_dish();
            }

            ui.separator();

            ui.label("Simulation Settings");
            let mut dynamics_changed = false;
            dynamics_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.initial_population, 1..=1000)
                        .text("Initial Population"),
                )
                .changed();
            dynamics_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.max_population, 10..=50000)
                        .text("Max Population"),
                )
                .changed();
            dynamics_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.reproduction_threshold, 1.0..=100.0)
                        .text("Repro Threshold"),
                )
                .changed();
            dynamics_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.reproduction_cost, 0.0..=50.0)
                        .text("Repro Cost"),
                )
                .changed();
            dynamics_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.basal_metabolic_cost, 0.0..=0.5)
                        .text("Basal Metabolic"),
                )
                .changed();
            if dynamics_changed {
                self.sync_dynamics_to_selected_dish();
            }
            ui.add(egui::Slider::new(&mut self.timesteps_per_frame, 1..=10)
                .text("Steps per Frame"));
            ui.add(egui::Slider::new(&mut self.move_threshold, -2.0..=2.0)
                .text("Move Threshold"));

            ui.separator();

            let dish_idx = self.selected_dish;
            let extant_expanded = &mut self.extant_lineages_expanded;
            let selected_id = &mut self.selected_individual_id;
            if let Some(dish) = self.petri_dishes.get(dish_idx) {
                ui.heading("Statistics");
                ui.label(format!("Individuals: {}", dish.simulation.individuals.len()));
                let max_pop = dish.simulation.config.max_population;
                if max_pop > 0 {
                    ui.label(format!("Population cap: {}/{}", dish.simulation.individuals.len(), max_pop));
                }
                ui.label(format!("Timestep: {}", dish.timestep));
                let (generations, generation_time) = Self::generation_metrics(dish);
                ui.label(format!("Generations: {}", generations));
                if let Some(gt) = generation_time {
                    ui.label(format!("Estimated Generation Time: {:.2} steps", gt));
                } else {
                    ui.label("Estimated Generation Time: n/a");
                }
                ui.label(format!("Move Threshold: {:.2}", self.move_threshold));
                ui.label(format!("Avg Move Signal (last step): {:.3}", dish.last_avg_move_signal));

                ui.separator();
                Self::draw_extant_genotypes_panel(extant_expanded, selected_id, ui, dish);

                // Display selected hex information
                if let Some(hex_coord) = self.selected_hex {
                    match dish.hex_grid.get_cell(hex_coord) {
                        Some(cell) => {
                            ui.separator();
                            ui.heading("Selected Hex");
                            ui.label(format!("Coordinates: ({}, {})", hex_coord.q, hex_coord.r));
                            
                            let pop_density = cell.individual_indices.len();
                            ui.label(format!("Population: {}", pop_density));

                            let resource_level = cell.resource_level;
                            if self.hex_background_mode == HexBackgroundMode::Competition {
                                let max_occ = max_cell_occupancy(&dish.hex_grid.cells);
                                if max_occ > 0 {
                                    let pct = 100.0 * pop_density as f64 / max_occ as f64;
                                    ui.label(format!("Competition intensity: {pct:.0}% of dish peak"));
                                }
                                ui.label(format!("Resource Level: {:.2}", resource_level));
                            } else {
                                let resources_per_ind = if pop_density > 0 {
                                    resource_level / pop_density as f64
                                } else {
                                    resource_level
                                };
                                ui.label(format!("Resource Level: {:.2}", resource_level));
                                ui.label(format!("Resources/Individual: {:.2}", resources_per_ind));
                            }
                            
                            // Display unique lineage colors present in this hex
                            if !cell.individual_indices.is_empty() {
                            ui.separator();
                            ui.label("Lineages in this hex:");

                            let lineage_rows = Self::sorted_lineages_in_cell(dish, cell);

                            // Display each lineage with a button to select a random individual
                            for row in lineage_rows {
                                ui.horizontal(|ui| {
                                    let color_rect = egui::Rect::from_min_size(
                                        ui.cursor().min,
                                        egui::vec2(16.0, 16.0),
                                    );
                                    let (color_resp, color_painter) = ui.allocate_painter(
                                        color_rect.size(), 
                                        egui::Sense::hover()
                                    );
                                    color_painter.rect_filled(
                                        color_resp.rect,
                                        2.0,
                                        egui::Color32::from_rgb(row.color[0], row.color[1], row.color[2]),
                                    );
                                    
                                    ui.label(format!("Count: {}", row.pop));
                                    
                                    if ui.small_button("Select Random").clicked() {
                                        use rand::seq::SliceRandom;
                                        if let Some(&random_idx) = row.member_indices.as_slice().choose(&mut rand::thread_rng()) {
                                            if let Some(ind) = dish.simulation.individuals.get(random_idx) {
                                                self.selected_individual_id = Some(ind.id);
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        },
                        None => {
                            ui.label(format!("Selected hex ({}, {}) not found", hex_coord.q, hex_coord.r));
                        }
                    }
                }

                if let Some(idx) = self.selected_individual_index(dish) {
                    if let Some(ind) = dish.simulation.individuals.get(idx) {
                        ui.separator();
                        ui.heading("Selected Individual");
                        ui.label(format!("ID: {}", ind.id));
                        ui.label(format!("Resources: {:.2}", ind.resources));
                        ui.label(format!("Age: {}", ind.age));
                        ui.label(format!("Position: {}", ind.position));
                        ui.label(format!("Genotype: {}", genotype_signature(ind)));
                        ui.label(format!("Brain: {}", ind.brain.kind().as_str()));
                        draw_brain_json_export_ui(
                            ui,
                            &self.session_id,
                            self.last_brain_export_path.as_ref(),
                        );
                        ui.collapsing("Neural network diagram", |ui| {
                            nn_viz::draw_brain_panel(ui, &ind.brain);
                        });
                        let lineage_hex = format!(
                            "#{:02X}{:02X}{:02X}",
                            ind.lineage_color[0], ind.lineage_color[1], ind.lineage_color[2]
                        );
                        ui.label(format!("Lineage Color: {}", lineage_hex));
                        let color_rect = egui::Rect::from_min_size(
                            ui.cursor().min,
                            egui::vec2(24.0, 12.0),
                        );
                        let (color_resp, color_painter) = ui.allocate_painter(color_rect.size(), egui::Sense::hover());
                        color_painter.rect_filled(
                            color_resp.rect,
                            2.0,
                            egui::Color32::from_rgb(
                                ind.lineage_color[0],
                                ind.lineage_color[1],
                                ind.lineage_color[2],
                            ),
                        );
                        ui.add_space(4.0);
                    }
                }
            }
        });

        // Right panel - Neural Network / Cladogram / Run config
        if self.show_nn_diagram || self.show_cladogram || self.show_run_config {
            egui::SidePanel::right("viz_panel")
                .min_width(400.0)
                .resizable(true)
                .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                if self.show_nn_diagram {
                    ui.heading("Neural Network");

                    if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                        if let Some(idx) = self.selected_individual_index(dish) {
                            if let Some(ind) = dish.simulation.individuals.get(idx) {
                                nn_viz::draw_brain_panel(ui, &ind.brain);
                                ui.separator();
                                ui.label(format!("Brain kind: {}", ind.brain.kind().as_str()));
                                ui.label(format!("Structure: {}", ind.brain.structure_signature()));
                                ui.label(format!("Complexity: {} nodes", ind.complexity()));
                                draw_brain_json_export_ui(
                                    ui,
                                    &self.session_id,
                                    self.last_brain_export_path.as_ref(),
                                );
                            } else {
                                ui.label("Select an individual to see its neural network");
                            }
                        } else {
                            ui.label("Click an individual on the dish (or use Select Random in Extant Genotypes).");
                        }
                    } else {
                        ui.label("Select an individual to see its neural network");
                    }
                }

                if self.show_run_config {
                    if self.show_nn_diagram || self.show_cladogram {
                        ui.separator();
                    }
                    if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                        draw_dish_run_config_panel(ui, dish);
                    } else {
                        ui.heading("Run Configuration");
                        ui.label("No active dish.");
                    }
                }

                if self.show_cladogram {
                    ui.separator();
                    ui.heading("Cladogram");
                    ui.label("Phylogenetic tree visualization");
                    ui.add(egui::Slider::new(&mut self.phylogeny_zoom, 0.2..=4.0).text("Phylogeny Zoom"));
                    ui.add(egui::Slider::new(&mut self.phylogeny_height, 140.0..=1200.0).text("Phylogeny Height"));
                    ui.add(egui::Slider::new(&mut self.phylogeny_max_render_nodes, 200..=10000).text("Max Render Nodes"));
                    
                    let mut next_draw_ids: Option<Vec<i32>> = None;
                    let mut next_y_rank: Option<HashMap<i32, f32>> = None;
                    if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                        ui.label(format!("Total nodes: {}", dish.cladogram.nodes.len()));
                        
                        if let Some(idx) = self.selected_individual_index(dish) {
                            if let Some(ind) = dish.simulation.individuals.get(idx) {
                                let lineage = dish.cladogram.get_lineage_path(ind.id);
                                ui.label(format!("Lineage depth: {}", lineage.len()));
                                ui.label(format!("Ancestor IDs: {:?}", &lineage[..lineage.len().min(10)]));
                            }
                        }

                        let area_h = self.phylogeny_height;

                        let mut ids: Vec<i32> = dish.cladogram.nodes.keys().copied().collect();
                        ids.sort_by_key(|id| {
                            let node = &dish.cladogram.nodes[id];
                            (node.birth_time, *id)
                        });

                        if !ids.is_empty() {
                            // Performance guardrail: for large trees, render a focused subset.
                            // Priority order is extant leaves + their ancestor paths, then recent nodes.
                            let original_total = ids.len();
                            if ids.len() > self.phylogeny_max_render_nodes {
                                let id_set: HashSet<i32> = ids.iter().copied().collect();
                                let mut extant: Vec<i32> = ids
                                    .iter()
                                    .copied()
                                    .filter(|id| dish.cladogram.nodes[id].death_time.is_none())
                                    .collect();
                                extant.sort_unstable_by_key(|id| {
                                    let n = &dish.cladogram.nodes[id];
                                    (-n.birth_time, -*id)
                                });

                                let mut keep: HashSet<i32> = HashSet::new();

                                for id in &self.phylogeny_last_draw_ids {
                                    if id_set.contains(id) {
                                        keep.insert(*id);
                                        if keep.len() >= self.phylogeny_max_render_nodes {
                                            break;
                                        }
                                    }
                                }

                                for id in &extant {
                                    let mut cur = *id;
                                    loop {
                                        keep.insert(cur);
                                        if keep.len() >= self.phylogeny_max_render_nodes {
                                            break;
                                        }
                                        if let Some(parent_id) = dish.cladogram.nodes[&cur].parent_id {
                                            cur = parent_id;
                                        } else {
                                            break;
                                        }
                                    }
                                    if keep.len() >= self.phylogeny_max_render_nodes {
                                        break;
                                    }
                                }

                                if keep.len() < self.phylogeny_max_render_nodes {
                                    let mut recent = ids.clone();
                                    recent.sort_unstable_by_key(|id| {
                                        let n = &dish.cladogram.nodes[id];
                                        (-n.birth_time, -*id)
                                    });
                                    for id in recent {
                                        keep.insert(id);
                                        if keep.len() >= self.phylogeny_max_render_nodes {
                                            break;
                                        }
                                    }
                                }

                                ids.retain(|id| keep.contains(id));
                                ids.sort_by_key(|id| {
                                    let node = &dish.cladogram.nodes[id];
                                    (node.birth_time, *id)
                                });
                            }

                            next_draw_ids = Some(ids.clone());

                            ui.label(format!("Showing {} / {} nodes", ids.len(), original_total));

                            // Zoom applies to both axes for quick magnification; the surrounding
                            // ScrollArea lets users navigate large pedigrees without clipping.
                            let canvas_w = ((ui.available_width() - 8.0).max(200.0) * self.phylogeny_zoom).max(220.0);
                            let canvas_h = (area_h.max(140.0) * self.phylogeny_zoom.max(0.2)).max(160.0);

                            egui::ScrollArea::both().show(ui, |ui| {
                                let (response, painter) = ui.allocate_painter(
                                    egui::vec2(canvas_w, canvas_h),
                                    egui::Sense::hover(),
                                );

                                let left = response.rect.left() + 12.0;
                                let right = response.rect.right() - 12.0;
                                let top = response.rect.top() + 10.0;
                                let bottom = response.rect.bottom() - 20.0;

                                let mut roots: Vec<i32> = ids
                                .iter()
                                .copied()
                                .filter(|id| dish.cladogram.nodes[id].parent_id.is_none())
                                .collect();
                                roots.sort_unstable();

                                let draw_set: HashSet<i32> = ids.iter().copied().collect();

                                let mut depth: HashMap<i32, usize> = HashMap::new();
                                let mut stack: Vec<(i32, usize)> = roots.iter().map(|id| (*id, 0usize)).collect();
                                while let Some((id, d)) = stack.pop() {
                                let insert = match depth.get(&id) {
                                    Some(prev) => d < *prev,
                                    None => true,
                                };
                                if !insert {
                                    continue;
                                }
                                depth.insert(id, d);
                                let mut children = dish.cladogram.nodes[&id].children.clone();
                                children.sort_unstable_by_key(|child_id| {
                                    let n = &dish.cladogram.nodes[child_id];
                                    (n.birth_time, *child_id)
                                });
                                children.reverse();
                                for child_id in children {
                                    if draw_set.contains(&child_id) {
                                        stack.push((child_id, d + 1));
                                    }
                                }
                            }
                            for id in &ids {
                                depth.entry(*id).or_insert(0);
                            }

                                // Assign y by post-order traversal so leaves spread out and
                                // internal nodes are centered above their descendant spans.
                                fn assign_y(
                                id: i32,
                                nodes: &HashMap<i32, evo_movement_rust::cladogram::LineageNode>,
                                draw_set: &HashSet<i32>,
                                last_y_rank: &HashMap<i32, f32>,
                                y_map: &mut HashMap<i32, f32>,
                                next_y: &mut f32,
                            ) -> f32 {
                                if let Some(v) = y_map.get(&id) {
                                    return *v;
                                }
                                let node = &nodes[&id];
                                let mut children = node.children.clone();
                                children.sort_unstable_by(|a, b| {
                                    let ya = last_y_rank.get(a).copied().unwrap_or(f32::INFINITY);
                                    let yb = last_y_rank.get(b).copied().unwrap_or(f32::INFINITY);
                                    ya.partial_cmp(&yb)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                        .then_with(|| {
                                            let na = &nodes[a];
                                            let nb = &nodes[b];
                                            (na.birth_time, *a).cmp(&(nb.birth_time, *b))
                                        })
                                });
                                let mut child_ys = Vec::new();
                                for child_id in children {
                                    if draw_set.contains(&child_id) {
                                        child_ys.push(assign_y(child_id, nodes, draw_set, last_y_rank, y_map, next_y));
                                    }
                                }
                                let y_val = if child_ys.is_empty() {
                                    let y = *next_y;
                                    *next_y += 1.0;
                                    y
                                } else {
                                    child_ys.iter().sum::<f32>() / child_ys.len() as f32
                                };
                                y_map.insert(id, y_val);
                                y_val
                            }

                                let mut y_map: HashMap<i32, f32> = HashMap::new();
                                let mut next_y = 0.0f32;
                                for root in roots {
                                    let _ = assign_y(
                                        root,
                                        &dish.cladogram.nodes,
                                        &draw_set,
                                        &self.phylogeny_last_y_rank,
                                        &mut y_map,
                                        &mut next_y,
                                    );
                                }
                                for id in &ids {
                                    if !y_map.contains_key(id) {
                                        let _ = assign_y(
                                            *id,
                                            &dish.cladogram.nodes,
                                            &draw_set,
                                            &self.phylogeny_last_y_rank,
                                            &mut y_map,
                                            &mut next_y,
                                        );
                                    }
                                }
                                next_y_rank = Some(y_map.clone());

                                let mut effective_depth_max = depth.values().copied().max().unwrap_or(1);
                                let extant_ids: HashSet<i32> = dish
                                .cladogram
                                .nodes
                                .iter()
                                .filter_map(|(id, n)| {
                                    if n.death_time.is_none() && draw_set.contains(id) {
                                        Some(*id)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if !extant_ids.is_empty() {
                                effective_depth_max += 1;
                            }
                                let depth_span = effective_depth_max.max(1) as f32;
                                let y_span = (next_y.max(1.0) - 1.0).max(1.0);

                                let mut positions: HashMap<i32, egui::Pos2> = HashMap::new();
                                for id in &ids {
                                let d = *depth.get(id).unwrap_or(&0);
                                let draw_depth = if extant_ids.contains(id) {
                                    effective_depth_max
                                } else {
                                    d
                                };
                                let x = left + (draw_depth as f32 / depth_span) * (right - left);
                                let y_norm = y_map.get(id).copied().unwrap_or(0.0) / y_span;
                                let y = top + y_norm * (bottom - top);
                                positions.insert(*id, egui::pos2(x, y));
                            }

                                for id in &ids {
                                let node = &dish.cladogram.nodes[id];
                                if let Some(parent_id) = node.parent_id {
                                    if let (Some(parent_pos), Some(child_pos)) =
                                        (positions.get(&parent_id), positions.get(id))
                                    {
                                        let branch_color = egui::Color32::from_rgb(
                                            node.color[0],
                                            node.color[1],
                                            node.color[2],
                                        );
                                        let branch_width = (1.2 + node.genetic_distance_to_parent as f32 * 2.0)
                                            .clamp(1.0, 4.0);
                                        // Pedigree elbow segments keep branches readable in dense trees.
                                        let elbow = egui::pos2(child_pos.x, parent_pos.y);
                                        painter.line_segment(
                                            [*parent_pos, elbow],
                                            egui::Stroke::new(branch_width, branch_color),
                                        );
                                        painter.line_segment(
                                            [elbow, *child_pos],
                                            egui::Stroke::new(branch_width, branch_color),
                                        );
                                    }
                                }
                            }

                                for id in &ids {
                                let node = &dish.cladogram.nodes[id];
                                if let Some(pos) = positions.get(id) {
                                    let is_extant = node.death_time.is_none();
                                    let mut radius = if is_extant { 4.0 } else { 3.0 };
                                    let mut stroke = egui::Stroke::new(1.0, egui::Color32::BLACK);
                                    if self.selected_individual_id == Some(*id) {
                                        radius = 6.0;
                                        stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
                                    }
                                    painter.circle_filled(
                                        *pos,
                                        radius,
                                        egui::Color32::from_rgb(
                                            node.color[0],
                                            node.color[1],
                                            node.color[2],
                                        ),
                                    );
                                    painter.circle_stroke(*pos, radius, stroke);

                                    if is_extant {
                                        painter.text(
                                            egui::pos2(pos.x + 5.0, pos.y),
                                            egui::Align2::LEFT_CENTER,
                                            format!("{}", id),
                                            egui::FontId::proportional(9.0),
                                            egui::Color32::LIGHT_GRAY,
                                        );
                                    }
                                }
                            }

                                painter.text(
                                egui::pos2(left, bottom + 2.0),
                                egui::Align2::LEFT_TOP,
                                "Pedigree view (extant organisms shown as terminal tips)",
                                egui::FontId::proportional(10.0),
                                egui::Color32::GRAY,
                            );
                            });
                        }
                    }
                    if let Some(ids) = next_draw_ids {
                        self.phylogeny_last_draw_ids = ids;
                    }
                    if let Some(y_rank) = next_y_rank {
                        self.phylogeny_last_y_rank = y_rank;
                    }
                }
                });
            });
        }

        // Central panel - Main visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.show_all_dishes {
                ui.heading("Arena — All Petri Dishes");
                ui.label("Click a dish preview to select it. Drag individuals on the sandbox dish.");
                let cols = 2usize.max((self.petri_dishes.len() + 1) / 2);
                egui::Grid::new("all_dishes_grid")
                    .num_columns(cols.min(3))
                    .spacing([12.0, 10.0])
                    .show(ui, |ui| {
                        let total = self.petri_dishes.len();
                        let tile_size = egui::vec2(320.0, 240.0);
                        for i in 0..total {
                            ui.vertical(|ui| {
                                let dish = &self.petri_dishes[i];
                                let selected_marker = if i == self.selected_dish { "▶ " } else { "  " };
                                ui.heading(format!("{selected_marker}{}", dish.name));
                                ui.label(dish_info_line(dish));
                                if dish.is_sandbox {
                                    ui.colored_label(egui::Color32::LIGHT_YELLOW, "Manual experiment dish");
                                }
                                self.draw_dish_thumbnail(ui, i, tile_size);
                            });
                            if (i + 1) % cols.min(3) == 0 {
                                ui.end_row();
                            }
                        }
                    });
                ui.separator();
            }

            let dish_name = if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                dish.name.clone()
            } else {
                return;
            };

            let is_sandbox = self
                .petri_dishes
                .get(self.selected_dish)
                .map(|d| d.is_sandbox)
                .unwrap_or(false);
            let dish_info = self
                .petri_dishes
                .get(self.selected_dish)
                .map(dish_info_line)
                .unwrap_or_default();
            let selected_dish_idx = self.selected_dish;

            ui.heading(&dish_name);
            ui.horizontal(|ui| {
                ui.label(&dish_info);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("+").on_hover_text("Zoom in").clicked() {
                        self.zoom = (self.zoom + 0.15).min(3.0);
                    }
                    ui.label(format!("{:.0}%", self.zoom * 100.0));
                    if ui.small_button("−").on_hover_text("Zoom out").clicked() {
                        self.zoom = (self.zoom - 0.15).max(0.25);
                    }
                    ui.label("Zoom:");
                });
            });
            if self.hex_background_mode == HexBackgroundMode::Competition {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 140, 80),
                    "Background: competition heatmap (dark = empty, warm = crowded)",
                );
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(170, 170, 120),
                    "Background: environmental resource distribution",
                );
            }
            if is_sandbox {
                let seed = ui.button("Seed random population").clicked();
                if seed {
                    if let Some(d) = self.petri_dishes.get_mut(selected_dish_idx) {
                        d.init_population(d.simulation.config.initial_population as usize);
                    }
                }
                ui.label("Drag individuals within this dish to reposition them.");
            }

            let dish_opt = self.petri_dishes.get(self.selected_dish);
            if dish_opt.is_none() {
                return;
            }
            let dish = dish_opt.unwrap();

            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for coord in dish.hex_grid.cells.keys() {
                let (x, y) = coord.to_pixel(dish.hex_grid.hex_size);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }

            let (min_x, max_x, min_y, max_y) = if min_x.is_finite() {
                (min_x, max_x, min_y, max_y)
            } else {
                (0.0, 1.0, 0.0, 1.0)
            };

            let width_px = (max_x - min_x) + 2.0 * dish.hex_grid.hex_size;
            let height_px = (max_y - min_y) + 2.0 * dish.hex_grid.hex_size;
            let available_size = ui.available_size();
            let scale = (available_size.x / width_px).min(available_size.y / height_px) * self.zoom.max(0.01);
            let hex_size = dish.hex_grid.hex_size;
            let padding = hex_size;
            let cells = &dish.hex_grid.cells;
            let individuals = &dish.simulation.individuals;
            let selected_ind_id = self.selected_individual_id;
            let move_trails = &dish.move_trails;
            let active_coords = &dish.active_coords;
            let max_occupants = max_cell_occupancy(cells);
            let bg_mode = self.hex_background_mode;

            let mut clicked_hex: Option<HexCoord> = None;
            let mut dragged_hex: Option<HexCoord> = None;
            let mut drag_active = false;
            let mut drag_stopped = false;

            egui::ScrollArea::both().show(ui, |ui| {
                let (response, painter) = ui.allocate_painter(
                    egui::vec2(width_px * scale, height_px * scale),
                    egui::Sense::click_and_drag(),
                );

                // Draw hexagonal grid
                for (coord, cell) in cells {
                    let (x, y) = coord.to_pixel(hex_size);
                    let local_x = (x - min_x + padding) * scale;
                    let local_y = (y - min_y + padding) * scale;
                    let screen_pos = response.rect.min + egui::vec2(local_x, local_y);

                    // Draw hexagon background (resources or competition heatmap)
                    let hex_points = hexagon_points(screen_pos, hex_size * scale);
                    let color = hex_background_color(
                        bg_mode,
                        cell.resource_level,
                        cell.individual_indices.len(),
                        max_occupants,
                    );

                    painter.add(egui::Shape::convex_polygon(
                        hex_points,
                        color,
                        egui::Stroke::new(0.5, egui::Color32::GRAY),
                    ));

                    // In competition mode, still show lineage dots; in resource mode same as before.
                    if let Some(dom) = Self::dominant_lineage_color(dish, cell) {
                        painter.circle_filled(
                            screen_pos,
                            (3.0 * scale).max(2.2),
                            egui::Color32::from_rgb(dom[0], dom[1], dom[2]),
                        );

                        if let Some(sel_id) = selected_ind_id {
                            let selected_in_cell = cell.individual_indices.iter().any(|idx| {
                                individuals
                                    .get(*idx)
                                    .map(|ind| ind.id == sel_id)
                                    .unwrap_or(false)
                            });
                            if selected_in_cell {
                                painter.circle_stroke(
                                    screen_pos,
                                    (5.0 * scale).max(3.0),
                                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                                );
                            }
                        }
                    }
                }

                for trail in move_trails {
                    let alpha = (1.0_f32 - trail.age as f32 / TRAIL_MAX_AGE as f32).clamp(0.0, 1.0);
                    if alpha <= 0.0 {
                        continue;
                    }
                    let Some(from_coord) = active_coords.get(trail.from_pos) else {
                        continue;
                    };
                    let Some(to_coord) = active_coords.get(trail.to_pos) else {
                        continue;
                    };
                    let from_px = from_coord.to_pixel(hex_size);
                    let to_px = to_coord.to_pixel(hex_size);
                    let from_screen = response.rect.min
                        + egui::vec2(
                            (from_px.0 - min_x + padding) * scale,
                            (from_px.1 - min_y + padding) * scale,
                        );
                    let to_screen = response.rect.min
                        + egui::vec2(
                            (to_px.0 - min_x + padding) * scale,
                            (to_px.1 - min_y + padding) * scale,
                        );
                    let color = egui::Color32::from_rgba_unmultiplied(
                        trail.color[0],
                        trail.color[1],
                        trail.color[2],
                        (alpha * 180.0) as u8,
                    );
                    painter.line_segment(
                        [from_screen, to_screen],
                        egui::Stroke::new((2.0 * scale).max(1.0), color),
                    );
                    painter.circle_filled(
                        from_screen,
                        (2.0 * scale).max(1.5),
                        color,
                    );
                }

                // Handle clicks for selection
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let rel_pos = pos - response.rect.min;
                        let hex_coord = HexCoord::from_pixel(
                            (rel_pos.x / scale) - padding + min_x,
                            (rel_pos.y / scale) - padding + min_y,
                            hex_size,
                        );
                        clicked_hex = Some(hex_coord);
                    }
                }
                if response.secondary_clicked() {
                    self.selected_individual_id = None;
                    self.selected_hex = None;
                }

                if response.dragged() {
                    drag_active = true;
                    if let Some(pos) = response.interact_pointer_pos() {
                        let rel_pos = pos - response.rect.min;
                        let hex_coord = HexCoord::from_pixel(
                            (rel_pos.x / scale) - padding + min_x,
                            (rel_pos.y / scale) - padding + min_y,
                            hex_size,
                        );
                        dragged_hex = Some(hex_coord);
                    }
                }
                if response.drag_stopped() {
                    drag_stopped = true;
                }
            });

            // Handle selection outside of the closure
            if let Some(coord) = clicked_hex {
                if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                    // Always set the selected hex
                    self.selected_hex = Some(coord);
                    println!("Selected hex: ({}, {})", coord.q, coord.r);
                    
                    if let Some(cell) = dish.hex_grid.get_cell(coord) {
                        if !cell.individual_indices.is_empty() {
                            if let Some(ind) = dish.simulation.individuals.get(cell.individual_indices[0]) {
                                self.selected_individual_id = Some(ind.id);
                            }
                        } else {
                            // Clicked on empty hex, clear individual selection
                            self.selected_individual_id = None;
                        }
                    } else {
                        println!("Warning: Clicked hex ({}, {}) not found in grid", coord.q, coord.r);
                    }
                }
            }

            if drag_active {
                if let Some(selected_id) = self.selected_individual_id {
                    if !self.is_dragging_individual {
                        self.paused_before_drag = self.paused;
                    }
                    self.is_dragging_individual = true;
                    self.paused = true;

                    if let Some(coord) = dragged_hex {
                        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
                            if let Some(target_idx) = dish.coord_to_index.get(&coord).copied() {
                                if let Some(ind_idx) = dish
                                    .simulation
                                    .individuals
                                    .iter()
                                    .position(|ind| ind.id == selected_id)
                                {
                                    dish.simulation.individuals[ind_idx].position = target_idx;
                                    dish.rebuild_hex_occupancy();
                                }
                            }
                        }
                    }
                }
            }

            if drag_stopped && self.is_dragging_individual {
                self.is_dragging_individual = false;
                self.paused = self.paused_before_drag;
            }
        });

        // Run simulation if not paused (~24 FPS)
        let frame_interval = Duration::from_secs_f32(1.0 / TARGET_FPS);
        if !self.paused {
            let now = Instant::now();
            let should_step = match self.last_frame {
                None => true,
                Some(prev) => now.duration_since(prev) >= frame_interval,
            };
            if should_step {
                self.step_simulation();
                self.last_frame = Some(now);
            }
            if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
                dish.age_move_trails();
            }
            ctx.request_repaint_after(frame_interval);
        } else {
            self.last_frame = None;
        }
    }
}

fn hexagon_points(center: egui::Pos2, size: f32) -> Vec<egui::Pos2> {
    let mut points = Vec::new();
    for i in 0..6 {
        let angle = std::f32::consts::PI / 3.0 * i as f32 + std::f32::consts::PI / 6.0;
        let x = center.x + size * angle.cos();
        let y = center.y + size * angle.sin();
        points.push(egui::pos2(x, y));
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_direction_test_dish(direction_bias: f64) -> (PetriDish, HexCoord) {
        let mut cfg = Config::default();
        cfg.nn_nodes = 3;
        cfg.nn_inputs = 1;
        cfg.nn_layers = 1;
        cfg.nn_nodes2 = 0;
        cfg.movement_cost_factor = 0.0;
        cfg.maintenance_cost_per_node = 0.0;
        cfg.basal_metabolic_cost = 0.0;
        cfg.death_risk = 0.0;
        cfg.reproduction_threshold = 1e9;

        let radius = 1;
        let mut dish = PetriDish::new_hex_dish(0, radius, 10.0, cfg);
        let center = HexCoord::new(radius, radius);
        let center_idx = dish
            .coord_to_index
            .get(&center)
            .copied()
            .expect("center coordinate should exist");

        let mut nn = NeuralNetwork::new(3, 1, 1, 0);
        for w in nn.weights.iter_mut() {
            *w = 0.0;
        }
        for b in nn.biases.iter_mut() {
            *b = 0.0;
        }
        // Keep move_signal positive regardless of direction node contribution.
        nn.biases[0] = 2.0;
        nn.biases[1] = direction_bias;
        nn.biases[2] = 2.0;

        let mut ind = Individual::from_legacy_nn(nn, center_idx, 1);
        ind.resources = 10.0;
        dish.simulation.individuals.push(ind);
        dish.rebuild_hex_occupancy();
        (dish, center)
    }

    #[test]
    fn directional_movement_maps_to_expected_neighbor_sectors() {
        let cases = [
            (-5.0, 0usize), // ~0°
            (-0.35, 2usize), // ~120°
            (3.0, 5usize),  // ~300°
        ];

        for (direction_bias, expected_sector) in cases {
            let (mut dish, center) = make_direction_test_dish(direction_bias);
            dish.step(-10.0, 1.0);

            let pos_idx = dish.simulation.individuals[0].position;
            let moved_coord = dish.active_coords[pos_idx];
            let expected_coord = center.neighbors()[expected_sector];

            assert_eq!(
                moved_coord, expected_coord,
                "direction bias {} should map to sector {}",
                direction_bias, expected_sector
            );
        }
    }

    #[test]
    fn movement_is_suppressed_when_below_threshold() {
        let (mut dish, center) = make_direction_test_dish(-5.0);
        let center_idx = dish
            .coord_to_index
            .get(&center)
            .copied()
            .expect("center coordinate should exist");

        // direction_bias=-5 yields a positive move signal via node 0+2,
        // but set threshold above that signal to suppress movement.
        dish.step(2.0, 1.0);

        let pos_idx = dish.simulation.individuals[0].position;
        assert_eq!(pos_idx, center_idx, "individual should remain in place when move_signal <= threshold");
    }
}
