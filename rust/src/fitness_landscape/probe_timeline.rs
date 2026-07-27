//! Extract best-by-resources brains from staged sweep snapshots for probe timelines.

use crate::brain::goo_graph::GooGraphBrain;
use crate::brain::{BrainGenome, BrainInner, FounderWeightMode};
use crate::config::{ArchitecturePolicy, Config};
use crate::fitness_landscape::{architecture_label, SweepCell};
use crate::simulation::Simulation;
use crate::snapshot::{PopulationSnapshot, SnapshotIndividual};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CellKey {
    pub architecture: String,
    pub x_label: String,
    pub y_label: String,
    pub founder_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeMeta {
    pub brain_kind: String,
    pub movement_output_mode: String,
    pub movement_cost_factor: f64,
    pub resource_seed: u64,
    pub resource_scale: f64,
    pub resource_contrast: f64,
    pub env_perlin_spatial_scale: f64,
    pub env_perlin_time_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineTimepoint {
    pub steps: i32,
    pub label: String,
    pub brain_json: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub individual_id: Option<i32>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub missing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeTimeline {
    pub version: u32,
    pub best_metric: String,
    pub cell: TimelineCell,
    pub probe_meta: ProbeMeta,
    pub timepoints: Vec<TimelineTimepoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineCell {
    pub architecture: String,
    pub x_label: String,
    pub y_label: String,
    pub founder_mode: String,
    pub seed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineManifestEntry {
    pub file: String,
    pub architecture: String,
    pub x_label: String,
    pub y_label: String,
    pub founder_mode: String,
    pub seed: u32,
    pub available_steps: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineManifest {
    pub version: u32,
    pub entries: Vec<TimelineManifestEntry>,
}

#[derive(Debug, Clone)]
pub struct RunRef {
    pub architecture: String,
    pub x_label: String,
    pub y_label: String,
    pub founder_mode: String,
    pub seed: u32,
}

pub fn sanitize_filename_part(s: &str) -> String {
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

pub fn cell_key_from_run(run: &RunRef) -> CellKey {
    CellKey {
        architecture: run.architecture.clone(),
        x_label: run.x_label.clone(),
        y_label: run.y_label.clone(),
        founder_mode: run.founder_mode.clone(),
    }
}

pub fn snapshot_path(stage_dir: &Path, cell: &CellKey, seed: u32) -> PathBuf {
    stage_dir.join("snapshots").join(format!(
        "{}__{}__{}__{}__seed{}.cbor",
        sanitize_filename_part(&cell.architecture),
        sanitize_filename_part(&cell.x_label),
        sanitize_filename_part(&cell.y_label),
        sanitize_filename_part(&cell.founder_mode),
        seed,
    ))
}

pub fn timeline_json_filename(cell: &CellKey, seed: u32) -> String {
    format!(
        "{}__{}__{}__{}__seed{}.json",
        sanitize_filename_part(&cell.architecture),
        sanitize_filename_part(&cell.x_label),
        sanitize_filename_part(&cell.y_label),
        sanitize_filename_part(&cell.founder_mode),
        seed,
    )
}

pub fn find_sweep_cell<'a>(
    cells: &'a [SweepCell],
    architecture: &str,
    x_label: &str,
    y_label: &str,
) -> Option<&'a SweepCell> {
    cells.iter().find(|c| {
        architecture_label(&c.cfg) == architecture && c.x_label == x_label && c.y_label == y_label
    })
}

pub fn best_by_resources(snapshot: &PopulationSnapshot) -> Option<&SnapshotIndividual> {
    snapshot
        .individuals
        .iter()
        .filter(|ind| ind.brain_json.as_ref().is_some_and(|j| !j.is_empty()))
        .max_by(|a, b| {
            a.resources
                .partial_cmp(&b.resources)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

pub fn probe_meta_from_cfg(cfg: &Config, brain: &BrainGenome) -> ProbeMeta {
    ProbeMeta {
        brain_kind: brain.kind().as_str().to_string(),
        movement_output_mode: cfg.resolved_movement_output_mode().label().to_string(),
        movement_cost_factor: cfg.movement_cost_factor,
        resource_seed: cfg.seed as u64,
        resource_scale: cfg.resource_scale,
        resource_contrast: cfg.resource_contrast,
        env_perlin_spatial_scale: cfg.env_perlin_spatial_scale,
        env_perlin_time_scale: cfg.env_perlin_time_scale,
    }
}

fn prep_config_for_run(mut cfg: Config, seed: u32, cap_multiplier: f64) -> Config {
    cfg.seed = seed;
    cfg.write_outputs = false;
    cfg.write_snapshot = false;
    cfg.write_config = false;
    cfg.show_progress = false;
    cfg.output_run_dir.clear();
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

/// Reconstruct the t=0 founder brain exactly as the sweep initializes a fresh run.
pub fn founder_brain_json(
    cell: &SweepCell,
    founder_mode: FounderWeightMode,
    seed: u32,
    cap_multiplier: f64,
) -> io::Result<String> {
    let cfg = prep_config_for_run(cell.cfg.clone(), seed, cap_multiplier);
    let mut sim = Simulation::new(&cfg);
    sim.init_population_with_founder_mode(1, Some(founder_mode));
    seed_goo_founders(
        &mut sim,
        &cfg,
        cell.goo_generate_nodes,
        seed as u64,
    );
    let ind = sim.individuals.first().ok_or_else(|| {
        io::Error::new(io::ErrorKind::Other, "founder population empty")
    })?;
    serde_json::to_string(&ind.brain).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

fn founder_mode_from_label(label: &str) -> Option<FounderWeightMode> {
    match label.trim().to_ascii_lowercase().as_str() {
        "neutral" | "zero" => Some(FounderWeightMode::Neutral),
        "random" | "random_uniform" | "uniform" => Some(FounderWeightMode::RandomUniform),
        "random_small" | "small" => Some(FounderWeightMode::RandomSmall),
        _ => None,
    }
}

pub fn build_timeline(
    run: &RunRef,
    cell: &SweepCell,
    stages: &[i32],
    output_dir: &Path,
    cap_multiplier: f64,
) -> io::Result<ProbeTimeline> {
    let founder_mode = founder_mode_from_label(&run.founder_mode).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown founder mode: {}", run.founder_mode),
        )
    })?;

    let cfg = prep_config_for_run(cell.cfg.clone(), run.seed, cap_multiplier);
    let founder_json = founder_brain_json(cell, founder_mode, run.seed, cap_multiplier)?;
    let founder_brain: BrainGenome = serde_json::from_str(&founder_json).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, e)
    })?;
    let probe_meta = probe_meta_from_cfg(&cfg, &founder_brain);

    let mut timepoints = Vec::new();

    for &steps in stages {
        if steps == 0 {
            timepoints.push(TimelineTimepoint {
                steps: 0,
                label: "t=0 founder".to_string(),
                brain_json: founder_json.clone(),
                resources: None,
                individual_id: None,
                missing: false,
            });
            continue;
        }

        let stage_dir = output_dir.join(format!("stage_{steps}"));
        let snap_path = snapshot_path(&stage_dir, &cell_key_from_run(run), run.seed);
        if !snap_path.is_file() {
            timepoints.push(TimelineTimepoint {
                steps,
                label: format!("t={steps} best"),
                brain_json: String::new(),
                resources: None,
                individual_id: None,
                missing: true,
            });
            continue;
        }

        let snapshot = PopulationSnapshot::read_from_path(&snap_path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", snap_path.display()),
            )
        })?;

        match best_by_resources(&snapshot) {
            Some(best) => {
                timepoints.push(TimelineTimepoint {
                    steps,
                    label: format!("t={steps} best"),
                    brain_json: best.brain_json.clone().unwrap_or_default(),
                    resources: Some(best.resources),
                    individual_id: Some(best.id),
                    missing: false,
                });
            }
            None => {
                timepoints.push(TimelineTimepoint {
                    steps,
                    label: format!("t={steps} best"),
                    brain_json: String::new(),
                    resources: None,
                    individual_id: None,
                    missing: true,
                });
            }
        }
    }

    Ok(ProbeTimeline {
        version: 1,
        best_metric: "resources".to_string(),
        cell: TimelineCell {
            architecture: run.architecture.clone(),
            x_label: run.x_label.clone(),
            y_label: run.y_label.clone(),
            founder_mode: run.founder_mode.clone(),
            seed: run.seed,
        },
        probe_meta,
        timepoints,
    })
}

pub fn write_timeline(output_dir: &Path, timeline: &ProbeTimeline) -> io::Result<PathBuf> {
    let dir = output_dir.join("probe_timelines");
    fs::create_dir_all(&dir)?;
    let key = CellKey {
        architecture: timeline.cell.architecture.clone(),
        x_label: timeline.cell.x_label.clone(),
        y_label: timeline.cell.y_label.clone(),
        founder_mode: timeline.cell.founder_mode.clone(),
    };
    let filename = timeline_json_filename(&key, timeline.cell.seed);
    let path = dir.join(&filename);
    let json = serde_json::to_string_pretty(timeline)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(&path, json)?;
    Ok(path)
}

pub fn write_manifest(output_dir: &Path, entries: &[TimelineManifestEntry]) -> io::Result<()> {
    let dir = output_dir.join("probe_timelines");
    fs::create_dir_all(&dir)?;
    let manifest = TimelineManifest {
        version: 1,
        entries: entries.to_vec(),
    };
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(dir.join("manifest.json"), json)
}

pub fn timeline_has_data(timeline: &ProbeTimeline) -> bool {
    timeline
        .timepoints
        .iter()
        .any(|tp| !tp.missing && !tp.brain_json.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::PopulationSnapshot;

    fn sample_individual(id: i32, resources: f32, brain_json: Option<String>) -> SnapshotIndividual {
        SnapshotIndividual {
            id,
            position: 0,
            resources,
            age: 1,
            nn_nodes: 0,
            nn_nodes2: 0,
            nn_inputs: 8,
            nn_layers: 1,
            weights: Vec::new(),
            weights_multi: Vec::new(),
            biases: Vec::new(),
            weights2: Vec::new(),
            biases2: Vec::new(),
            brain_kind: None,
            brain_json,
        }
    }

    #[test]
    fn best_by_resources_picks_highest() {
        let snap = PopulationSnapshot {
            version: 2,
            timestep: 10000,
            rng_seed_hint: 0,
            next_id: 1,
            env_width: 10,
            env_height: 10,
            hex_dish_radius: 0,
            env_perlin_time: 0.0,
            env_resources: Vec::new(),
            env_perlin_base: Vec::new(),
            individuals: vec![
                sample_individual(1, 5.0, Some(r#"{"kind":"perceptron"}"#.to_string())),
                sample_individual(2, 12.0, Some(r#"{"kind":"perceptron"}"#.to_string())),
                sample_individual(3, 8.0, Some(r#"{"kind":"perceptron"}"#.to_string())),
            ],
        };
        let best = best_by_resources(&snap).unwrap();
        assert_eq!(best.id, 2);
        assert_eq!(best.resources, 12.0);
    }

    #[test]
    fn best_by_resources_skips_empty_brain_json() {
        let snap = PopulationSnapshot {
            version: 2,
            timestep: 10000,
            rng_seed_hint: 0,
            next_id: 1,
            env_width: 10,
            env_height: 10,
            hex_dish_radius: 0,
            env_perlin_time: 0.0,
            env_resources: Vec::new(),
            env_perlin_base: Vec::new(),
            individuals: vec![
                sample_individual(1, 99.0, None),
                sample_individual(2, 5.0, Some(r#"{"kind":"perceptron"}"#.to_string())),
            ],
        };
        let best = best_by_resources(&snap).unwrap();
        assert_eq!(best.id, 2);
    }

    #[test]
    fn sanitize_and_snapshot_path_roundtrip() {
        let cell = CellKey {
            architecture: "Perceptron".to_string(),
            x_label: "nodes=4".to_string(),
            y_label: "baseline".to_string(),
            founder_mode: "random_uniform".to_string(),
        };
        let path = snapshot_path(Path::new("stage_10000"), &cell, 244);
        assert!(path.to_string_lossy().contains("Perceptron__nodes=4__baseline__random_uniform__seed244.cbor"));
        let fname = timeline_json_filename(&cell, 244);
        assert!(fname.ends_with("__seed244.json"));
    }

    #[test]
    fn founder_brain_json_parses() {
        let cfg = Config::default();
        let cell = SweepCell {
            cfg,
            x_label: "nodes=1".to_string(),
            y_label: "baseline".to_string(),
            goo_generate_nodes: 1,
            diagram_seed: 0,
        };
        let json = founder_brain_json(
            &cell,
            FounderWeightMode::RandomUniform,
            42,
            1.0,
        )
        .expect("founder brain");
        crate::brain_playground::parse_brain_json(&json).expect("valid brain json");
    }
}
