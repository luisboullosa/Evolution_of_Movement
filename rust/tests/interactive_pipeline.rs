//! Headless sim -> snapshot -> interactive-load path for all brain architecture policies.
use evo_movement_rust::brain::BrainGenome;
use evo_movement_rust::config::{ArchitecturePolicy, Config, GridTopology};
use evo_movement_rust::simulation::Simulation;
use evo_movement_rust::snapshot::{PopulationSnapshot, SnapshotIndividual};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn pipeline_config(policy: ArchitecturePolicy, seed: u32) -> Config {
    let mut cfg = Config::default();
    cfg.seed = seed;
    cfg.steps = 120;
    cfg.initial_population = 40;
    cfg.max_population = 120;
    cfg.grid_width = 18;
    cfg.grid_height = 18;
    cfg.grid_topology = GridTopology::Hex2d;
    cfg.architecture_policy = policy;
    cfg.architecture_mutation_rate = 0.03;
    cfg.max_brain_nodes = 24;
    cfg.nn_nodes = 4;
    cfg.nn_layers = 2;
    cfg.nn_nodes2 = 4;
    cfg.nn_inputs = 8;
    cfg.movement_cost_factor = 0.5;
    cfg.reproduction_threshold = 20.0;
    cfg.write_outputs = false;
    cfg.show_progress = false;
    cfg.use_gpu_step = false;
    cfg.write_snapshot = true;
    cfg.snapshot_filename = "snapshot.cbor".to_string();
    cfg
}

fn genotype_signature_from_snapshot(rec: &SnapshotIndividual) -> String {
    if let Some(ref json) = rec.brain_json {
        if let Some(sig) = BrainGenome::structure_signature_from_brain_json(json) {
            return sig;
        }
    }
    if rec.nn_layers >= 2 || rec.weights2.iter().any(|&w| w != 0.0) {
        format!(
            "m:{}:{}:{}:{}",
            rec.nn_nodes, rec.nn_inputs, rec.nn_nodes2, rec.nn_layers
        )
    } else {
        format!("p:{}:{}", rec.nn_inputs, rec.nn_nodes.max(1))
    }
}

fn brain_kind_from_snapshot(rec: &SnapshotIndividual) -> Option<String> {
    if let Some(kind) = &rec.brain_kind {
        return Some(kind.clone());
    }
    rec.brain_json.as_ref().and_then(|json| {
        serde_json::from_str::<serde_json::Value>(json)
            .ok()
            .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(str::to_string))
    })
}

fn run_and_snapshot(policy: ArchitecturePolicy, seed: u32, dir: &PathBuf) -> PopulationSnapshot {
    std::fs::create_dir_all(dir).expect("create temp output dir");
    let mut cfg = pipeline_config(policy, seed);
    cfg.output_run_dir = dir.to_string_lossy().into_owned();

    let mut sim = Simulation::new(&cfg);
    sim.init_population(cfg.initial_population as usize);
    sim.run();

    let snap_path = dir.join("snapshot.cbor");
    assert!(
        snap_path.exists(),
        "snapshot missing for {:?} at {}",
        policy,
        snap_path.display()
    );
    PopulationSnapshot::read_from_path(&snap_path).expect("read snapshot")
}

fn assert_interactive_load_path(snapshot: &PopulationSnapshot, label: &str) {
    assert!(
        !snapshot.individuals.is_empty(),
        "{label}: snapshot has no individuals"
    );

    let mut kinds = HashSet::new();
    let mut sigs = HashMap::<String, usize>::new();
    for rec in &snapshot.individuals {
        let sig = genotype_signature_from_snapshot(rec);
        *sigs.entry(sig).or_insert(0) += 1;
        if let Some(kind) = brain_kind_from_snapshot(rec) {
            kinds.insert(kind);
        }
        if let Some(ref json) = rec.brain_json {
            let parsed = BrainGenome::structure_signature_from_brain_json(json)
                .expect("{label}: lightweight signature parse failed");
            let brain: BrainGenome =
                serde_json::from_str(json).expect("{label}: full brain deserialize failed");
            assert_eq!(
                parsed,
                brain.structure_signature(),
                "{label}: signature mismatch"
            );
        }
    }

    assert!(
        sigs.len() >= 1,
        "{label}: expected at least one genotype bucket"
    );
    eprintln!(
        "[pipeline] {label}: {} individuals, {} genotypes, kinds {:?}",
        snapshot.individuals.len(),
        sigs.len(),
        kinds
    );

    // Subsample path (interactive caps at max_population)
    let display_cap = 20usize;
    let loaded: Vec<_> = snapshot
        .individuals
        .iter()
        .take(display_cap)
        .map(|rec| rec.clone().into_individual())
        .collect();
    assert_eq!(loaded.len(), display_cap.min(snapshot.individuals.len()));
    for mut ind in loaded {
        let _ = ind.brain.structure_signature();
        let _ = ind.decide_action(
            [1.0; 6],
            0,
            0.1,
            &mut rand::thread_rng(),
            &mut [0.0; 64],
            &mut [0.0; 64],
        );
    }
}

#[test]
fn pipeline_perceptron_snapshot_to_interactive() {
    let dir = std::env::temp_dir().join("evo_pipeline_perceptron");
    let snap = run_and_snapshot(ArchitecturePolicy::PerceptronOnly, 42, &dir);
    assert_interactive_load_path(&snap, "perceptron");
    assert!(
        snap.individuals.iter().all(|r| {
            brain_kind_from_snapshot(r).as_deref() == Some("perceptron")
        }),
        "perceptron policy should only produce perceptron brains"
    );
}

#[test]
fn pipeline_mlp_snapshot_to_interactive() {
    let dir = std::env::temp_dir().join("evo_pipeline_mlp");
    let snap = run_and_snapshot(ArchitecturePolicy::EvolveMlp, 43, &dir);
    assert_interactive_load_path(&snap, "mlp");
}

#[test]
fn pipeline_spatial_graph_snapshot_to_interactive() {
    let dir = std::env::temp_dir().join("evo_pipeline_spatial");
    let snap = run_and_snapshot(ArchitecturePolicy::EvolveSpatialGraph, 44, &dir);
    assert_interactive_load_path(&snap, "spatial_graph");
}

#[test]
fn pipeline_goo_snapshot_to_interactive() {
    let dir = std::env::temp_dir().join("evo_pipeline_goo");
    let snap = run_and_snapshot(ArchitecturePolicy::EvolveGoo, 45, &dir);
    assert_interactive_load_path(&snap, "goo");
}
