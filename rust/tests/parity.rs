//! CPU vs GPU-forward parity integration tests.
use evo_movement_rust::config::{ArchitecturePolicy, Config, GridTopology};
use evo_movement_rust::simulation::Simulation;

fn parity_config(policy: ArchitecturePolicy, topology: GridTopology) -> Config {
    let mut cfg = Config::default();
    cfg.seed = 42;
    cfg.steps = 20;
    cfg.initial_population = 8;
    cfg.max_population = 32;
    cfg.grid_width = 12;
    cfg.grid_height = 8;
    cfg.architecture_policy = policy;
    cfg.grid_topology = topology;
    cfg.write_outputs = false;
    cfg.show_progress = false;
    cfg.movement_cost_factor = 0.5;
    cfg.reproduction_threshold = 1e9;
    cfg.death_risk = 0.0;
    cfg.use_gpu_step = false;
    cfg
}

fn run_sim(cfg: &Config, steps: i32) -> Simulation {
    let mut sim = Simulation::new(cfg);
    sim.init_population(cfg.initial_population as usize);
    for _ in 0..steps {
        sim.step();
    }
    sim
}

fn assert_sim_parity(a: &Simulation, b: &Simulation) {
    assert_eq!(a.individuals.len(), b.individuals.len());
    for (ia, ib) in a.individuals.iter().zip(b.individuals.iter()) {
        assert_eq!(ia.id, ib.id, "id mismatch");
        assert_eq!(ia.position, ib.position, "position mismatch for id {}", ia.id);
        assert!(
            (ia.resources - ib.resources).abs() < 1e-9,
            "resources mismatch for id {}: {} vs {}",
            ia.id,
            ia.resources,
            ib.resources
        );
    }
    assert_eq!(a.env.resources.len(), b.env.resources.len());
    for (ea, eb) in a.env.resources.iter().zip(b.env.resources.iter()) {
        assert!((ea - eb).abs() < 1e-9, "env mismatch: {} vs {}", ea, eb);
    }
}

fn assert_forward_parity(policy: ArchitecturePolicy, topology: GridTopology, steps: i32) {
    let base = parity_config(policy, topology);
    let cpu = run_sim(&base, steps);
    let mut gpu_cfg = base.clone();
    gpu_cfg.use_gpu_step = true;
    let gpu_path = run_sim(&gpu_cfg, steps);
    assert_sim_parity(&cpu, &gpu_path);
}

#[test]
fn parity_ring_perceptron() {
    assert_forward_parity(ArchitecturePolicy::PerceptronOnly, GridTopology::Ring1d, 15);
}

#[test]
fn parity_ring_mlp() {
    let mut base = parity_config(ArchitecturePolicy::EvolveMlp, GridTopology::Ring1d);
    base.nn_layers = 2;
    base.nn_nodes = 4;
    base.nn_nodes2 = 2;
    let cpu = run_sim(&base, 12);
    let mut gpu_cfg = base.clone();
    gpu_cfg.use_gpu_step = true;
    assert_sim_parity(&cpu, &run_sim(&gpu_cfg, 12));
}

#[test]
fn parity_hex_mlp() {
    let mut base = parity_config(ArchitecturePolicy::EvolveMlp, GridTopology::Hex2d);
    base.nn_layers = 2;
    base.nn_nodes = 4;
    base.nn_nodes2 = 2;
    let cpu = run_sim(&base, 10);
    let mut gpu_cfg = base.clone();
    gpu_cfg.use_gpu_step = true;
    assert_sim_parity(&cpu, &run_sim(&gpu_cfg, 10));
}

#[test]
fn parity_spatial_graph() {
    assert_forward_parity(ArchitecturePolicy::EvolveSpatialGraph, GridTopology::Ring1d, 10);
}

#[test]
fn parity_goo() {
    assert_forward_parity(ArchitecturePolicy::EvolveGoo, GridTopology::Ring1d, 10);
}

#[test]
#[ignore = "requires CUDA device; run with --ignored"]
fn parity_ring_perceptron_cuda_device() {
    let mut cfg = parity_config(ArchitecturePolicy::PerceptronOnly, GridTopology::Ring1d);
    cfg.use_gpu_step = true;
    let cpu = run_sim(
        &{
            let mut c = cfg.clone();
            c.use_gpu_step = false;
            c
        },
        10,
    );
    let gpu = run_sim(&cfg, 10);
    assert_sim_parity(&cpu, &gpu);
}
