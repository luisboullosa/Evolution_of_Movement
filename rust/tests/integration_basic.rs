use evo_movement_rust::config::Config;
use evo_movement_rust::simulation::Simulation;
use std::fs;

#[test]
fn integration_simulation_basic() {
    let mut cfg = Config::default();
    cfg.grid_width = 6;
    cfg.nn_nodes = 2;
    cfg.nn_inputs = 1;
    cfg.nn_layers = 1;
    cfg.nn_nodes2 = 0;
    cfg.randomize_start_positions = true;
    cfg.radiate = 0;
    cfg.seed = 13;
    let mut sim = Simulation::new(&cfg);
    sim.init_population(4);
    // run a few steps to exercise core loop
    for _ in 0..3 {
        sim.step();
    }
    assert!(sim.individuals.len() >= 4);
}

#[test]
fn integration_config_from_file() {
    let mut tmp = std::env::temp_dir();
    tmp.push("evo_movement_test_config.json");
    let contents = r#"{ "grid_width": 3, "steps": 2 }"#;
    let _ = fs::write(&tmp, contents);
    let cfg = Config::from_file(&tmp).expect("should load config");
    assert_eq!(cfg.grid_width, 3);
    assert_eq!(cfg.steps, 2);
    let _ = fs::remove_file(&tmp);
}

#[test]
fn integration_resource_consumption_and_reproduction() {
    let mut cfg = Config::default();
    cfg.grid_width = 1;
    cfg.nn_nodes = 1;
    cfg.nn_inputs = 1;
    cfg.nn_layers = 1;
    cfg.nn_nodes2 = 0;
    cfg.randomize_start_positions = false;
    cfg.start_pos = 0;
    cfg.radiate = 0;
    cfg.seed = 123;
    cfg.initial_population = 1;
    cfg.reproduction_threshold = 10.5;
    cfg.reproduction_cost = 1.0;
    cfg.maintenance_cost_per_node = 0.0;
    cfg.movement_cost_factor = 0.0;
    cfg.death_risk = 0.0;
    cfg.env_time_speed = 0.0;
    cfg.env_time_amplitude = 0.0;
    cfg.resource_scale = 1.0;

    let mut sim = Simulation::new(&cfg);
    sim.init_population(1);

    sim.env.set_perlin_base(&[2.0]);

    let parent = sim.individuals.get_mut(0).expect("expected initial individual");
    parent.resources = 10.0;
    parent.nn.weights[0] = 0.0;
    parent.nn.biases[0] = 0.0;

    sim.step();

    assert_eq!(sim.individuals.len(), 2, "expected one offspring after reproduction");
    assert!(sim.env.resources[0] <= 1e-9, "expected gathered resources to be depleted from environment");
}
