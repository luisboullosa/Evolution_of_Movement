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
