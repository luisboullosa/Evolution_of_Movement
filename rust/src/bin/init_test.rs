use evo_movement_rust::{config::Config, simulation::Simulation};
use std::path::Path;

fn main() {
    use std::env;
    let mut args = env::args().skip(1);
    let mut cfg_path: Option<String> = None;
    while let Some(a) = args.next() {
        if a == "-c" || a == "--config" {
            if let Some(p) = args.next() {
                cfg_path = Some(p);
            }
        } else if a.starts_with("--config=") {
            cfg_path = a.splitn(2, '=').nth(1).map(|s| s.to_string());
        }
    }
    let cfg = if let Some(p) = cfg_path {
        let path = Path::new(&p);
        if path.exists() {
            Config::try_load(path)
        } else {
            Config::default()
        }
    } else {
        let p1 = Path::new("input/config_test.json");
        let p2 = Path::new("../input/config_test.json");
        if p1.exists() {
            Config::try_load(p1)
        } else if p2.exists() {
            Config::try_load(p2)
        } else {
            Config::default()
        }
    };
    println!(
        "Loaded config for init test: steps={}, grid_width={}, initial_population={}",
        cfg.steps, cfg.grid_width, cfg.initial_population
    );

    let mut cfg2 = cfg.clone();
    cfg2.steps = 1;
    cfg2.initial_population = 10;
    cfg2.show_progress = false; // avoid interactive input thread for tests

    let mut sim = Simulation::new(&cfg2);
    sim.init_population(cfg2.initial_population as usize);
    println!(
        "Initialized sim: individuals={}, env_width={}",
        sim.individuals.len(),
        sim.env.width
    );
}
