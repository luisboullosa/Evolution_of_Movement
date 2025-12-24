use evo_movement_rust::{config::Config, simulation::Simulation};
use std::env;
use std::ffi::OsStr;
use std::path::Path;

fn sanitize_basename(p: &str) -> String {
    let path = Path::new(p);
    if path.is_absolute() {
        if let Some(b) = path.file_name().and_then(OsStr::to_str) {
            return b.to_string();
        }
    }
    // fallback: return as-is
    p.to_string()
}

fn print_help() {
    println!("evo-movement-rust usage:");
    println!("  --config-file=PATH            Load config overrides from PATH (JSON)");
    println!("  --env-perlin-spatial-scale=N  Perlin spatial scale (double)");
    println!("  --env-perlin-time-scale=N     Perlin temporal scale (double)");
    println!("  --env-time-speed=N            Perlin time speed (double)");
    println!("  --env-time-amplitude=N        Perlin time amplitude (double)");
    println!("  --env-change-rate=N           Gaussian env noise sigma (double)");
    println!("  --perlin-time / --no-perlin-time    Enable/disable perlin-time evolution");
    println!("  --seed=N                      RNG seed (unsigned)");
    println!("  --steps=N                     Number of timesteps to run (int)");
    println!("  --report-interval=N           Reporting interval (int)");
    println!("  --initial-population=N        Starting population size (int)");
    println!("  --show-progress / --no-progress  Show textual progress bar and enable 'q' to stop");
    println!("  --print-frames / --no-print-frames  Print ASCII frames to stdout");
    println!("  --config-out=PATH             Write resolved config to PATH inside sim dir");
    println!("  --help, -h                    Show this help message");
}

fn parse_arg_value<T: std::str::FromStr>(arg: &str, flag: &str) -> Option<T> {
    if arg.starts_with(flag) {
        if arg.len() > flag.len() && arg.as_bytes()[flag.len()] == b'=' {
            return arg[flag.len() + 1..].parse::<T>().ok();
        }
    }
    None
}

fn main() {
    let mut args = env::args().skip(1);
    let mut cfg: Option<Config> = None;

    // first pass: capture explicit --config-file if provided
    let mut remaining: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        if a == "--help" || a == "-h" {
            print_help();
            return;
        }
        if a.starts_with("--config-file") {
            if let Some(p) = if a.contains('=') {
                a.splitn(2, '=').nth(1).map(|s| s.to_string())
            } else {
                args.next()
            } {
                // try load from file (panic on error to surface mistakes)
                match Config::from_file(&p) {
                    Ok(c) => cfg = Some(c),
                    Err(e) => {
                        eprintln!("Failed to load config from {}: {}", p, e);
                        return;
                    }
                }
            }
            continue;
        }
        remaining.push(a);
    }

    // start with loaded or default config
    let mut cfg = cfg.unwrap_or_else(|| Config::default());

    // re-parse remaining args to apply overrides
    let mut it = remaining.into_iter();
    while let Some(a) = it.next() {
        if a == "--help" || a == "-h" {
            print_help();
            return;
        }

        if a.starts_with("--env-perlin-spatial-scale") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--env-perlin-spatial-scale") {
                cfg.env_perlin_spatial_scale = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.env_perlin_spatial_scale = v;
                    continue;
                }
            }
        }

        if a.starts_with("--env-perlin-time-scale") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--env-perlin-time-scale") {
                cfg.env_perlin_time_scale = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.env_perlin_time_scale = v;
                    continue;
                }
            }
        }

        if a.starts_with("--env-time-speed") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--env-time-speed") {
                cfg.env_time_speed = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.env_time_speed = v;
                    continue;
                }
            }
        }

        if a.starts_with("--env-time-amplitude") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--env-time-amplitude") {
                cfg.env_time_amplitude = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.env_time_amplitude = v;
                    continue;
                }
            }
        }

        if a.starts_with("--env-change-rate") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--env-change-rate") {
                cfg.env_change_rate = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.env_change_rate = v;
                    continue;
                }
            }
        }

        if a.starts_with("--movement-cost-factor") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--movement-cost-factor") {
                cfg.movement_cost_factor = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.movement_cost_factor = v;
                    continue;
                }
            }
        }

        if a.starts_with("--maintenance-cost-per-node") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--maintenance-cost-per-node") {
                cfg.maintenance_cost_per_node = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.maintenance_cost_per_node = v;
                    continue;
                }
            }
        }

        if a == "--no-perlin-time" {
            cfg.env_use_perlin_time = false;
            continue;
        }
        if a == "--perlin-time" {
            cfg.env_use_perlin_time = true;
            continue;
        }
        if a == "--no-progress" {
            cfg.show_progress = false;
            continue;
        }
        if a == "--show-progress" {
            cfg.show_progress = true;
            continue;
        }

        if a.starts_with("--seed") {
            if let Some(v) = parse_arg_value::<u32>(&a, "--seed") {
                cfg.seed = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<u32>() {
                    cfg.seed = v;
                    continue;
                }
            }
        }

        if a.starts_with("--steps") {
            if let Some(v) = parse_arg_value::<i32>(&a, "--steps") {
                cfg.steps = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<i32>() {
                    cfg.steps = v;
                    continue;
                }
            }
        }

        if a.starts_with("--report-interval") {
            if let Some(v) = parse_arg_value::<i32>(&a, "--report-interval") {
                cfg.report_interval = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<i32>() {
                    cfg.report_interval = v;
                    continue;
                }
            }
        }

        if a.starts_with("--initial-population") {
            if let Some(v) = parse_arg_value::<i32>(&a, "--initial-population") {
                cfg.initial_population = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<i32>() {
                    cfg.initial_population = v;
                    continue;
                }
            }
        }

        if a.starts_with("--max-population") {
            if let Some(v) = parse_arg_value::<i32>(&a, "--max-population") {
                cfg.max_population = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<i32>() {
                    cfg.max_population = v;
                    continue;
                }
            }
        }

        if a.starts_with("--death-risk") {
            if let Some(v) = parse_arg_value::<f64>(&a, "--death-risk") {
                cfg.death_risk = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<f64>() {
                    cfg.death_risk = v;
                    continue;
                }
            }
        }

        if a.starts_with("--config-out") {
            cfg.write_config = true;
            if a.contains('=') {
                if let Some(v) = a.splitn(2, '=').nth(1) {
                    cfg.config_out_path = sanitize_basename(v);
                }
            } else if let Some(p) = it.next() {
                cfg.config_out_path = sanitize_basename(&p);
            }
            continue;
        }

        if a.starts_with("--burn-in") {
            if let Some(v) = parse_arg_value::<i32>(&a, "--burn-in") {
                cfg.burn_in = v;
                continue;
            } else if let Some(p) = it.next() {
                if let Ok(v) = p.parse::<i32>() {
                    cfg.burn_in = v;
                    continue;
                }
            }
        }

        if a == "--print-frames" {
            cfg.print_frames = true;
            continue;
        }
        if a == "--no-print-frames" {
            cfg.print_frames = false;
            continue;
        }

        // Try parse as seed if arg is numeric
        if let Ok(v) = a.parse::<u32>() {
            cfg.seed = v;
            continue;
        }
        // otherwise ignore unknown flag
    }

    // validate config
    cfg.validate();

    // create and run simulation
    let mut sim = Simulation::new(&cfg);
    sim.init_population(cfg.initial_population as usize);
    sim.run();
}
