//! Post-process staged sweep snapshots into probe timeline JSON for the fitness landscape viewer.

use evo_movement_rust::config::Config;
use evo_movement_rust::fitness_landscape::{
    build_cells_to_sweep, build_timeline, expand_output_mode_cells, find_sweep_cell,
    timeline_has_data, write_manifest, write_timeline, RunRef, SweepProfile, TimelineManifestEntry,
};
use serde::Deserialize;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
struct StateRun {
    architecture: String,
    x_label: String,
    y_label: String,
    founder_mode: String,
    seed: u32,
}

#[derive(Debug, Deserialize)]
struct StagedSweepState {
    #[serde(default)]
    all_runs: Vec<StateRun>,
}

fn print_help() {
    eprintln!(
        "Usage: evo_probe_timeline --output-dir PATH [options]

Build probe_timelines/*.json from staged sweep snapshots (best individual by resources).

Options:
  --output-dir PATH           Sweep output directory (required)
  --stages 0,10000,50000      Timeline step labels (default: 0,10000,50000)
  --profile evolution_hunt    Sweep profile (default: evolution_hunt)
  --base-config-root PATH     Config root (default: configs/ecology_plateau80/evolution_hunt)
  --sweep-output-modes        Match --sweep-output-modes from evo_fitness_landscape
  --cap-multiplier N          Population cap multiplier (default: 1.8)
  --cell ARCH X Y FOUNDER SEED  Process a single run only
  --from-state                Read runs from staged_sweep_state.json (default)
"
    );
}

fn parse_arg_value<T: std::str::FromStr>(arg: &str, prefix: &str) -> Option<T> {
    if let Some(v) = arg.strip_prefix(&format!("{prefix}=")) {
        return v.parse().ok();
    }
    None
}

fn parse_stages(raw: &str) -> io::Result<Vec<i32>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let s = part.trim();
        if s.is_empty() {
            continue;
        }
        let v: i32 = s.parse().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("invalid stage: {s}"))
        })?;
        if !out.contains(&v) {
            out.push(v);
        }
    }
    if out.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "at least one stage required",
        ));
    }
    Ok(out)
}

fn config_path(base_root: &Path, name: &str) -> PathBuf {
    base_root.join(format!("{name}.json"))
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

fn parse_sweep_profile(raw: &str) -> Option<SweepProfile> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "full" | "" => Some(SweepProfile::Full),
        "evolution_hunt" | "evolution-hunt" | "hunt" => Some(SweepProfile::EvolutionHunt),
        _ => None,
    }
}

fn load_runs_from_state(output_dir: &Path) -> io::Result<Vec<RunRef>> {
    let path = output_dir.join("staged_sweep_state.json");
    let raw = fs::read_to_string(&path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("read {}: {e}", path.display()),
        )
    })?;
    let state: StagedSweepState = serde_json::from_str(&raw).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, e)
    })?;
    Ok(state
        .all_runs
        .into_iter()
        .map(|r| RunRef {
            architecture: r.architecture,
            x_label: r.x_label,
            y_label: r.y_label,
            founder_mode: r.founder_mode,
            seed: r.seed,
        })
        .collect())
}

fn dedupe_runs(runs: Vec<RunRef>) -> Vec<RunRef> {
    let mut out = Vec::new();
    for run in runs {
        if !out.iter().any(|r: &RunRef| {
            r.architecture == run.architecture
                && r.x_label == run.x_label
                && r.y_label == run.y_label
                && r.founder_mode == run.founder_mode
                && r.seed == run.seed
        }) {
            out.push(run);
        }
    }
    out
}

fn main() -> io::Result<()> {
    let mut output_dir = PathBuf::new();
    let mut base_root = PathBuf::from("configs/ecology_plateau80/evolution_hunt");
    let mut stages = vec![0, 10_000, 50_000];
    let mut sweep_profile = SweepProfile::EvolutionHunt;
    let mut sweep_output_modes = false;
    let mut cap_multiplier = 1.8f64;
    let mut single_cell: Option<RunRef> = None;
    let mut from_state = true;

    for arg in env::args().skip(1) {
        if arg == "--help" || arg == "-h" {
            print_help();
            return Ok(());
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--output-dir") {
            output_dir = PathBuf::from(v);
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--base-config-root") {
            base_root = PathBuf::from(v);
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--stages") {
            stages = parse_stages(&v)?;
            continue;
        }
        if let Some(v) = parse_arg_value::<String>(&arg, "--profile")
            .or_else(|| parse_arg_value::<String>(&arg, "--sweep-profile"))
        {
            sweep_profile = parse_sweep_profile(&v).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown profile: {v}"),
                )
            })?;
            continue;
        }
        if arg == "--sweep-output-modes" {
            sweep_output_modes = true;
            continue;
        }
        if let Some(v) = parse_arg_value::<f64>(&arg, "--cap-multiplier") {
            cap_multiplier = v.max(1.0);
            continue;
        }
        if arg == "--cell" {
            single_cell = None;
            continue;
        }
        if arg.starts_with("--cell=") {
            let parts: Vec<&str> = arg["--cell=".len()..].splitn(5, ',').collect();
            if parts.len() == 5 {
                single_cell = Some(RunRef {
                    architecture: parts[0].to_string(),
                    x_label: parts[1].to_string(),
                    y_label: parts[2].to_string(),
                    founder_mode: parts[3].to_string(),
                    seed: parts[4].parse().unwrap_or(0),
                });
            }
            continue;
        }
        if arg == "--from-state" {
            from_state = true;
            continue;
        }
    }

    if output_dir.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--output-dir is required",
        ));
    }

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
        }
    }

    let mut cells_to_sweep = build_cells_to_sweep(
        &base_perceptron,
        &base_mlp,
        &spatial_bases,
        &base_goo,
        sweep_profile,
    );
    if sweep_output_modes {
        cells_to_sweep = expand_output_mode_cells(cells_to_sweep);
    }

    let runs = if let Some(run) = single_cell {
        vec![run]
    } else if from_state {
        dedupe_runs(load_runs_from_state(&output_dir)?)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no runs to process (use --from-state or --cell=...)",
        ));
    };

    if runs.is_empty() {
        println!("No runs to process.");
        return Ok(());
    }

    let mut written = 0usize;
    let mut skipped = 0usize;
    let mut manifest_entries = Vec::new();

    for run in &runs {
        let Some(cell) = find_sweep_cell(
            &cells_to_sweep,
            &run.architecture,
            &run.x_label,
            &run.y_label,
        ) else {
            eprintln!(
                "[skip] no sweep cell match: {} {} x {} founder={} seed={}",
                run.architecture, run.x_label, run.y_label, run.founder_mode, run.seed
            );
            skipped += 1;
            continue;
        };

        let timeline = match build_timeline(run, cell, &stages, &output_dir, cap_multiplier) {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "[skip] {} {} x {} seed={}: {e}",
                    run.architecture, run.x_label, run.y_label, run.seed
                );
                skipped += 1;
                continue;
            }
        };

        if !timeline_has_data(&timeline) {
            eprintln!(
                "[skip] no timeline data: {} {} x {} seed={}",
                run.architecture, run.x_label, run.y_label, run.seed
            );
            skipped += 1;
            continue;
        }

        let path = write_timeline(&output_dir, &timeline)?;
        written += 1;
        let available_steps: Vec<i32> = timeline
            .timepoints
            .iter()
            .filter(|tp| !tp.missing && !tp.brain_json.is_empty())
            .map(|tp| tp.steps)
            .collect();
        manifest_entries.push(TimelineManifestEntry {
            file: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            architecture: run.architecture.clone(),
            x_label: run.x_label.clone(),
            y_label: run.y_label.clone(),
            founder_mode: run.founder_mode.clone(),
            seed: run.seed,
            available_steps,
        });
    }

    write_manifest(&output_dir, &manifest_entries)?;
    println!(
        "probe timelines: wrote {written} · skipped {skipped} · manifest {} entries",
        manifest_entries.len()
    );
    Ok(())
}
