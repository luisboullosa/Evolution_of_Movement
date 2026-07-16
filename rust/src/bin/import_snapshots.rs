//! Import legacy `snapshot.cbor` files into `output/snapshots/` with JSON manifests.

use evo_movement_rust::config::Config;
use evo_movement_rust::run_record::RunRecordContext;
use evo_movement_rust::snapshot_registry;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct RunRecordImport {
    timing: TimingImport,
    sim: SimImport,
}

#[derive(Debug, Deserialize)]
struct TimingImport {
    finished_at_unix_secs: u64,
}

#[derive(Debug, Deserialize)]
struct SimImport {
    config_file: Option<String>,
    output_run_dir: String,
    final_population: usize,
}

fn find_repo_root(mut cur: PathBuf) -> PathBuf {
    loop {
        if cur.join("CMakeLists.txt").exists() || cur.join(".git").exists() {
            return cur;
        }
        if let Some(parent) = cur.parent() {
            cur = parent.to_path_buf();
        } else {
            return cur;
        }
    }
}

fn resolve_config_path(repo_root: &Path, config_file: &str) -> PathBuf {
    let p = PathBuf::from(config_file);
    if p.is_absolute() {
        return p;
    }
    let from_rust = repo_root.join("rust").join(config_file);
    if from_rust.is_file() {
        return from_rust;
    }
    repo_root.join(config_file)
}

fn read_text_file(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    if bytes.starts_with(&[0xFF, 0xFE]) {
        let u16: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        return Ok(String::from_utf16_lossy(&u16));
    }
    let text = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        String::from_utf8_lossy(&bytes[3..]).into_owned()
    } else {
        String::from_utf8_lossy(&bytes).into_owned()
    };
    Ok(text)
}

fn load_run_record(run_dir: &Path) -> Option<RunRecordImport> {
    let path = run_dir.join("run_record.json");
    let text = match read_text_file(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[skip] cannot read {}: {}", path.display(), e);
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("[skip] cannot parse {}: {}", path.display(), e);
            None
        }
    }
}

fn import_one(repo_root: &Path, snapshot_cbor: &Path) -> bool {
    let snapshot_cbor = snapshot_cbor
        .canonicalize()
        .unwrap_or_else(|_| snapshot_cbor.to_path_buf());
    let Some(run_dir) = snapshot_cbor.parent() else {
        return false;
    };
    let record = match load_run_record(run_dir) {
        Some(r) => r,
        None => {
            eprintln!(
                "[skip] no run_record.json in {}",
                run_dir.display()
            );
            return false;
        }
    };

    let config_path = record
        .sim
        .config_file
        .as_deref()
        .map(|cf| resolve_config_path(repo_root, cf));
    let cfg = match config_path.and_then(|p| Config::from_file(p.to_str()?).ok()) {
        Some(c) => c,
        None => {
            eprintln!(
                "[skip] could not load config for {}",
                snapshot_cbor.display()
            );
            return false;
        }
    };

    let ctx = RunRecordContext {
        config_file: record.sim.config_file.clone(),
    };
    let simdir = if record.sim.output_run_dir.is_empty() {
        run_dir.to_string_lossy().into_owned()
    } else {
        record.sim.output_run_dir.clone()
    };

    snapshot_registry::import_existing_snapshot(
        &snapshot_cbor,
        &simdir,
        &cfg,
        &ctx,
        record.sim.final_population,
        Some(record.timing.finished_at_unix_secs),
    )
    .is_some()
}

fn collect_legacy_snapshots(output_dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let Ok(entries) = std::fs::read_dir(output_dir) else {
        return paths;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let direct = p.join("snapshot.cbor");
            if direct.is_file() {
                paths.push(direct);
                continue;
            }
            if let Ok(sub) = std::fs::read_dir(&p) {
                for sub_entry in sub.flatten() {
                    let sub_p = sub_entry.path();
                    if sub_p.is_dir() {
                        let nested = sub_p.join("snapshot.cbor");
                        if nested.is_file() {
                            paths.push(nested);
                        }
                    }
                }
            }
        }
    }
    paths.sort();
    paths
}

fn main() {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let repo_root = find_repo_root(cwd);
    let output_dir = repo_root.join("output");

    let mut targets: Vec<PathBuf> = std::env::args()
        .skip(1)
        .map(PathBuf::from)
        .collect();
    if targets.is_empty() {
        targets = collect_legacy_snapshots(&output_dir);
    }

    if targets.is_empty() {
        eprintln!("No snapshot.cbor files found under {}", output_dir.display());
        std::process::exit(1);
    }

    let total = targets.len();
    let mut imported = 0usize;
    for path in targets {
        if import_one(&repo_root, &path) {
            imported += 1;
        }
    }
    println!(
        "[snapshot-registry] imported {imported} of {total} candidate(s) -> {}",
        snapshot_registry::resolve_registry_dir().display()
    );
}
