//! Central registry for population snapshots (`output/snapshots/{id}.cbor` + `{id}.json`).

use crate::build_info;
use crate::config::{ArchitecturePolicy, Config};
use crate::run_record::RunRecordContext;
use crate::snapshot::PopulationSnapshot;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const REGISTRY_DIR: &str = "output/snapshots";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub id: String,
    pub label: String,
    /// Filename within the registry directory, e.g. `{id}.cbor`.
    pub snapshot_file: String,
    pub registered_at_unix_secs: u64,
    pub source: String,
    pub source_run_dir: String,
    pub source_snapshot_path: String,
    pub config_file: Option<String>,
    pub architecture_policy: String,
    pub steps_completed: i32,
    pub final_population: usize,
    pub hex_dish_radius: u32,
    pub seed: u32,
    pub architecture_mutation_rate: f64,
    pub max_brain_nodes: i32,
    pub version: String,
    pub build_id: String,
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn policy_slug(policy: ArchitecturePolicy) -> &'static str {
    match policy {
        ArchitecturePolicy::PerceptronOnly => "perceptron",
        ArchitecturePolicy::EvolveMlp => "mlp",
        ArchitecturePolicy::EvolveSpatialGraph => "spatial",
        ArchitecturePolicy::EvolveGoo => "goo",
    }
}

pub fn new_snapshot_id(
    policy: ArchitecturePolicy,
    seed: u32,
    timestep: i32,
    finished_unix: u64,
) -> String {
    format!(
        "{}_{}_t{}_{:x}",
        policy_slug(policy),
        seed,
        timestep,
        finished_unix
    )
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

pub fn registry_path(repo_root: &Path) -> PathBuf {
    repo_root.join(REGISTRY_DIR)
}

pub fn resolve_registry_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    registry_path(&find_repo_root(cwd))
}

/// Copy a written snapshot into the central registry and append to `index.jsonl`.
pub fn register_headless_snapshot(
    snapshot: &PopulationSnapshot,
    source_path: &Path,
    simdir: &str,
    cfg: &Config,
    ctx: &RunRecordContext,
    final_population: usize,
    finished_unix: Option<u64>,
) -> Option<SnapshotManifest> {
    let repo_root = find_repo_root(
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    );
    let reg_dir = registry_path(&repo_root);
    if let Err(e) = fs::create_dir_all(&reg_dir) {
        eprintln!(
            "[warn] snapshot registry: failed to create {}: {}",
            reg_dir.display(),
            e
        );
        return None;
    }

    let finished = finished_unix.unwrap_or_else(unix_now_secs);
    let id = new_snapshot_id(cfg.architecture_policy, cfg.seed, snapshot.timestep, finished);
    let cbor_name = format!("{id}.cbor");
    let json_name = format!("{id}.json");
    let dest_cbor = reg_dir.join(&cbor_name);

    if let Err(e) = fs::copy(source_path, &dest_cbor) {
        eprintln!(
            "[warn] snapshot registry: failed to copy {} -> {}: {}",
            source_path.display(),
            dest_cbor.display(),
            e
        );
        return None;
    }

    let label = format!(
        "{} · t={} · n={} · seed={}",
        cfg.architecture_policy.label(),
        snapshot.timestep,
        final_population,
        cfg.seed
    );

    let manifest = SnapshotManifest {
        id: id.clone(),
        label,
        snapshot_file: cbor_name,
        registered_at_unix_secs: finished,
        source: "headless".to_string(),
        source_run_dir: simdir.to_string(),
        source_snapshot_path: source_path.to_string_lossy().into_owned(),
        config_file: ctx.config_file.clone(),
        architecture_policy: cfg.architecture_policy.label().to_string(),
        steps_completed: snapshot.timestep,
        final_population,
        hex_dish_radius: snapshot.hex_dish_radius,
        seed: cfg.seed,
        architecture_mutation_rate: cfg.architecture_mutation_rate,
        max_brain_nodes: cfg.max_brain_nodes,
        version: build_info::CRATE_VERSION.to_string(),
        build_id: build_info::BUILD_ID.to_string(),
    };

    let manifest_path = reg_dir.join(&json_name);
    match serde_json::to_string_pretty(&manifest) {
        Ok(json) => {
            if let Err(e) = fs::write(&manifest_path, json) {
                eprintln!(
                    "[warn] snapshot registry: failed to write {}: {}",
                    manifest_path.display(),
                    e
                );
            }
        }
        Err(e) => eprintln!("[warn] snapshot registry: failed to serialize manifest: {e}"),
    }

    let index_path = reg_dir.join("index.jsonl");
    if let Ok(line) = serde_json::to_string(&manifest) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index_path)
        {
            let _ = writeln!(file, "{line}");
        }
    }

    println!(
        "[snapshot-registry] registered {id} -> {}",
        dest_cbor.display()
    );
    Some(manifest)
}

/// All `*.cbor` files directly under the registry directory (flat layout).
pub fn list_registry_snapshot_paths(registry_dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = fs::read_dir(registry_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if p.extension().and_then(|e| e.to_str()) == Some("cbor") {
                    paths.push(p);
                }
            }
        }
    }
    paths.sort();
    paths
}

fn manifest_path_for_cbor(registry_dir: &Path, cbor_path: &Path) -> PathBuf {
    let stem = cbor_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("snapshot");
    registry_dir.join(format!("{stem}.json"))
}

pub fn manifest_label(registry_dir: &Path, cbor_path: &Path) -> Option<String> {
    let manifest_path = manifest_path_for_cbor(registry_dir, cbor_path);
    let text = fs::read_to_string(manifest_path).ok()?;
    let manifest: SnapshotManifest = serde_json::from_str(&text).ok()?;
    Some(manifest.label)
}

/// `(cbor path, optional label)` for every snapshot in the flat registry.
pub fn list_registry_entries(registry_dir: &Path) -> Vec<(PathBuf, Option<String>)> {
    list_registry_snapshot_paths(registry_dir)
        .into_iter()
        .map(|p| {
            let label = manifest_label(registry_dir, &p);
            (p, label)
        })
        .collect()
}

pub fn is_source_registered(registry_dir: &Path, source_path: &Path) -> bool {
    let index = registry_dir.join("index.jsonl");
    let Ok(content) = fs::read_to_string(index) else {
        return false;
    };
    let source = source_path.to_string_lossy();
    let canonical = source_path.canonicalize().ok();
    for line in content.lines() {
        let Ok(manifest) = serde_json::from_str::<SnapshotManifest>(line) else {
            continue;
        };
        if manifest.source_snapshot_path == source {
            return true;
        }
        if let Some(ref canon) = canonical {
            if PathBuf::from(&manifest.source_snapshot_path) == *canon {
                return true;
            }
        }
    }
    false
}

/// Import a snapshot written by an older run layout into the flat registry.
pub fn import_existing_snapshot(
    source_cbor: &Path,
    simdir: &str,
    cfg: &Config,
    ctx: &RunRecordContext,
    final_population: usize,
    finished_unix: Option<u64>,
) -> Option<SnapshotManifest> {
    let reg_dir = resolve_registry_dir();
    if is_source_registered(&reg_dir, source_cbor) {
        println!(
            "[snapshot-registry] already registered: {}",
            source_cbor.display()
        );
        return None;
    }
    let snapshot = match PopulationSnapshot::read_from_path(source_cbor) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[snapshot-registry] failed to read {}: {}",
                source_cbor.display(),
                e
            );
            return None;
        }
    };
    register_headless_snapshot(
        &snapshot,
        source_cbor,
        simdir,
        cfg,
        ctx,
        final_population,
        finished_unix,
    )
}

/// Prefer flat `output/snapshots/*.cbor`; fall back to nested `*/snapshot.cbor`.
pub fn collect_snapshots_from_dir(dir: &Path) -> Vec<PathBuf> {
    let flat = list_registry_snapshot_paths(dir);
    if !flat.is_empty() {
        return flat;
    }
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let snap = p.join("snapshot.cbor");
                if snap.is_file() {
                    paths.push(snap);
                }
            }
        }
    }
    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_id_is_stable_format() {
        let id = new_snapshot_id(ArchitecturePolicy::EvolveGoo, 146, 199_999, 1_700_000_000);
        assert!(id.starts_with("goo_146_t199999_"));
    }
}
