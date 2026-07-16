//! Persist headless run timing and parameters for future runtime estimates.

use crate::build_info;
use crate::config::{ArchitecturePolicy, GridTopology};
use crate::hex_simulation::grid_cell_count;
use crate::simulation::Simulation;
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Optional metadata supplied by the headless CLI (config path, etc.).
#[derive(Debug, Clone, Default)]
pub struct RunRecordContext {
    pub config_file: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BinaryInfo {
    pub name: String,
    pub version: String,
    pub build_id: String,
    pub version_label: String,
    pub cuda_arch: String,
}

#[derive(Debug, Serialize)]
pub struct WorkerInfo {
    pub binary: BinaryInfo,
    pub host_os: String,
    pub cpu_threads: usize,
    pub gpu_step_configured: bool,
    pub gpu_population_compatible: bool,
    pub gpu_steps_executed: u64,
    pub cpu_brain_steps_executed: u64,
}

#[derive(Debug, Serialize)]
pub struct SimParams {
    pub config_file: Option<String>,
    pub output_run_dir: String,
    pub architecture_policy: String,
    pub grid_topology: String,
    pub hex_dish_radius: i32,
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_cells: usize,
    pub steps_requested: i32,
    pub steps_completed: i32,
    pub report_interval: i32,
    pub initial_population: i32,
    pub max_population: i32,
    pub final_population: usize,
    pub seed: u32,
    pub use_gpu_step: bool,
    pub max_brain_nodes: i32,
    pub nn_nodes: i32,
    pub nn_layers: i32,
    pub nn_inputs: i32,
    pub architecture_mutation_rate: f64,
    pub neutral_founder_weights: bool,
    pub write_snapshot: bool,
    pub write_outputs: bool,
    pub show_progress: bool,
}

#[derive(Debug, Serialize)]
pub struct TimingInfo {
    pub duration_secs: f64,
    pub ms_per_step: f64,
    pub steps_per_sec: f64,
    pub finished_at_unix_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct RunRecord {
    pub timing: TimingInfo,
    pub worker: WorkerInfo,
    pub sim: SimParams,
    pub prop_moved_final: f64,
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn current_binary_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn policy_label(policy: ArchitecturePolicy) -> String {
    policy.label().to_string()
}

fn topology_label(topology: GridTopology) -> String {
    match topology {
        GridTopology::Ring1d => "ring_1d".to_string(),
        GridTopology::Hex2d => "hex_2d".to_string(),
    }
}

pub fn build_worker_info(sim: &Simulation) -> WorkerInfo {
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    WorkerInfo {
        binary: BinaryInfo {
            name: current_binary_name(),
            version: build_info::CRATE_VERSION.to_string(),
            build_id: build_info::BUILD_ID.to_string(),
            version_label: build_info::VERSION_LABEL.to_string(),
            cuda_arch: build_info::CUDA_ARCH.to_string(),
        },
        host_os: std::env::consts::OS.to_string(),
        cpu_threads,
        gpu_step_configured: sim.config.use_gpu_step,
        gpu_population_compatible: sim.tensor.population_is_gpu_compatible(),
        gpu_steps_executed: sim.gpu_steps_executed,
        cpu_brain_steps_executed: sim.cpu_brain_steps_executed,
    }
}

pub fn build_sim_params(
    sim: &Simulation,
    ctx: &RunRecordContext,
    steps_requested: i32,
    steps_completed: i32,
    final_population: usize,
) -> SimParams {
    let cfg = &sim.config;
    SimParams {
        config_file: ctx.config_file.clone(),
        output_run_dir: cfg.output_run_dir.clone(),
        architecture_policy: policy_label(cfg.architecture_policy),
        grid_topology: topology_label(cfg.grid_topology),
        hex_dish_radius: cfg.hex_dish_radius,
        grid_width: cfg.grid_width,
        grid_height: cfg.grid_height,
        grid_cells: grid_cell_count(cfg),
        steps_requested,
        steps_completed,
        report_interval: cfg.report_interval,
        initial_population: cfg.initial_population,
        max_population: cfg.max_population,
        final_population,
        seed: cfg.seed,
        use_gpu_step: cfg.use_gpu_step,
        max_brain_nodes: cfg.max_brain_nodes,
        nn_nodes: cfg.nn_nodes,
        nn_layers: cfg.nn_layers,
        nn_inputs: cfg.nn_inputs,
        architecture_mutation_rate: cfg.architecture_mutation_rate,
        neutral_founder_weights: cfg.neutral_founder_weights,
        write_snapshot: cfg.write_snapshot,
        write_outputs: cfg.write_outputs,
        show_progress: cfg.show_progress,
    }
}

pub fn write_run_record(simdir: &str, record: &RunRecord) {
    if simdir.is_empty() {
        return;
    }
    let path = Path::new(simdir).join("run_record.json");
    if let Ok(json) = serde_json::to_string_pretty(record) {
        let _ = std::fs::write(&path, json);
    }
    append_global_history(simdir, record);
}

fn repo_output_dir(simdir: &str) -> Option<PathBuf> {
    let simpath = Path::new(simdir);
    for ancestor in simpath.ancestors() {
        if ancestor.file_name().and_then(|n| n.to_str()) == Some("output") {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn append_global_history(simdir: &str, record: &RunRecord) {
    let Some(output_dir) = repo_output_dir(simdir) else {
        return;
    };
    let _ = std::fs::create_dir_all(&output_dir);
    let history_path = output_dir.join("run_history.jsonl");
    if let Ok(line) = serde_json::to_string(record) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&history_path)
        {
            let _ = writeln!(file, "{line}");
        }
    }
}

pub fn persist_run_record(
    sim: &Simulation,
    ctx: &RunRecordContext,
    simdir: &str,
    steps_requested: i32,
    steps_completed: i32,
    duration_secs: f64,
    final_population: usize,
    prop_moved_final: f64,
) {
    let steps_done = steps_completed.max(1) as f64;
    let record = RunRecord {
        timing: TimingInfo {
            duration_secs,
            ms_per_step: duration_secs * 1000.0 / steps_done,
            steps_per_sec: steps_done / duration_secs.max(1e-9),
            finished_at_unix_secs: unix_now_secs(),
        },
        worker: build_worker_info(sim),
        sim: build_sim_params(sim, ctx, steps_requested, steps_completed, final_population),
        prop_moved_final,
    };
    write_run_record(simdir, &record);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn run_record_serializes_key_fields() {
        let cfg = Config::default();
        let sim = Simulation::new(&cfg);
        let ctx = RunRecordContext {
            config_file: Some("configs/test.json".to_string()),
        };
        let record = RunRecord {
            timing: TimingInfo {
                duration_secs: 120.0,
                ms_per_step: 12.0,
                steps_per_sec: 83.33,
                finished_at_unix_secs: 1_700_000_000,
            },
            worker: build_worker_info(&sim),
            sim: build_sim_params(&sim, &ctx, 10_000, 10_000, 500),
            prop_moved_final: 0.01,
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("\"architecture_policy\""));
        assert!(json.contains("\"ms_per_step\""));
        assert!(json.contains("\"cuda_arch\""));
    }
}
