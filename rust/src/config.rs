use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

/// Configuration structure for the simulator.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    // --- core runtime / defaults ---
    /// Grid width (number of environment cells). Must be >= 1.
    /// Example: 25, 50, 100
    pub grid_width: usize,

    /// RNG seed. Any unsigned 32-bit integer.
    pub seed: u32,

    /// Global multiplier for resources in the environment.
    /// Example: 0.5 (sparser), 1.0 (default), 2.0 (richer)
    pub resource_scale: f64,

    // --- environment dynamics ---

    /// Speed of temporal changes in the environment.
    /// Example: 0.0 (static), 0.001 (slow), 0.01 (fast)
    pub env_time_speed: f64,

    /// Amplitude of time-varying component (scale multiplier).
    pub env_time_amplitude: f64,

    /// Spatial scale for Perlin noise (smaller -> smoother features).
    pub env_perlin_spatial_scale: f64,

    /// Time scale applied to Perlin evaluation.
    pub env_perlin_time_scale: f64,

    // --- perlin / fBm parameters ---
    pub perlin_octaves: usize,
    pub perlin_persistence: f64,
    pub perlin_lacunarity: f64,

    // --- runtime UI / output ---
    pub show_progress: bool,

    // --- population / neural network ---
    /// Number of individuals to place at simulation start.
    pub initial_population: i32,

    /// Maximum allowed population (safety cap).
    pub max_population: i32,

    pub start_pos: usize,
    pub randomize_start_positions: bool,
    pub nn_nodes: i32,
    pub nn_layers: i32,
    pub nn_nodes2: i32,
    pub nn_inputs: i32,
    pub nn_use_occupancy: bool,

    // --- run control ---
    pub steps: i32,
    pub report_interval: i32,
    pub burn_in: i32,
    pub write_frames: bool,

    /// Base directory under which per-run simulation folders (simXXXX) are created.
    pub output_dir_base: String,
    pub sim_prefix: String,
    pub sim_digits: i32,
    pub sim_number: i32,
    pub frames_filename: String,

    pub output_run_dir: String,
    pub output_budget_mb: f64,
    pub output_budget_warn_fraction: f64,
    pub output_budget_hard_stop: bool,

    pub replay_snapshot_path: String,
    pub replay_env_seed: i64,
    pub replay_randomize_start_positions: bool,

    pub step_mode: bool,
    pub snapshot_filename: String,
    pub write_snapshot: bool,

    // --- biology parameters (key knobs) ---
    /// Threshold resource level above which reproduction may occur.
    /// Example: small organisms: 5.0-20.0, large: 50.0+
    pub reproduction_threshold: f64,

    /// Cost (resource deducted) when an individual reproduces.
    pub reproduction_cost: f64,

    /// Per-step probability (or small risk factor) of death.
    pub death_risk: f64,

    /// Maintenance cost per occupied node (resource drain per node).
    pub maintenance_cost_per_node: f64,

    pub mutation_rate: f64,
    pub mutation_sigma: f64,
    pub movement_cost_factor: f64,

    pub radiate: i32,

    // --- visualization ---
    pub vis_cols: usize,
    pub vis_height: usize,
    pub vis_pop_height: usize,
    pub print_frames: bool,

    pub write_config: bool,
    pub config_out_path: String,

    pub write_outputs: bool,
    pub write_individual_deltas: bool,
    pub individuals_delta_out_path: String,
    pub delta_full_interval: i32,
    pub individuals_encoding: String,

    pub env_out_path: String,
    pub individuals_out_path: String,

    pub nn_index_flush_interval: i32,
    pub nn_index_flush_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            grid_width: 100,
            seed: 12345,
            resource_scale: 1.0,
            // perlin time always enabled
            env_time_speed: 0.015,
            env_time_amplitude: 1.0,
            env_perlin_spatial_scale: 0.05,
            env_perlin_time_scale: 0.05,
            perlin_octaves: 5,
            perlin_persistence: 0.5,
            perlin_lacunarity: 2.0,
            show_progress: false,
            initial_population: 200,
            max_population: 100000,
            start_pos: 0,
            randomize_start_positions: true,
            nn_nodes: 10,
            nn_layers: 2,
            nn_nodes2: 10,
            nn_inputs: 5,
            nn_use_occupancy: true,
            steps: 1000,
            report_interval: 1,
            burn_in: 0,
            write_frames: false,
            output_dir_base: "output/example_sim".to_string(),
            sim_prefix: "sim".to_string(),
            sim_digits: 4,
            sim_number: -1,
            frames_filename: "frames.tsv".to_string(),
            output_run_dir: String::new(),
            output_budget_mb: 0.0,
            output_budget_warn_fraction: 0.8,
            output_budget_hard_stop: false,
            replay_snapshot_path: String::new(),
            replay_env_seed: -1,
            replay_randomize_start_positions: false,
            step_mode: false,
            snapshot_filename: "snapshot.cbor".to_string(),
            write_snapshot: false,
            reproduction_threshold: 20.0,
            reproduction_cost: 5.0,
            death_risk: 1e-8,
            maintenance_cost_per_node: 0.0015,
            mutation_rate: 0.01,
            mutation_sigma: 0.1,
            movement_cost_factor: 2.0,
            radiate: 0,
            vis_cols: 120,
            vis_height: 12,
            vis_pop_height: 4,
            print_frames: false,
            write_config: false,
            config_out_path: String::new(),
            write_outputs: true,
            write_individual_deltas: false,
            individuals_delta_out_path: "output/example_sim/individuals.delta.jsonl".to_string(),
            delta_full_interval: 10,
            individuals_encoding: "jsonl".to_string(),
            env_out_path: "output/example_sim/env.tsv".to_string(),
            individuals_out_path: "output/example_sim/individuals.tsv".to_string(),
            nn_index_flush_interval: 0,
            nn_index_flush_size: 0,
        }
    }
}

impl Config {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &str) -> Result<Self, io::Error> {
        let contents = fs::read_to_string(path)?;
        serde_json::from_str(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Try to load configuration from a JSON file, returning None on error.
    pub fn try_load<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path_str = path.as_ref().to_str()?;
        Self::from_file(path_str).ok()
    }

    /// Save configuration to a JSON file.
    pub fn to_file(&self, path: &str) -> Result<(), io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Validate configuration parameters.
    pub fn validate(&self) {
        if self.grid_width == 0 {
            eprintln!("Warning: grid_width is 0, using 1");
        }
        if self.steps < 0 {
            eprintln!("Warning: steps is negative");
        }
        if self.initial_population < 0 {
            eprintln!("Warning: initial_population is negative");
        }
    }
}
