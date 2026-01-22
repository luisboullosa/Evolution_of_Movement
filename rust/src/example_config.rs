//! Annotated example configuration for the simulator.
//!
//! This file is a commented copy of `config.rs` intended as
//! documentation and an in-repo example. Use it as a template
//! for creating JSON config files passed to the binary via
//! `--config path/to/config.json`.
//!
//! Example usage (create a JSON file from the sample below):
//! ```json
//! {
//!   "grid_width": 100,
//!   "seed": 12345,
//!   "resource_scale": 1.0,
//!   (Perlin temporal evolution is always enabled),
//!   "env_time_speed": 0.015,
//!   "env_time_amplitude": 1.0,
//!   "env_perlin_spatial_scale": 0.05,
//!   "env_perlin_time_scale": 0.05,
//!   (additive `env_change_rate` removed; Perlin time is used)
//!   "perlin_octaves": 5,
//!   "perlin_persistence": 0.5,
//!   "perlin_lacunarity": 2.0,
//!   "show_progress": false,
//!   "initial_population": 500,
//!   "steps": 2000,
//!   "write_frames": false,
//!   "output_dir_base": "output/my_scan",
//!   "sim_prefix": "sim",
//!   "reproduction_threshold": 20.0,
//!   "reproduction_cost": 5.0,
//!   "death_risk": 1e-8,
//!   "maintenance_cost_per_node": 0.0015,
//!   "movement_cost_factor": 2.0
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Example config structure — fields and comments mirror `Config`.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct ExampleConfig {
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

    // --- visualization ---
    pub vis_cols: usize,
    pub vis_height: usize,
    pub vis_pop_height: usize,
    pub print_frames: bool,

    pub write_config: bool,
    pub config_out_path: String,

    pub env_out_path: String,
    pub individuals_out_path: String,

    pub nn_index_flush_interval: i32,
    pub nn_index_flush_size: usize,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        ExampleConfig {
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
            reproduction_threshold: 20.0,
            reproduction_cost: 5.0,
            death_risk: 1e-8,
            maintenance_cost_per_node: 0.0015,
            mutation_rate: 0.01,
            mutation_sigma: 0.1,
            movement_cost_factor: 2.0,
            vis_cols: 120,
            vis_height: 12,
            vis_pop_height: 4,
            print_frames: false,
            write_config: false,
            config_out_path: String::new(),
            env_out_path: "output/example_sim/env.tsv".to_string(),
            individuals_out_path: "output/example_sim/individuals.tsv".to_string(),
            nn_index_flush_interval: 0,
            nn_index_flush_size: 0,
        }
    }
}
