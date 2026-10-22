use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitecturePolicy {
    PerceptronOnly,
    EvolveMlp,
    EvolveSpatialGraph,
    EvolveGoo,
}

impl ArchitecturePolicy {
    pub fn label(self) -> &'static str {
        match self {
            ArchitecturePolicy::PerceptronOnly => "Perceptron",
            ArchitecturePolicy::EvolveMlp => "Multilayer perceptron (MLP)",
            ArchitecturePolicy::EvolveSpatialGraph => "Spatial graph (SG)",
            ArchitecturePolicy::EvolveGoo => "Graph of operations (GoO)",
        }
    }

    pub fn label_short(self) -> &'static str {
        match self {
            ArchitecturePolicy::PerceptronOnly => "Perceptron",
            ArchitecturePolicy::EvolveMlp => "MLP",
            ArchitecturePolicy::EvolveSpatialGraph => "Spatial graph",
            ArchitecturePolicy::EvolveGoo => "GoO",
        }
    }

    pub fn config_key(self) -> &'static str {
        match self {
            ArchitecturePolicy::PerceptronOnly => "perceptron_only",
            ArchitecturePolicy::EvolveMlp => "evolve_mlp",
            ArchitecturePolicy::EvolveSpatialGraph => "evolve_spatial_graph",
            ArchitecturePolicy::EvolveGoo => "evolve_goo",
        }
    }
}

impl Default for ArchitecturePolicy {
    fn default() -> Self {
        ArchitecturePolicy::PerceptronOnly
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GridTopology {
    Ring1d,
    Hex2d,
}

/// Locomotion speed preset: minimum steps between hex moves when the brain signals move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovementSpeed {
    /// One hex per timestep when moving.
    Fast,
    /// One hex per five timesteps when moving.
    Medium,
    /// One hex per twenty-five timesteps when moving.
    Slow,
}

/// Ecology tuning profile applied when `movement_speed` is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EcologyProfile {
    #[default]
    Standard,
    /// Smaller caps, lower node maintenance; tuned for ~80% of `max_population` at equilibrium.
    Plateau80,
}

impl MovementSpeed {
    pub fn movement_interval(self) -> u32 {
        match self {
            MovementSpeed::Fast => 1,
            MovementSpeed::Medium => 5,
            MovementSpeed::Slow => 25,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MovementSpeed::Fast => "fast",
            MovementSpeed::Medium => "medium",
            MovementSpeed::Slow => "slow",
        }
    }
}

impl Default for GridTopology {
    fn default() -> Self {
        GridTopology::Ring1d
    }
}

/// How brain activations map to gather/move magnitude and hex direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovementOutputMode {
    /// Spatial/graph: dedicated readout weights with tanh on both outputs (legacy default).
    SeparateReadout,
    /// Spatial/graph: dedicated readout weights without output tanh (avoids double tanh in step_engine).
    LinearReadout,
    /// Perceptron-style: move = sum(activations/thought), direction = activations[1] or [0].
    HiddenSum,
    /// Spatial: centered vector sum of node activations; uniform nodes → no move.
    HexCompass,
    /// GoO: move = thought[0], direction = thought[1] (not sum of all dims).
    SplitChannels,
}

impl MovementOutputMode {
    pub fn label(self) -> &'static str {
        match self {
            MovementOutputMode::SeparateReadout => "separate_readout",
            MovementOutputMode::LinearReadout => "linear_readout",
            MovementOutputMode::HiddenSum => "hidden_sum",
            MovementOutputMode::HexCompass => "hex_compass",
            MovementOutputMode::SplitChannels => "split_channels",
        }
    }

    pub fn spatial_sweep_modes() -> &'static [MovementOutputMode] {
        &[
            MovementOutputMode::SeparateReadout,
            MovementOutputMode::LinearReadout,
            MovementOutputMode::HiddenSum,
            MovementOutputMode::HexCompass,
        ]
    }

    pub fn goo_sweep_modes() -> &'static [MovementOutputMode] {
        &[
            MovementOutputMode::HiddenSum,
            MovementOutputMode::SplitChannels,
        ]
    }
}

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

    /// Stretch Perlin samples away from 0.5 before scaling (1 = off, >1 = sharper patches).
    #[serde(default = "default_resource_contrast")]
    pub resource_contrast: f64,

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

    // --- evolvable brain architecture ---
    pub architecture_policy: ArchitecturePolicy,
    pub architecture_mutation_rate: f64,
    /// When true, growth mutations copy a parent node/layer/edge weights plus noise.
    #[serde(default)]
    pub structural_duplication: bool,
    pub max_brain_nodes: i32,
    /// Spatial architecture: `hex_edge` (default blob), `cardinal_4`, or `legacy_graph` (landmark graph).
    #[serde(default = "default_spatial_blob_layout")]
    pub spatial_blob_layout: String,
    /// When true, founders keep zero weights until mutation (radiate still applies bursts).
    pub neutral_founder_weights: bool,
    pub grid_topology: GridTopology,
    pub grid_height: usize,
    /// Axial radius for hexagonal dish (0 = rectangular/toroidal hex grid).
    pub hex_dish_radius: i32,
    /// When true with `hex_dish_radius > 0`, edge hex moves wrap through the dish bounding box.
    #[serde(default)]
    pub hex_dish_torus: bool,
    pub replicates: u32,
    pub export_cladogram: bool,
    pub use_gpu_step: bool,

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

    /// When true, ecology presets do not overwrite `maintenance_cost_per_node`.
    #[serde(default)]
    pub lock_maintenance_cost: bool,

    /// Flat per-step basal metabolic cost (independent of brain size).
    pub basal_metabolic_cost: f64,

    pub mutation_rate: f64,
    pub mutation_sigma: f64,
    pub movement_cost_factor: f64,

    /// Minimum steps between hex moves when `move_signal > 0`. Overridden when
    /// `movement_speed` is set in JSON. Default 1 (one hex per step).
    pub movement_interval: u32,

    /// Optional preset that sets `movement_interval` and ecology knobs via `ecology_presets`.
    #[serde(default)]
    pub movement_speed: Option<MovementSpeed>,

    /// Which ecology preset to apply with `movement_speed` (default: standard batch tuning).
    #[serde(default)]
    pub ecology_profile: EcologyProfile,

    /// Override how brain outputs link to movement magnitude and direction.
    /// When omitted, architecture-specific defaults apply (spatial: separate_readout, GoO: hidden_sum).
    #[serde(default)]
    pub movement_output_mode: Option<MovementOutputMode>,

    pub radiate: i32,

    // --- visualization ---
    pub vis_cols: usize,
    pub vis_height: usize,
    pub vis_pop_height: usize,
    pub print_frames: bool,

    pub write_config: bool,
    pub config_out_path: String,

    pub write_outputs: bool,
    pub debug_buffered_output_logs: bool,
    pub write_individual_deltas: bool,
    pub individuals_delta_out_path: String,
    pub delta_full_interval: i32,
    pub individuals_encoding: String,

    pub env_out_path: String,
    pub individuals_out_path: String,

    pub mutate_rates: bool,

    // --- developmental plasticity & motility morphology ---
    /// Duration of juvenile development in timesteps before morph commitment (0 = mature at birth).
    #[serde(default = "default_developmental_steps")]
    pub developmental_steps: u32,

    /// Extra metabolic upkeep per step for possessing flagella.
    #[serde(default = "default_flagella_maintenance_cost")]
    pub flagella_maintenance_cost: f64,

    /// Default developmental cue threshold for morph differentiation:
    /// if average juvenile per-capita cue >= morph_threshold -> develops flagella (motile).
    #[serde(default = "default_morph_threshold")]
    pub default_morph_threshold: f64,

    /// When true, individuals feed at their location (or arrival location when moving) every step.
    /// When false, moving forfeits feeding for that step (legacy behavior).
    #[serde(default = "default_allow_concurrent_eat_and_move")]
    pub allow_concurrent_eat_and_move: bool,

    pub nn_index_flush_interval: i32,
    pub nn_index_flush_size: usize,
}

fn default_developmental_steps() -> u32 {
    10
}

fn default_flagella_maintenance_cost() -> f64 {
    0.02
}

fn default_morph_threshold() -> f64 {
    0.5
}

fn default_allow_concurrent_eat_and_move() -> bool {
    true
}

fn default_resource_contrast() -> f64 {
    1.0
}

fn default_spatial_blob_layout() -> String {
    "hex_edge".to_string()
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
            resource_contrast: 1.0,
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
            nn_inputs: 8,
            nn_use_occupancy: true,
            architecture_policy: ArchitecturePolicy::PerceptronOnly,
            architecture_mutation_rate: 0.005,
            structural_duplication: false,
            max_brain_nodes: 32,
            spatial_blob_layout: default_spatial_blob_layout(),
            neutral_founder_weights: true,
            grid_topology: GridTopology::Ring1d,
            grid_height: 0,
            hex_dish_radius: 0,
            hex_dish_torus: false,
            replicates: 1,
            export_cladogram: false,
            use_gpu_step: false,
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
            maintenance_cost_per_node: 0.025,
            lock_maintenance_cost: false,
            basal_metabolic_cost: 0.05,
            mutation_rate: 0.01,
            mutation_sigma: 0.1,
            movement_cost_factor: 2.0,
            movement_interval: 1,
            movement_speed: None,
            ecology_profile: EcologyProfile::default(),
            movement_output_mode: None,
            radiate: 0,
            vis_cols: 120,
            vis_height: 12,
            vis_pop_height: 4,
            print_frames: false,
            write_config: false,
            config_out_path: String::new(),
            write_outputs: true,
            debug_buffered_output_logs: false,
            write_individual_deltas: false,
            individuals_delta_out_path: "output/example_sim/individuals.delta.jsonl".to_string(),
            delta_full_interval: 10,
            individuals_encoding: "jsonl".to_string(),
            env_out_path: "output/example_sim/env.tsv".to_string(),
            individuals_out_path: "output/example_sim/individuals.tsv".to_string(),
            mutate_rates: false,
            developmental_steps: default_developmental_steps(),
            flagella_maintenance_cost: default_flagella_maintenance_cost(),
            default_morph_threshold: default_morph_threshold(),
            allow_concurrent_eat_and_move: default_allow_concurrent_eat_and_move(),
            nn_index_flush_interval: 0,
            nn_index_flush_size: 0,
        }
    }
}

impl Config {
    /// Load configuration from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let contents = fs::read_to_string(path.as_ref())?;
        let mut cfg: Config = serde_json::from_str(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        cfg.finalize();
        Ok(cfg)
    }

    /// Try to load configuration from a JSON file, returning None on error.
    pub fn try_load<P: AsRef<Path>>(path: P) -> Option<Self> {
        Self::from_file(path).ok()
    }

    /// Save configuration to a JSON file.
    pub fn to_file(&self, path: &str) -> Result<(), io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Resolve `movement_speed` preset and clamp movement interval.
    pub fn finalize(&mut self) {
        if let Some(speed) = self.movement_speed {
            match self.ecology_profile {
                EcologyProfile::Plateau80 => {
                    crate::ecology_presets::apply_plateau80_movement_speed(self, speed);
                }
                EcologyProfile::Standard => {
                    crate::ecology_presets::apply_movement_speed(self, speed);
                }
            }
        } else {
            crate::ecology_presets::apply_architecture_ecology_offset(self);
        }
        if self.movement_interval == 0 {
            self.movement_interval = 1;
        }
        if self.use_gpu_step && !self.gpu_supports_movement_output_mode() {
            eprintln!(
                "Warning: disabling use_gpu_step — GPU forward only supports default movement_output_mode for this architecture"
            );
            self.use_gpu_step = false;
        }
        if matches!(
            self.architecture_policy,
            ArchitecturePolicy::PerceptronOnly
        ) {
            self.nn_inputs = crate::brain::PERCEPTRON_SENSOR_DIM as i32;
        }
    }

    pub fn resolved_movement_output_mode(&self) -> MovementOutputMode {
        self.movement_output_mode.unwrap_or_else(|| {
            crate::brain::default_movement_output_mode(
                self.architecture_policy,
                self.uses_legacy_spatial_graph(),
            )
        })
    }

    fn gpu_supports_movement_output_mode(&self) -> bool {
        let mode = self.resolved_movement_output_mode();
        match self.architecture_policy {
            ArchitecturePolicy::EvolveGoo => mode == MovementOutputMode::HiddenSum,
            ArchitecturePolicy::EvolveSpatialGraph if self.uses_legacy_spatial_graph() => {
                mode == MovementOutputMode::SeparateReadout
            }
            ArchitecturePolicy::EvolveSpatialGraph => false,
            _ => true,
        }
    }

    pub fn uses_legacy_spatial_graph(&self) -> bool {
        self.spatial_blob_layout.eq_ignore_ascii_case("legacy_graph")
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
        if self.max_population > 0 && self.initial_population > self.max_population {
            eprintln!(
                "Warning: initial_population ({}) exceeds max_population ({}); founders will be clamped",
                self.initial_population, self.max_population
            );
        }
    }
}
