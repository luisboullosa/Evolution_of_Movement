#pragma once
#include <string>

struct Config {
    // -----------------------------
    // Core runtime / default params
    // -----------------------------
    // Spatial grid width (number of cells)
    size_t grid_width = 100;
    // RNG seed for reproducible runs
    unsigned seed = 12345;

    // Deprecated / currently unused: global multiplier applied to Perlin output.
    // Left here for backwards compatibility; can be removed if you don't need it.
    double resource_scale = 1.0;

    // -----------------------------
    // Environment dynamics
    // -----------------------------
    // Two alternative environmental models are supported:
    // 1) Perlin time evolution (used by default when env_use_perlin_time == true)
    //    - Smooth, spatially and temporally correlated changes driven by Perlin noise.
    // 2) Gaussian additive noise (fallback)
    //    - Simple independent per-cell Gaussian perturbations controlled by env_change_rate.

    // Use Perlin time-evolution (default = true). If false, the simulator will use
    // additive Gaussian noise controlled by env_change_rate.
    bool env_use_perlin_time = true;

    // Perlin time-evolution parameters (used when env_use_perlin_time == true)
    // advance speed through the Perlin time axis (units: time coordinate per timestep)
    double env_time_speed = 0.015;
    // amplitude of the time-dimension contribution (multiplies the time-slice effect)
    double env_time_amplitude = 1.0;
    // spatial and temporal scaling applied when sampling 2D Perlin noise
    double env_perlin_spatial_scale = 0.05;
    double env_perlin_time_scale = 0.05;

    // Additive Gaussian fallback (used when env_use_perlin_time == false)
    // Standard deviation of per-cell additive noise per timestep (>= 0)
    double env_change_rate = 0.1;

    // -----------------------------
    // Console / runtime UI
    // -----------------------------
    // Display a small textual progress bar and enable 'q'+Enter to stop early
    bool show_progress = true;

    // -----------------------------
    // Population / neural network
    // -----------------------------
    int initial_population = 20000;
    // Optional cap on population size. When >0, reproduction will be skipped once
    // the population reaches this size. Default 0 means no cap.
    int max_population = 100000;
    // When randomize_start_positions == true, the simulation samples random start positions
    // using the RNG seeded by `seed`. If false, all individuals start at `start_pos`.
    size_t start_pos = 0;
    bool randomize_start_positions = true;
    int nn_nodes = 10;
    // number of hidden layers in the neural network (1 = single hidden layer, 2 = two-layer network)
    int nn_layers = 2;
    // when nn_layers == 2, use nn_nodes2 if >0, otherwise use nn_nodes for second layer size
    int nn_nodes2 = 10;
    // number of numeric inputs provided to the neural network. Default 1 for
    // backward compatibility. When set to 4 the inputs are expected to be
    // {left, center, right, occupancy}.
    int nn_inputs = 5;

    // When true, the last input (occupancy / local population) is provided
    // to the network. This can be turned off to disable the occupancy feature
    // while still using other inputs.
    bool nn_use_occupancy = true;

    // -----------------------------
    // Simulation run control & output
    // -----------------------------
    int steps = 100500;
    int report_interval = 1;
    // initial burn-in period (timesteps). No frames/snapshots are written before this time.
    // Default set to 0 to report from the first step.
    int burn_in = 100000;
    bool write_frames = true;

    // webviewer output is written into a per-simulation directory under `webviewer/`.
    // Example: webviewer/sim0001/frames.tsv, env.tsv.<t>.tsv, individuals.tsv.<t>.tsv, nn_index.json
    std::string webviewer_base = "webviewer";
    std::string sim_prefix = "sim"; // prefix for simulation folders
    int sim_digits = 4;             // zero-pad width for simulation number
    int sim_number = -1;            // if >=0, use this simulation number; otherwise auto-increment
    std::string frames_filename = "frames.tsv";

    // -----------------------------
    // Biology parameters
    // -----------------------------
    double reproduction_threshold = 20.0;
    double reproduction_cost = 5.0;
    double death_risk = 0.00000001; //per timestep lived
    double maintenance_cost_per_node = 0.0015; //0.01
    double mutation_rate = 0.01;
    double mutation_sigma = 0.1; //0.01
    double movement_cost_factor = 2.0; //1.0

    // -----------------------------
    // Visualization
    // -----------------------------
    size_t vis_cols = 120;
    size_t vis_height = 12;
    size_t vis_pop_height = 4;
    // If true, print the ASCII frame visualization to the console
    bool print_frames = false;

    // When true, write a single resolved config header line into the frames file and
    // optionally write a separate config_out_path file inside the sim directory.
    bool write_config = false;
    std::string config_out_path = ""; // when set, a copy of the resolved config is written here in the sim dir

    // output logs for visualization
    std::string env_out_path = "webviewer/sim9999/env.tsv";
    std::string individuals_out_path = "webviewer/sim9999/individuals.tsv";

    // -----------------------------
    // nn_index flush controls
    // -----------------------------
    // If >0, flush the in-memory nn_index cache to disk every this many reports.
    // A value of 0 disables periodic flushing (only flushed at the end of run()).
    int nn_index_flush_interval = 0;
    // If >0, flush the in-memory nn_index cache when the number of cached
    // unique NN entries reaches this threshold.
    size_t nn_index_flush_size = 0;

    // -----------------------------
    // Runtime config utilities
    // -----------------------------
    // Load overrides from a JSON file (simple parser that recognizes the keys used by Config)
    // Returns true if the file was found and parsed (even partially), false otherwise.
    bool load_from_file(const std::string &path) ;

    // Validate and clamp configuration values. Returns true if config is valid (or was fixed), false if fatal.
    bool validate() ;
};
