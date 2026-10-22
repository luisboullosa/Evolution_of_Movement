#pragma once
#include "Environment.h"
#include "Individual.h"
#include "Config.h"
#include <vector>
#include <random>
#include <map>

struct Simulation {
    Environment env;
    std::vector<Individual> individuals;
    std::mt19937 rng;
    int next_id = 1;
    // last action info for visualization: per individual resources spent and delta moved
    std::vector<double> last_spend;
    std::vector<int> last_delta;
    // number of movers in the most recent step (computed during step() to avoid an extra pass)
    size_t last_movers = 0;

    Simulation(const Config &cfg);
    void init_population(int n);
    void step();
    void run();
    // dump environment resources and individual positions to TSV files
    void dump_env(const std::string &path) const;
    void dump_individuals(const std::string &path) const;
    // Simple ASCII visualization of the environment and individuals.
    // cols: number of characters wide to compress the environment into.
    void visualize(size_t cols = 100) const;
    // Return the visualization string (does not print).
    // height: number of rows for the vertical bar plot
    std::string get_visualization(size_t cols = 100, size_t height = 8) const;

    // stored configuration
    Config config;
    // in-memory cache for nn_index to avoid repeated file writes during simulation
    // mutable so const dump_individuals can update it
    mutable std::map<std::string, std::string> nn_index_cache;
    mutable std::string nn_index_target_dir;
    mutable bool nn_index_dirty = false;
};
