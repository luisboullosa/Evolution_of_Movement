#pragma once
#include "NeuralNetwork.h"
#include <random>

struct Individual {
    NeuralNetwork nn;
    double resources = 10.0;
    size_t position = 0;
    int id = 0;
    // age in timesteps (incremented each step)
    int age = 0;

    static double maintenance_cost_per_node;
    static double reproduction_threshold;
    static double reproduction_cost;
    static double mutation_rate;
    static double mutation_sigma;
    static double movement_cost_factor;

    Individual() = default;
    Individual(const NeuralNetwork &n, size_t pos, int id_);

    // take action: either gather or move. Returns resources spent on movement (0 if gather).
    // Provide explicit neighborhood inputs: left, center, right and occupancy (local population).
    // If occupancy is not used, set occupancy to 0 or the value will be ignored depending on Config.
    double act(double left_resources, double center_resources, double right_resources, int local_population, std::mt19937 &rng);

    // reproduction: returns a new Individual (child) and deducts costs
    Individual reproduce(std::mt19937 &rng, int &next_id);
};
