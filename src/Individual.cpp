#include "Individual.h"
#include <algorithm>
#include <cmath>

// defaults; will be overridden by Simulation using the Config
double Individual::maintenance_cost_per_node = 0.01;
double Individual::reproduction_threshold = 20.0;
double Individual::reproduction_cost = 5.0;
double Individual::mutation_rate = 0.01;
double Individual::mutation_sigma = 0.1;
double Individual::movement_cost_factor = 0.1;

Individual::Individual(const NeuralNetwork &n, size_t pos, int id_)
    : nn(n), position(pos), id(id_) {}

double Individual::act(double left_resources, double center_resources, double right_resources, int local_population, std::mt19937 &rng) {
    // Decide: prepare NN inputs. Historically a single input = local resource.
    // New input vector layout: {left, center, right [, occupancy]} depending on
    // configuration (nn_use_occupancy). The caller (Simulation) provides real
    // neighbor values; Individual just respects nn_use_occupancy by including
    // or omitting the occupancy slot when calling the NN.
    std::vector<double> inputs;
    // inputs: left, center, right, occupancy, internal resources
    inputs.reserve(5);
    inputs.push_back(left_resources);
    inputs.push_back(center_resources);
    inputs.push_back(right_resources);
    // occupancy may be included or not depending on the runtime configuration
    inputs.push_back(static_cast<double>(local_population));
    // include the individual's internal resources as an input
    inputs.push_back(resources);

    // If the network was constructed with fewer inputs, nn.forward will handle
    // trimming/compatibility (it looks at its own inputs count).
    auto out = nn.forward(inputs);
    double direction = out.first; // -1..1
    double magnitude = out.second; // positive

    // probabilistic decision: if magnitude < threshold -> gather, else move
    if (magnitude < 0.5) {
        // gather: allocation proportional to resources available / local pop
        double gathered = center_resources / std::max(1, local_population);
        resources += gathered;
        return 0.0;
    } else {
        // move: spend some resources proportional to magnitude (clamped)
        double spend = std::min(resources, magnitude * movement_cost_factor);
        resources -= spend;
        return spend;
    }
}

Individual Individual::reproduce(std::mt19937 &rng, int &next_id) {
    // cost of reproduction
    double repro_cost = reproduction_cost;
    double half = resources / 2.0;
    resources = half - repro_cost/2.0;
    Individual child = *this;
    child.id = next_id++;
    child.resources = half - repro_cost/2.0;

    // mutate small chance per weight/bias
    std::uniform_real_distribution<double> prob(0.0,1.0);
    std::normal_distribution<double> mut(0.0, mutation_sigma);
    for (auto &w: child.nn.weights) if (prob(rng) < mutation_rate) w += mut(rng);
    for (auto &b: child.nn.biases)  if (prob(rng) < mutation_rate) b += mut(rng);

    if (child.nn.type == NeuralNetwork::Type::MULTI_INPUT) {
        for (auto &w: child.nn.weights_multi) if (prob(rng) < mutation_rate) w += mut(rng);
    }

    // mutate second-layer weights/biases if present
    if (child.nn.layers == 2) {
        for (auto &w: child.nn.weights2) if (prob(rng) < mutation_rate) w += mut(rng);
        for (auto &b: child.nn.biases2)  if (prob(rng) < mutation_rate) b += mut(rng);
    }

    return child;
}
