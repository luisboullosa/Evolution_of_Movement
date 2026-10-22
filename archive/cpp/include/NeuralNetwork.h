#pragma once
#include <vector>

// Very small, single-hidden-layer neural network used by Individuals.
// Historically this supported a single scalar input only. We now optionally
// support multiple inputs (e.g., left/center/right resource + occupancy)
// while keeping the original single-input behaviour as the default for
// backward compatibility.
struct NeuralNetwork {
    enum class Type { SINGLE_INPUT = 0, MULTI_INPUT = 1 };

    // For SINGLE_INPUT networks the `weights` vector contains one weight per node
    // (the historic layout). For MULTI_INPUT networks the `weights_multi`
    // flat vector stores weights in row-major order: node_i, input_j ->
    // weights_multi[i * inputs + j].
    std::vector<double> weights; // legacy: one weight per node (single-input)
    std::vector<double> weights_multi; // size == nodes * inputs when multi-input
    std::vector<double> biases;
    // optional second layer weights and biases (for a 2-layer network)
    std::vector<double> weights2; // size == nodes * nodes2 when present (row-major)
    std::vector<double> biases2;

    size_t nodes = 0;
    size_t inputs = 1; // number of input features expected by network
    size_t nodes2 = 0; // if >0, indicates a second layer with this many nodes
    Type type = Type::SINGLE_INPUT;
    int layers = 1; // 1 = single hidden layer (legacy), 2 = two-layer network

    NeuralNetwork() = default;
    // nodes: number of hidden/output nodes; inputs: number of numeric inputs (1 by default)
    NeuralNetwork(size_t nodes_, size_t inputs_ = 1, int layers_ = 1, size_t nodes2_ = 0);

    // forward: single-input (backwards-compatible)
    std::pair<double,double> forward(double input) const;

    // forward: multi-input API (accepts vector of inputs). If the network was
    // constructed as single-input this simply re-uses the legacy weights.
    std::pair<double,double> forward(const std::vector<double> &input) const;
};
