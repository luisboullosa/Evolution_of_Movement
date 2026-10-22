#include "NeuralNetwork.h"
#include <numeric>
#include <cmath>

NeuralNetwork::NeuralNetwork(size_t nodes_, size_t inputs_, int layers_, size_t nodes2_) {
    nodes = nodes_;
    inputs = inputs_ ? inputs_ : 1;
    layers = layers_ > 1 ? 2 : 1;
    nodes2 = (layers == 2) ? (nodes2_ ? nodes2_ : nodes_) : 0;
    type = (inputs == 1) ? Type::SINGLE_INPUT : Type::MULTI_INPUT;
    if (type == Type::SINGLE_INPUT) {
        weights.resize(nodes);
    } else {
        weights_multi.resize(nodes * inputs);
    }
    biases.resize(nodes);
    if (layers == 2) {
        // second layer maps from first-layer nodes to nodes2 outputs
        weights2.resize(nodes2 * nodes);
        biases2.resize(nodes2);
    }
}

std::pair<double,double> NeuralNetwork::forward(double input) const {
    // backwards-compatible single-input path
    // compute first-layer activations
    std::vector<double> layer1(nodes, 0.0);
    if (type == Type::SINGLE_INPUT) {
        for (size_t i = 0; i < nodes; ++i) {
            layer1[i] = std::tanh(weights[i] * input + biases[i]);
        }
    } else {
        for (size_t i = 0; i < nodes; ++i) {
            double w = weights_multi[i * inputs + 0];
            layer1[i] = std::tanh(w * input + biases[i]);
        }
    }
    if (layers == 1) {
        double sum = 0.0;
        for (double v : layer1) sum += v;
        double direction = std::tanh(sum);
        double magnitude = std::abs(sum);
        return {direction, magnitude};
    }
    // layers == 2: compute second layer from layer1
    std::vector<double> layer2(nodes2, 0.0);
    for (size_t i = 0; i < nodes2; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < nodes; ++j) {
            dot += weights2[i * nodes + j] * layer1[j];
        }
        layer2[i] = std::tanh(dot + biases2[i]);
    }
    double sum = 0.0;
    for (double v : layer2) sum += v;
    double direction = std::tanh(sum);
    double magnitude = std::abs(sum);
    return {direction, magnitude};
}

std::pair<double,double> NeuralNetwork::forward(const std::vector<double> &input) const {
    // multi-input path. If the network is single-input we fall back to using
    // the first input and legacy weights.
    // compute first layer
    std::vector<double> layer1(nodes, 0.0);
    if (type == Type::SINGLE_INPUT) {
        double in0 = input.empty() ? 0.0 : input[0];
        for (size_t i = 0; i < nodes; ++i) {
            layer1[i] = std::tanh(weights[i] * in0 + biases[i]);
        }
    } else {
        size_t in_count = std::min(inputs, input.size());
        for (size_t i = 0; i < nodes; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < in_count; ++j) dot += weights_multi[i * inputs + j] * input[j];
            layer1[i] = std::tanh(dot + biases[i]);
        }
    }
    if (layers == 1) {
        double sum = 0.0;
        for (double v: layer1) sum += v;
        double direction = std::tanh(sum);
        double magnitude = std::abs(sum);
        return {direction, magnitude};
    }
    // second layer
    std::vector<double> layer2(nodes2, 0.0);
    for (size_t i = 0; i < nodes2; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < nodes; ++j) dot += weights2[i * nodes + j] * layer1[j];
        layer2[i] = std::tanh(dot + biases2[i]);
    }
    double sum = 0.0;
    for (double v: layer2) sum += v;
    double direction = std::tanh(sum);
    double magnitude = std::abs(sum);
    return {direction, magnitude};
}
