#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>

int main(){
    // Two-layer network: first layer 3 nodes, second layer 2 nodes, 3 inputs
    NeuralNetwork nn(3, 3, 2, 2);
    // Set deterministic weights for layer1 (weights_multi) and biases
    // layer1: nodes=3, inputs=3
    for (size_t i=0;i<nn.weights_multi.size();++i) nn.weights_multi[i] = 0.0;
    nn.weights_multi[0] = 0.5; // node0 input0
    nn.weights_multi[3] = -0.4; // node1 input0
    nn.weights_multi[6] = 0.2; // node2 input0
    nn.biases[0] = -0.1; nn.biases[1] = 0.05; nn.biases[2] = 0.0;

    // layer2 weights (nodes2 x nodes)
    for (size_t i=0;i<nn.weights2.size();++i) nn.weights2[i] = 0.0;
    // node0 of layer2 connects to first-layer nodes with weights [0.3, -0.2, 0.1]
    nn.weights2[0 * nn.nodes + 0] = 0.3;
    nn.weights2[0 * nn.nodes + 1] = -0.2;
    nn.weights2[0 * nn.nodes + 2] = 0.1;
    // node1 of layer2 connects with [ -0.1, 0.25, 0.05 ]
    nn.weights2[1 * nn.nodes + 0] = -0.1;
    nn.weights2[1 * nn.nodes + 1] = 0.25;
    nn.weights2[1 * nn.nodes + 2] = 0.05;
    nn.biases2[0] = 0.0; nn.biases2[1] = 0.02;

    // input vector
    std::vector<double> input = {0.7, 0.0, 0.0};

    // manual forward pass:
    // layer1 activations
    auto a0 = std::tanh(0.5*input[0] + nn.biases[0]);
    auto a1 = std::tanh(-0.4*input[0] + nn.biases[1]);
    auto a2 = std::tanh(0.2*input[0] + nn.biases[2]);
    // layer2 dot inputs
    double d0 = 0.3*a0 + (-0.2)*a1 + 0.1*a2 + nn.biases2[0];
    double d1 = (-0.1)*a0 + 0.25*a1 + 0.05*a2 + nn.biases2[1];
    double l0 = std::tanh(d0);
    double l1 = std::tanh(d1);
    double s = l0 + l1;
    double expected_dir = std::tanh(s);
    double expected_mag = std::abs(s);

    auto out = nn.forward(input);
    double eps = 1e-9;
    if (std::abs(out.first - expected_dir) > eps || std::abs(out.second - expected_mag) > eps) {
        std::cerr << "Two-layer forward mismatch: got ("<<out.first<<","<<out.second<<") expected ("<<expected_dir<<","<<expected_mag<<")\n";
        return 1;
    }

    // also test scalar forward path (should treat scalar as first input)
    auto out2 = nn.forward(0.7);
    // compute expected using scalar path (we already did the same values)
    if (std::abs(out2.first - expected_dir) > eps || std::abs(out2.second - expected_mag) > eps) {
        std::cerr << "Two-layer scalar forward mismatch: got ("<<out2.first<<","<<out2.second<<") expected ("<<expected_dir<<","<<expected_mag<<")\n";
        return 1;
    }

    std::cout << "two-layer NN forward OK\n";
    return 0;
}
