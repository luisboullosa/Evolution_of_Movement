#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>

int main(){
    // Legacy single-input behaviour
    {
        NeuralNetwork nn(2);
        // deterministic weights/biases
        nn.weights[0] = 0.5; nn.biases[0] = -0.1;
        nn.weights[1] = -0.3; nn.biases[1] = 0.2;

        double input = 0.7;
        auto out = nn.forward(input);

        // compute expected
        double s = 0.0;
        s += std::tanh(0.5*input - 0.1);
        s += std::tanh(-0.3*input + 0.2);
        double expected_dir = std::tanh(s);
        double expected_mag = std::abs(s);

        double eps = 1e-9;
        if (std::abs(out.first - expected_dir) > eps || std::abs(out.second - expected_mag) > eps) {
            std::cerr << "NeuralNetwork forward mismatch (single): got ("<<out.first<<","<<out.second<<") expected ("<<expected_dir<<","<<expected_mag<<")\n";
            return 1;
        }
    }

    // Multi-input behaviour (2 nodes, 3 inputs)
    {
        NeuralNetwork nn(2, 3);
        // weights_multi layout: node0: w00,w01,w02 ; node1: w10,w11,w12
        // set some deterministic weights
        nn.weights_multi[0] = 0.5; // node0 input0
        nn.weights_multi[1] = 0.0; // node0 input1
        nn.weights_multi[2] = 0.0; // node0 input2
        nn.weights_multi[3] = -0.3; // node1 input0
        nn.weights_multi[4] = 0.0;
        nn.weights_multi[5] = 0.0;
        nn.biases[0] = -0.1; nn.biases[1] = 0.2;

        std::vector<double> input = {0.7, 0.0, 0.0};
        auto out = nn.forward(input);

        double s = 0.0;
        s += std::tanh(0.5*input[0] - 0.1);
        s += std::tanh(-0.3*input[0] + 0.2);
        double expected_dir = std::tanh(s);
        double expected_mag = std::abs(s);
        double eps = 1e-9;
        if (std::abs(out.first - expected_dir) > eps || std::abs(out.second - expected_mag) > eps) {
            std::cerr << "NeuralNetwork forward mismatch (multi): got ("<<out.first<<","<<out.second<<") expected ("<<expected_dir<<","<<expected_mag<<")\n";
            return 1;
        }
    }

    return 0;
}
