#include <cstdint>
#include <cstdbool>
#include <math.h>

__device__ static inline double nn_activate(double x) {
    return tanh(x);
}

__device__ static inline double signum(double value) {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    if (value == 0.0) {
        return 0.0;
    }
    return value;
}

extern "C" __global__ void tensor_step_kernel(
    int active_count,
    int env_width,
    const double* env_resources,
    const int32_t* occupancy,
    bool use_occupancy,
    double movement_cost_factor,
    int nodes,
    int inputs,
    int nodes2,
    int layers,
    const uint64_t* row_seeds,
    int layer1_stride,
    int multi_stride,
    int layer2_weights_stride,
    int layer2_bias_stride,
    const int32_t* ids,
    uint64_t* positions,
    double* resources,
    int32_t* ages,
    double* layer1_weights,
    double* layer1_biases,
    double* multi_weights,
    double* layer2_weights,
    double* layer2_biases,
    double* hidden_buf,
    double* out2_buf,
    double* last_spend,
    int32_t* last_delta,
    int32_t* last_movers_out
) {
    if (active_count <= 0) {
        if (last_movers_out != nullptr) {
            *last_movers_out = 0;
        }
        return;
    }

    int width = env_width > 0 ? env_width : 1;
    int movers = 0;
    double stack_inputs[5];

    for (int row = 0; row < active_count; ++row) {
        uint64_t current_position = positions[row];
        int pos = (width == 1) ? 0 : (int)(current_position % (uint64_t)width);
        if (pos < 0) {
            pos += width;
        }
        if (pos >= width) {
            pos = width - 1;
        }

        double center = 0.0;
        if (env_width > 0 && env_resources != nullptr) {
            if (pos >= 0 && pos < env_width) {
                center = env_resources[pos];
            }
        }

        double left = center;
        double right = center;
        if (env_resources != nullptr && env_width > 0) {
            int left_idx = pos == 0 ? (width - 1) : (pos - 1);
            int right_idx = (pos + 1) % width;
            if (left_idx >= 0 && left_idx < env_width) {
                left = env_resources[left_idx];
            }
            if (right_idx >= 0 && right_idx < env_width) {
                right = env_resources[right_idx];
            }
        }

        int local_population = 0;
        if (use_occupancy && occupancy != nullptr && pos >= 0 && pos < env_width) {
            local_population = occupancy[pos];
        }

        stack_inputs[0] = left;
        stack_inputs[1] = center;
        stack_inputs[2] = right;
        stack_inputs[3] = (double)local_population;
        stack_inputs[4] = resources[row];

        double direction = 0.0;
        double magnitude = 0.0;
        int row_layer1_base = row * layer1_stride;
        int row_multi_base = row * multi_stride;
        int row_layer2_weights_base = row * layer2_weights_stride;
        int row_layer2_bias_base = row * layer2_bias_stride;

        if (nodes > 0 && layer1_biases != nullptr && hidden_buf != nullptr) {
            if (inputs <= 1) {
                double input_value = center;
                for (int i = 0; i < nodes; ++i) {
                    int idx = row_layer1_base + i;
                    double acc = layer1_biases[idx];
                    if (layer1_weights != nullptr && layer1_stride > 0) {
                        acc += layer1_weights[idx] * input_value;
                    }
                    hidden_buf[i] = nn_activate(acc);
                }
            } else {
                int read_len = inputs < 5 ? inputs : 5;
                for (int i = 0; i < nodes; ++i) {
                    double acc = layer1_biases[row_layer1_base + i];
                    if (multi_weights != nullptr && multi_stride > 0) {
                        for (int j = 0; j < read_len; ++j) {
                            int idx = row_multi_base + i * inputs + j;
                            acc += multi_weights[idx] * stack_inputs[j];
                        }
                    }
                    hidden_buf[i] = nn_activate(acc);
                }
            }

            if (layers == 2 && nodes2 > 0 && layer2_biases != nullptr && layer2_weights != nullptr && out2_buf != nullptr) {
                double sum_out2 = 0.0;
                for (int j = 0; j < nodes2; ++j) {
                    double acc = layer2_biases[row_layer2_bias_base + j];
                    if (layer2_weights_stride > 0) {
                        for (int i = 0; i < nodes; ++i) {
                            int idx = row_layer2_weights_base + i * nodes2 + j;
                            acc += layer2_weights[idx] * hidden_buf[i];
                        }
                    }
                    double out2 = nn_activate(acc);
                    out2_buf[j] = out2;
                    sum_out2 += out2;
                }
                magnitude = sum_out2;
            } else {
                double sum_hidden = 0.0;
                for (int i = 0; i < nodes; ++i) {
                    sum_hidden += hidden_buf[i];
                }
                magnitude = sum_hidden;
            }
        }

        double out = 0.0;
        if (magnitude < 0.5) {
            double denom = (double)(local_population < 1 ? 1 : local_population);
            if (denom != 0.0) {
                resources[row] += center / denom;
            }
        } else {
            double spend = magnitude * movement_cost_factor;
            if (isnan(spend)) {
                spend = 0.0;
            }
            double spend_clamped = spend;
            if (spend_clamped > resources[row]) {
                spend_clamped = resources[row];
            }
            resources[row] -= spend_clamped;
            if (isnan(direction)) {
                out = spend_clamped;
            } else {
                out = spend_clamped * signum(direction);
            }
        }

        if (out != 0.0) {
            int32_t delta = (int32_t)round(out);
            last_spend[row] = fabs(out);
            last_delta[row] = delta;
            movers += 1;
            int64_t new_position = (int64_t)current_position + delta;
            if (width != 0) {
                new_position %= width;
            }
            if (new_position < 0) {
                new_position += width;
            }
            if (new_position < 0) {
                new_position = 0;
            }
            positions[row] = (uint64_t)new_position;
        }
    }

    if (last_movers_out != nullptr) {
        *last_movers_out = movers;
    }
}
