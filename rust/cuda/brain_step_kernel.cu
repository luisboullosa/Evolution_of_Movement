#include <cstdint>
#include <cmath>

#define SENSOR_DIM 8
#define PERCEPTRON_SENSOR_DIM 3
#define EMBED_DIM 8
#define THOUGHT_DIM 8
#define MAX_GOO_PARENTS 8

#define BRAIN_PERCEPTRON 0
#define BRAIN_MLP 1
#define BRAIN_SPATIAL 2
#define BRAIN_GOO 3

#define GOO_GENERATE 0
#define GOO_AGGREGATE 1
#define GOO_SCORE 2
#define GOO_KEEP_BEST 3

struct DecisionRowGpu {
    double move_signal;
    double direction_degrees;
    double movement_spend;
    double gathered;
    int move_one_step;
};

__device__ static inline double nn_activate(double x) {
    return tanh(x);
}

__device__ static inline double direction_signal_to_degrees(double signal) {
    double normalized = tanh(signal);
    normalized = (normalized + 1.0) * 0.5;
    if (normalized < 0.0) normalized = 0.0;
    if (normalized > 1.0) normalized = 1.0;
    return normalized * 360.0;
}

__device__ static inline void decode_outputs(
    const double* layer, int layer_len, double* move_signal, double* direction_signal
) {
    double sum = 0.0;
    for (int i = 0; i < layer_len; ++i) {
        sum += layer[i];
    }
    *move_signal = sum;
    *direction_signal = (layer_len > 1) ? layer[1] : layer[0];
}

__device__ static inline void build_ring_edges(
    const double* env, int width, int pos, double edges[6]
) {
    int w = width > 0 ? width : 1;
    int p = pos % w;
    if (p < 0) p += w;
    double center = env[p];
    double left = env[(p + w - 1) % w];
    double right = env[(p + 1) % w];
    edges[0] = left;
    edges[1] = left;
    edges[2] = center;
    edges[3] = right;
    edges[4] = right;
    edges[5] = center;
}

__device__ static inline int wrap(int v, int m) {
    if (m <= 0) return 0;
    int r = v % m;
    return (r < 0) ? (r + m) : r;
}

__device__ static inline int hex_to_flat(int q, int r, int width, int height) {
    q = wrap(q, width);
    r = wrap(r, height);
    return r * width + q;
}

__device__ static inline void flat_to_hex(int flat, int width, int* q, int* r) {
    *q = flat % width;
    *r = flat / width;
}

__device__ static inline void hex_neighbor(int flat, int sector, int width, int height, int* out_flat) {
    int q, r;
    flat_to_hex(flat, width, &q, &r);
    static const int dq[6] = {1, 1, 0, -1, -1, 0};
    static const int dr[6] = {0, -1, -1, 0, 1, 1};
    int s = sector % 6;
    *out_flat = hex_to_flat(q + dq[s], r + dr[s], width, height);
}

__device__ static inline void build_hex_edges(
    const double* env, int width, int height, int cell_count, int pos, double edges[6]
) {
    int p = pos % cell_count;
    if (p < 0) p += cell_count;
    for (int i = 0; i < 6; ++i) {
        int nflat;
        hex_neighbor(p, i, width, height, &nflat);
        edges[i] = env[nflat];
    }
}

__device__ static inline void build_sensors(
    const double edges[6], int local_pop, double resources, double sensors[SENSOR_DIM]
) {
    for (int i = 0; i < 6; ++i) sensors[i] = edges[i];
    sensors[6] = (double)local_pop;
    sensors[7] = resources;
}

__device__ static inline void build_perceptron_sensors(
    double center, int local_pop, double resources, double sensors[SENSOR_DIM]
) {
    for (int i = 0; i < SENSOR_DIM; ++i) sensors[i] = 0.0;
    sensors[0] = center;
    sensors[1] = (double)local_pop;
    sensors[2] = resources;
}

__device__ static inline void decision_from_output(
    double move_signal_raw,
    double direction_signal_raw,
    double center,
    int local_pop,
    double individual_resources,
    double movement_cost_factor,
    DecisionRowGpu* out
) {
    double direction_degrees = direction_signal_to_degrees(direction_signal_raw);
    out->move_signal = move_signal_raw;
    out->direction_degrees = direction_degrees;
    if (move_signal_raw <= 0.0) {
        double denom = (double)(local_pop < 1 ? 1 : local_pop);
        out->gathered = center / denom;
        out->movement_spend = 0.0;
        out->move_one_step = 0;
        return;
    }
    double spend = fabs(move_signal_raw) * movement_cost_factor;
    if (isnan(spend)) spend = 0.0;
    if (spend > individual_resources) spend = individual_resources;
    out->gathered = 0.0;
    out->movement_spend = spend;
    out->move_one_step = 1;
}

__device__ static inline void forward_perceptron(
    const double sensors[SENSOR_DIM],
    int nodes,
    int inputs,
    int row,
    int multi_stride,
    int layer1_stride,
    const double* multi_weights,
    const double* layer1_biases,
    double hidden[32],
    double* move_signal,
    double* direction_signal
) {
    int read_inputs = inputs < SENSOR_DIM ? inputs : SENSOR_DIM;
    if (nodes > 32) nodes = 32;
    for (int i = 0; i < nodes; ++i) {
        double acc = layer1_biases[row * layer1_stride + i];
        int row_multi_base = row * multi_stride;
        for (int j = 0; j < read_inputs; ++j) {
            acc += multi_weights[row_multi_base + i * inputs + j] * sensors[j];
        }
        hidden[i] = nn_activate(acc);
    }
    decode_outputs(hidden, nodes, move_signal, direction_signal);
}

__device__ static inline void forward_mlp(
    const double sensors[SENSOR_DIM],
    int nodes,
    int inputs,
    int nodes2,
    int layers,
    int row,
    int multi_stride,
    int layer1_stride,
    int layer2_weights_stride,
    int layer2_bias_stride,
    const double* multi_weights,
    const double* layer1_weights,
    const double* layer1_biases,
    const double* layer2_weights,
    const double* layer2_biases,
    double hidden[32],
    double out2[32],
    double* move_signal,
    double* direction_signal
) {
    int read_inputs = inputs < SENSOR_DIM ? inputs : SENSOR_DIM;
    if (nodes > 32) nodes = 32;
    if (nodes2 > 32) nodes2 = 32;

    if (inputs <= 1) {
        double center = 0.0;
        for (int i = 0; i < SENSOR_DIM; ++i) center += sensors[i];
        center /= (double)SENSOR_DIM;
        for (int i = 0; i < nodes; ++i) {
            double acc = layer1_biases[row * layer1_stride + i];
            acc += layer1_weights[row * layer1_stride + i] * center;
            hidden[i] = nn_activate(acc);
        }
    } else {
        for (int i = 0; i < nodes; ++i) {
            double acc = layer1_biases[row * layer1_stride + i];
            int row_multi_base = row * multi_stride;
            for (int j = 0; j < read_inputs; ++j) {
                acc += multi_weights[row_multi_base + i * inputs + j] * sensors[j];
            }
            hidden[i] = nn_activate(acc);
        }
    }

    if (layers == 2 && nodes2 > 0 && layer2_weights != nullptr && layer2_biases != nullptr) {
        int row_w2 = row * layer2_weights_stride;
        int row_b2 = row * layer2_bias_stride;
        for (int j = 0; j < nodes2; ++j) {
            double acc = layer2_biases[row_b2 + j];
            for (int i = 0; i < nodes; ++i) {
                acc += layer2_weights[row_w2 + i * nodes2 + j] * hidden[i];
            }
            out2[j] = nn_activate(acc);
        }
        decode_outputs(out2, nodes2, move_signal, direction_signal);
    } else {
        decode_outputs(hidden, nodes, move_signal, direction_signal);
    }
}

__device__ static inline void landmark_observation(
    const double sensors[SENSOR_DIM],
    int position,
    unsigned char sector_mask,
    double observe[EMBED_DIM]
) {
    for (int i = 0; i < EMBED_DIM; ++i) observe[i] = 0.0;
    int active = 0;
    for (int s = 0; s < 6; ++s) {
        if ((sector_mask >> s) & 1) {
            if (s < EMBED_DIM) observe[s] += sensors[s];
            active++;
        }
    }
    if (active > 0) {
        double inv = 1.0 / (double)active;
        for (int s = 0; s < 6 && s < EMBED_DIM; ++s) {
            if ((sector_mask >> s) & 1) observe[s] *= inv;
        }
    }
    if (EMBED_DIM > 6) observe[6] = sensors[6];
    if (EMBED_DIM > 7) observe[7] = sensors[7];
    if (EMBED_DIM > SENSOR_DIM) {
        observe[SENSOR_DIM] = fmod((double)position, 1000.0) / 1000.0;
    }
}

__device__ static inline void forward_spatial(
    const double sensors[SENSOR_DIM],
    int position,
    int row,
    int max_brain_nodes,
    int max_edges,
    int max_readout_features,
    int landmark_count,
    int edge_count,
    const int* spatial_edges,
    const double* spatial_edge_weights,
    const double* spatial_readout_weights,
    const double* spatial_readout_biases,
    const unsigned char* spatial_landmark_sectors,
    double learn_rate,
    int message_passes,
    double* spatial_runtime,
    double hidden[32],
    double* move_signal,
    double* direction_signal
) {
    int rt_base = row * max_brain_nodes * EMBED_DIM;
    int sector_base = row * max_brain_nodes;
    if (landmark_count > 0) {
        int best_idx = 0;
        double best_dist = 1e300;
        for (int i = 0; i < landmark_count && i < max_brain_nodes; ++i) {
            unsigned char mask = spatial_landmark_sectors[sector_base + i];
            if (mask == 0) mask = 0x3F;
            double observe[EMBED_DIM];
            landmark_observation(sensors, position, mask, observe);
            double dist = 0.0;
            for (int d = 0; d < EMBED_DIM; ++d) {
                double diff = spatial_runtime[rt_base + i * EMBED_DIM + d] - observe[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
        unsigned char best_mask = spatial_landmark_sectors[sector_base + best_idx];
        if (best_mask == 0) best_mask = 0x3F;
        double observe[EMBED_DIM];
        landmark_observation(sensors, position, best_mask, observe);
        for (int d = 0; d < EMBED_DIM; ++d) {
            int idx = rt_base + best_idx * EMBED_DIM + d;
            spatial_runtime[idx] += learn_rate * (observe[d] - spatial_runtime[idx]);
        }
    }

    double activations[32];
    for (int i = 0; i < max_brain_nodes; ++i) activations[i] = 0.0;
    for (int i = 0; i < landmark_count && i < max_brain_nodes; ++i) {
        double sum = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            sum += spatial_runtime[rt_base + i * EMBED_DIM + d];
        }
        activations[i] = sum / (double)EMBED_DIM;
    }

    int edge_base = row * max_edges * 3;
    int ew_base = row * max_edges;
    for (int pass = 0; pass < message_passes; ++pass) {
        double next[32];
        for (int i = 0; i < max_brain_nodes; ++i) next[i] = activations[i];
        for (int e = 0; e < edge_count && e < max_edges; ++e) {
            int from = spatial_edges[edge_base + e * 3];
            int to = spatial_edges[edge_base + e * 3 + 1];
            int widx = spatial_edges[edge_base + e * 3 + 2];
            if (from >= 0 && from < landmark_count && to >= 0 && to < max_brain_nodes && widx >= 0 && widx < max_edges) {
                double w = spatial_edge_weights[ew_base + widx];
                next[to] += w * nn_activate(activations[from]);
            }
        }
        for (int i = 0; i < landmark_count && i < max_brain_nodes; ++i) {
            activations[i] = nn_activate(next[i]);
        }
    }

    for (int i = 0; i < 32; ++i) hidden[i] = 0.0;
    for (int i = 0; i < landmark_count && i < 32; ++i) hidden[i] = activations[i];

    int feat_len = landmark_count + SENSOR_DIM;
    if (feat_len > max_readout_features) feat_len = max_readout_features;
    double features[64];
    for (int i = 0; i < feat_len && i < 64; ++i) features[i] = 0.0;
    for (int i = 0; i < landmark_count && i < feat_len; ++i) features[i] = activations[i];
    for (int i = 0; i < SENSOR_DIM && (landmark_count + i) < feat_len; ++i) {
        features[landmark_count + i] = sensors[i];
    }

    int rw_base = row * max_readout_features * 2;
    int rb_base = row * 2;
    double mv = spatial_readout_biases[rb_base];
    double dv = spatial_readout_biases[rb_base + 1];
    for (int i = 0; i < feat_len; ++i) {
        mv += spatial_readout_weights[rw_base + i * 2] * features[i];
        dv += spatial_readout_weights[rw_base + i * 2 + 1] * features[i];
    }
    *move_signal = nn_activate(mv);
    *direction_signal = nn_activate(dv);
}

__device__ static inline void goo_matvec(
    const double* params, int dim, const double* input, double* out
) {
    for (int i = 0; i < dim; ++i) {
        double acc = params[dim * dim + i];
        for (int j = 0; j < dim; ++j) {
            acc += params[i * dim + j] * input[j];
        }
        out[i] = nn_activate(acc);
    }
}

__device__ static inline void forward_goo(
    const double sensors[SENSOR_DIM],
    int row,
    int max_brain_nodes,
    int max_goo_params,
    int node_count,
    const unsigned char* goo_ops,
    const unsigned char* goo_parent_count,
    const int* goo_parents,
    const double* goo_params,
    double* move_signal,
    double* direction_signal
) {
    double base[THOUGHT_DIM];
    for (int i = 0; i < THOUGHT_DIM; ++i) base[i] = 0.0;
    for (int i = 0; i < THOUGHT_DIM && i < SENSOR_DIM; ++i) base[i] = sensors[i];

    double values[32][THOUGHT_DIM];
    double scores[32];
    for (int i = 0; i < 32; ++i) {
        scores[i] = -1e300;
        for (int j = 0; j < THOUGHT_DIM; ++j) values[i][j] = 0.0;
    }

    int param_base = row * max_goo_params;
    int op_base = row * max_brain_nodes;
    int pc_base = row * max_brain_nodes;
    int p_base = row * max_brain_nodes * MAX_GOO_PARENTS;

    for (int ni = 0; ni < node_count && ni < max_brain_nodes && ni < 32; ++ni) {
        unsigned char op = goo_ops[op_base + ni];
        unsigned char pc = goo_parent_count[pc_base + ni];
        int parents[MAX_GOO_PARENTS];
        for (int pi = 0; pi < MAX_GOO_PARENTS; ++pi) {
            parents[pi] = goo_parents[p_base + ni * MAX_GOO_PARENTS + pi];
        }

        if (op == GOO_GENERATE) {
            int offset = ni % (max_goo_params - THOUGHT_DIM * THOUGHT_DIM - THOUGHT_DIM + 1);
            if (offset < 0) offset = 0;
            goo_matvec(&goo_params[param_base + offset], THOUGHT_DIM, base, values[ni]);
        } else if (op == GOO_AGGREGATE) {
            int np = pc > 0 ? (int)pc : 1;
            for (int j = 0; j < THOUGHT_DIM; ++j) {
                double s = 0.0;
                for (int pi = 0; pi < np && pi < MAX_GOO_PARENTS; ++pi) {
                    int pid = parents[pi];
                    s += (pid >= 0 && pid < 32) ? values[pid][j] : base[j];
                }
                values[ni][j] = nn_activate(s / (double)np);
            }
        } else if (op == GOO_SCORE) {
            int pid = (pc > 0) ? parents[0] : 0;
            for (int j = 0; j < THOUGHT_DIM; ++j) {
                values[ni][j] = (pid >= 0 && pid < 32) ? values[pid][j] : base[j];
            }
            double sc = 0.0;
            for (int j = 0; j < THOUGHT_DIM; ++j) {
                sc += values[ni][j] * goo_params[param_base + (j % max_goo_params)];
            }
            scores[ni] = sc;
        } else if (op == GOO_KEEP_BEST) {
            int best_parent = (pc > 0) ? parents[0] : 0;
            double best_score = -1e300;
            for (int pi = 0; pi < (int)pc && pi < MAX_GOO_PARENTS; ++pi) {
                int pid = parents[pi];
                double sc = (pid >= 0 && pid < 32) ? scores[pid] : 0.0;
                if (sc > best_score || (sc == best_score && pid < best_parent)) {
                    best_score = sc;
                    best_parent = pid;
                }
            }
            for (int j = 0; j < THOUGHT_DIM; ++j) {
                values[ni][j] = (best_parent >= 0 && best_parent < 32) ? values[best_parent][j] : base[j];
            }
        }
    }

    int last = (node_count > 0) ? node_count - 1 : 0;
    if (last >= 32) last = 31;
    double sum = 0.0;
    for (int j = 0; j < THOUGHT_DIM; ++j) sum += values[last][j];
    *move_signal = sum;
    *direction_signal = values[last][1];
}

__device__ static inline void build_dish_edges(
    const double* env, int cell_count, int pos, const int32_t* dish_neighbors, double edges[6]
) {
    int p = pos % cell_count;
    if (p < 0) p += cell_count;
    double center = env[p];
    for (int i = 0; i < 6; ++i) {
        int n = dish_neighbors[p * 6 + i];
        if (n < 0 || n >= cell_count) n = p;
        edges[i] = env[n];
    }
    (void)center;
}

extern "C" __global__ void brain_step_kernel(
    int active_count,
    int env_cell_count,
    int grid_width,
    int grid_height,
    int grid_topology,
    const double* env_resources,
    const int32_t* occupancy,
    double movement_cost_factor,
    int nodes,
    int inputs,
    int nodes2,
    int layers,
    int layer1_stride,
    int multi_stride,
    int layer2_weights_stride,
    int layer2_bias_stride,
    int max_brain_nodes,
    int max_edges,
    int max_goo_params,
    int max_readout_features,
    const unsigned char* brain_kind,
    const uint64_t* positions,
    const double* resources,
    const double* layer1_weights,
    const double* layer1_biases,
    const double* multi_weights,
    const double* layer2_weights,
    const double* layer2_biases,
    const int32_t* spatial_landmark_count,
    const int32_t* spatial_edge_count,
    const int32_t* spatial_edges,
    const double* spatial_edge_weights,
    const double* spatial_readout_weights,
    const double* spatial_readout_biases,
    const unsigned char* spatial_landmark_sectors,
    const double* spatial_learn_rate,
    const int32_t* spatial_message_passes,
    double* spatial_runtime,
    const int32_t* goo_node_count,
    const unsigned char* goo_ops,
    const unsigned char* goo_parent_count,
    const int32_t* goo_parents,
    const double* goo_params,
    const int32_t* dish_neighbors,
    DecisionRowGpu* decision_out
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= active_count) return;

    int width = grid_width > 0 ? grid_width : 1;
    int height = grid_height > 0 ? grid_height : width;
    int cell_count = env_cell_count > 0 ? env_cell_count : width;
    int pos = (int)(positions[row] % (uint64_t)cell_count);
    if (pos < 0) pos += cell_count;

    double edges[6];
    if (grid_topology == 2 && dish_neighbors != nullptr) {
        build_dish_edges(env_resources, cell_count, pos, dish_neighbors, edges);
    } else if (grid_topology == 1) {
        build_hex_edges(env_resources, width, height, cell_count, pos, edges);
    } else {
        build_ring_edges(env_resources, width, pos, edges);
    }

    double center = env_resources[pos];

    int local_pop = 0;
    if (occupancy != nullptr && pos >= 0 && pos < cell_count) {
        local_pop = occupancy[pos];
    }

    double sensors[SENSOR_DIM];
    if (brain_kind[row] == BRAIN_PERCEPTRON) {
        build_perceptron_sensors(center, local_pop, resources[row], sensors);
    } else {
        build_sensors(edges, local_pop, resources[row], sensors);
    }

    double hidden[32];
    double out2[32];
    double move_raw = 0.0;
    double dir_raw = 0.0;

    unsigned char kind = brain_kind[row];
    if (kind == BRAIN_PERCEPTRON) {
        forward_perceptron(
            sensors, nodes, inputs, row, multi_stride, layer1_stride,
            multi_weights, layer1_biases, hidden, &move_raw, &dir_raw
        );
    } else if (kind == BRAIN_MLP) {
        forward_mlp(
            sensors, nodes, inputs, nodes2, layers, row,
            multi_stride, layer1_stride, layer2_weights_stride, layer2_bias_stride,
            multi_weights, layer1_weights, layer1_biases, layer2_weights, layer2_biases,
            hidden, out2, &move_raw, &dir_raw
        );
    } else if (kind == BRAIN_SPATIAL) {
        forward_spatial(
            sensors, pos, row, max_brain_nodes, max_edges, max_readout_features,
            spatial_landmark_count[row], spatial_edge_count[row],
            spatial_edges, spatial_edge_weights,
            spatial_readout_weights, spatial_readout_biases, spatial_landmark_sectors,
            spatial_learn_rate[row], spatial_message_passes[row],
            spatial_runtime, hidden, &move_raw, &dir_raw
        );
    } else if (kind == BRAIN_GOO) {
        forward_goo(
            sensors, row, max_brain_nodes, max_goo_params,
            goo_node_count[row], goo_ops, goo_parent_count, goo_parents, goo_params,
            &move_raw, &dir_raw
        );
    }

    decision_from_output(
        move_raw, dir_raw, center, local_pop, resources[row],
        movement_cost_factor, &decision_out[row]
    );
}
