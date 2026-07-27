# CUDA Kernel Argument Alignment

Host launch: `run_brain_forward_gpu` in [rust/src/tensor.rs](../rust/src/tensor.rs).  
Kernel: `brain_step_kernel` in [rust/cuda/brain_step_kernel.cu](../rust/cuda/brain_step_kernel.cu).

The kernel performs **forward + decision only**. It does not mutate positions, resources, or environment. Rust applies decisions via `step_engine::apply_decisions_sequential`.

## Launch configuration

- Grid: `((active_count + 127) / 128, 1, 1)` blocks × 128 threads
- One thread per individual (`row = blockIdx.x * blockDim.x + threadIdx.x`)

## Argument table

| Position | Kernel parameter | Type | Host source |
| --- | --- | --- | --- |
| 1 | `active_count` | `int` | `usize_to_i32(count)` |
| 2 | `env_cell_count` | `int` | `grid_cell_count(config)` |
| 3 | `grid_width` | `int` | `config.grid_width` |
| 4 | `grid_height` | `int` | `config.grid_height` or width |
| 5 | `grid_topology` | `int` | `0` ring, `1` hex |
| 6 | `env_resources` | `const double*` | env slice device buffer |
| 7 | `occupancy` | `const int32_t*` | snapshot occupancy |
| 8 | `movement_cost_factor` | `double` | `config.movement_cost_factor` |
| 9 | `nodes` | `int` | MLP layer-1 size |
| 10 | `inputs` | `int` | sensor / NN input count |
| 11 | `nodes2` | `int` | MLP layer-2 size |
| 12 | `layers` | `int` | MLP layer count |
| 13 | `layer1_stride` | `int` | per-row layer-1 width |
| 14 | `multi_stride` | `int` | `nodes * inputs` |
| 15 | `layer2_weights_stride` | `int` | layer-2 weight row stride |
| 16 | `layer2_bias_stride` | `int` | layer-2 bias row stride |
| 17 | `max_brain_nodes` | `int` | spatial/goo padding |
| 18 | `max_edges` | `int` | spatial edge padding |
| 19 | `max_goo_params` | `int` | goo param padding |
| 20 | `brain_kind` | `const unsigned char*` | `0=P, 1=MLP, 2=SG, 3=Goo` |
| 21 | `positions` | `const uint64_t*` | read-only positions |
| 22 | `resources` | `const double*` | read-only resources |
| 23 | `layer1_weights` | `const double*` | optional / null |
| 24 | `layer1_biases` | `const double*` | optional / null |
| 25 | `multi_weights` | `const double*` | Perceptron + multi-input MLP |
| 26 | `layer2_weights` | `const double*` | optional |
| 27 | `layer2_biases` | `const double*` | optional |
| 28 | `spatial_landmark_count` | `const int32_t*` | per row |
| 29 | `spatial_edge_count` | `const int32_t*` | per row |
| 30 | `spatial_edges` | `const int32_t*` | `[pop, max_edges, 3]` flat |
| 31 | `spatial_edge_weights` | `const double*` | padded edge weights |
| 32 | `spatial_readout_weights` | `const double*` | readout |
| 33 | `spatial_readout_biases` | `const double*` | 2 per row |
| 34 | `spatial_learn_rate` | `const double*` | per row |
| 35 | `spatial_message_passes` | `const int32_t*` | per row |
| 36 | `spatial_runtime` | `double*` | `[pop, max_lm, EMBED_DIM]` read/write |
| 37 | `goo_node_count` | `const int32_t*` | per row |
| 38 | `goo_ops` | `const unsigned char*` | op type per node |
| 39 | `goo_parent_count` | `const unsigned char*` | per node |
| 40 | `goo_parents` | `const int32_t*` | `[pop, max_nodes, MAX_GOO_PARENTS]` |
| 41 | `goo_params` | `const double*` | padded params |
| 42 | `decision_out` | `DecisionRowGpu*` | output per row |

## DecisionRowGpu layout (`repr(C)`)

```rust
struct DecisionRowGpu {
    move_signal: f64,
    direction_degrees: f64,
    movement_spend: f64,
    gathered: f64,
    move_one_step: i32,  // 0 or 1
}
```

## Persistent device buffers

`GpuForwardState` in `tensor.rs` reuses `DeviceBuffer` allocations when slice lengths are unchanged (`sync_device_buffer`). Host data is refreshed each step with `copy_from`.

## Parity verification

- Rust integration: `rust/tests/parity.rs` (`cargo test --test parity`)
- Script: `scripts/compare_cuda_simulations.py` (CPU `--no-use-gpu-step` vs GPU `--use-gpu-step`)

Build PTX via `build.rs` compiling `cuda/brain_step_kernel.cu` (override arch with `CUDA_ARCH`, path with `TENSOR_KERNEL_PTX`).
