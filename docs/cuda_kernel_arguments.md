# CUDA Kernel Argument Alignment

This document verifies that `run_tensor_kernel_gpu` passes the arguments to
`tensor_step_kernel` in the same order, with compatible types, and with the
correct helpers for optional buffers. The CUDA kernel signature lives at
[rust/cuda/tensor_kernel.cu](rust/cuda/tensor_kernel.cu#L22-L88) and the host launch is in
[rust/src/tensor.rs](rust/src/tensor.rs#L458-L537).

| Position | Kernel parameter | Type | Host value / helper | Notes |
| --- | --- | --- | --- | --- |
| 1 | `active_count` | `int` | `usize_to_i32(count)` | Number of rows; converted safely to `i32`. |
| 2 | `env_width` | `int` | `usize_to_i32(width)` | Ensures zero-safe minimum in `run_tensor_kernel`. |
| 3 | `env_resources` | `const double*` | `env_buf.as_device_ptr()` | `env_buf` holds the whole slice (even though kernel reads up to `env_width`). |
| 4 | `occupancy` | `const int32_t*` | `occupancy_buf.as_device_ptr()` | Buffer always allocated/filled to `occupancy.len()`. |
| 5 | `use_occupancy` | `bool` | `config.nn_use_occupancy` | Direct config boolean. |
| 6 | `movement_cost_factor` | `double` | `config.movement_cost_factor` | Scalar control value. |
| 7 | `nodes` | `int` | `tensor.nodes as i32` | Layer 1 neuron count. |
| 8 | `inputs` | `int` | `tensor.inputs as i32` | Number of NN inputs; kernel uses it for multi-weight branch. |
| 9 | `nodes2` | `int` | `tensor.nodes2 as i32` | Optional second layer size. |
| 10 | `layers` | `int` | `tensor.layers` | Saved from config. |
| 11 | `row_seeds` | `const uint64_t*` | `row_seeds_buf.as_device_ptr()` | Sliced to `count` entries before upload. |
| 12 | `layer1_stride` | `int` | `usize_to_i32(tensor.layer1_stride)` | Matches layout used to `copy_to_tensor`. |
| 13 | `multi_stride` | `int` | `usize_to_i32(tensor.multi_stride)` | Needed when `inputs > 1`. |
| 14 | `layer2_weights_stride` | `int` | `usize_to_i32(tensor.layer2_weights_stride)` | Kernel iterates per-node weights. |
| 15 | `layer2_bias_stride` | `int` | `usize_to_i32(tensor.layer2_bias_stride)` | One bias per `nodes2`. |
| 16 | `ids` | `const int32_t*` | `ids_buf.as_device_ptr()` | `tensor.ids` up to `count`. |
| 17 | `positions` | `uint64_t*` | `positions_buf.as_device_ptr()` | Host vector converted to `u64`. |
| 18 | `resources` | `double*` | `resources_buf.as_device_ptr()` | Updated in-place. |
| 19 | `ages` | `int32_t*` | `ages_buf.as_device_ptr()` | Mirrors CPU path. |
| 20 | `layer1_weights` | `double*` | `device_ptr_or_null(layer1_weights_buf.as_ref())` | Optional backing; `null` is safe when stride is zero. |
| 21 | `layer1_biases` | `double*` | `device_ptr_or_null(layer1_biases_buf.as_ref())` | Same helper. |
| 22 | `multi_weights` | `double*` | `device_ptr_or_null(multi_weights_buf.as_ref())` | Only present when `tensor.multi_stride > 0`. |
| 23 | `layer2_weights` | `double*` | `device_ptr_or_null(layer2_weights_buf.as_ref())` | Passed per-node to second layer. |
| 24 | `layer2_biases` | `double*` | `device_ptr_or_null(layer2_biases_buf.as_ref())` | Optional for second layer. |
| 25 | `hidden_buf` | `double*` | `device_ptr_or_null(hidden_buf_dev.as_ref())` | Temporary size matches `tensor.nodes`. |
| 26 | `out2_buf` | `double*` | `device_ptr_or_null(out2_buf_dev.as_ref())` | Used when `nodes2 > 0`. |
| 27 | `last_spend` | `double*` | `last_spend_buf.as_device_ptr()` | Host buffer mirrored to GPU. |
| 28 | `last_delta` | `int32_t*` | `last_delta_buf.as_device_ptr()` | Int32 per row. |
| 29 | `last_movers_out` | `int32_t*` | `last_movers_buf.as_device_ptr()` | Kernel writes mover count back (converted to `usize`). |

Null handling for optional buffers is already centralized in
`device_ptr_or_null`, so the kernel never dereferences invalid pointers.
With this alignment confirmed, the next investigation step is to inspect why
the `UnknownError` occurs despite the launch arguments matching the signature.