# GPU Optimization Notes

## Current state (branch `feature/evolvable-graph-networks`)

- **Batch simulation (1D ring)**: optional CUDA path via `cust` + `tensor_kernel.cu` for MLP/perceptron brains only.
- **Hex / evolvable brains**: CPU path in `hex_simulation.rs` (spatial graph, GoO graph, hex grid).
- **Interactive viewer**: wgpu compute for resource grid updates only.

## Changes in this branch

1. **Cached CUDA module** — PTX module and kernel function load once via `OnceLock` in [`rust/src/tensor.rs`](../rust/src/tensor.rs), avoiding per-step `quick_init()` + module reload.
2. **Debug logging gated** — CUDA `[DEBUG]` output only in debug builds (`cuda_debug!` macro).
3. **Brain-aware routing** — `Simulation::use_cpu_brain_step` selects CPU when `grid_topology == hex_2d` or architecture policy is not `perceptron_only`.

## Remaining opportunities

| Priority | Item | Notes |
|----------|------|-------|
| High | Parallelize `tensor_step_kernel` | One thread per individual; per-row scratch buffers |
| High | Resident GPU buffers | Keep population tensors on device between steps |
| Medium | CPU SIMD batch forward | For perceptron/MLP on hex runs |
| Medium | 2D Perlin on GPU | Extend `resource_update.wgsl` pattern to batch env step |
| Low | [NVIDIA CCCL](https://github.com/nvidia/cccl) integration | Use CUB/Thrust in `cuda/brain_step.cu` after brain layouts stabilize |

## CCCL evaluation (spike plan)

CCCL is header-only C++ (CUB, Thrust, libcudacxx). Recommended approach:

1. Vendor CCCL headers via git submodule or `build.rs` fetch.
2. Prototype `cuda/brain_step.cu` with `thrust::for_each` over individual rows for fixed-topology MLP.
3. Keep spatial/GoO graphs on CPU until packed genome representation exists.
4. Compare against cached-module baseline using `scripts/compare_cuda_simulations.py`.

## Benchmarking

```powershell
cd rust
cargo test --lib
cargo run --release -- --grid-topology=hex_2d --architecture-policy=evolve_mlp --steps=1000 --initial-population=200
```

For GPU vs CPU parity on 1D runs:

```powershell
python scripts/compare_cuda_simulations.py
```
