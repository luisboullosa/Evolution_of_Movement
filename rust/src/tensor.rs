pub use crate::brain::gpu_pack::{DecisionRowGpu, PopulationBrainTensor};
pub use crate::population_tensor::{PopulationTensor, RowRngManager};
use crate::config::{Config, GridTopology};
use crate::hex_dish::HexDishLayout;
use crate::hex_simulation::grid_cell_count;
use crate::individual::Individual;
use crate::environment::Environment;
use crate::step_engine::{self, DecisionRow};
use crate::neural_network::NeuralNetwork;
use cust::context::Context;
use cust::error::CudaError;
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy, DevicePointer};
use cust::module::Module;
use cust::quick_init;
use cust::stream::{Stream, StreamFlags};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::convert::TryFrom;
use std::env;
use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;

macro_rules! cuda_debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        eprintln!($($arg)*);
    };
}

/// CUDA context + module kept alive for the process lifetime (`quick_init` context
/// is destroyed on drop, which invalidates all subsequent driver API calls).
struct CachedCudaRuntime {
    _context: Context,
    module: Module,
}

fn cached_cuda_module() -> Result<&'static Module, TensorKernelGpuError> {
    static RUNTIME: OnceLock<Result<CachedCudaRuntime, String>> = OnceLock::new();
    let result = RUNTIME.get_or_init(|| {
        let ptx_path = tensor_kernel_ptx_path().map_err(|e| e.to_string())?;
        cuda_debug!("[cuda] loading PTX from {:?}", ptx_path);
        let context = quick_init().map_err(|e| e.to_string())?;
        let module = Module::from_file(&ptx_path).map_err(|e| e.to_string())?;
        Ok(CachedCudaRuntime {
            _context: context,
            module,
        })
    });
    match result {
        Ok(wrapper) => Ok(&wrapper.module),
        Err(msg) => Err(TensorKernelGpuError::Init(msg.clone())),
    }
}

include!(concat!(env!("OUT_DIR"), "/tensor_kernel_path.rs"));

/// Persistent GPU buffers for brain forward (reallocated only when capacity grows).
pub struct GpuForwardState {
    capacity: usize,
    env_buf: Option<DeviceBuffer<f64>>,
    occupancy_buf: Option<DeviceBuffer<i32>>,
    positions_buf: Option<DeviceBuffer<u64>>,
    resources_buf: Option<DeviceBuffer<f64>>,
    brain_kind_buf: Option<DeviceBuffer<u8>>,
    layer1_weights_buf: Option<DeviceBuffer<f64>>,
    layer1_biases_buf: Option<DeviceBuffer<f64>>,
    multi_weights_buf: Option<DeviceBuffer<f64>>,
    layer2_weights_buf: Option<DeviceBuffer<f64>>,
    layer2_biases_buf: Option<DeviceBuffer<f64>>,
    spatial_landmark_count_buf: Option<DeviceBuffer<i32>>,
    spatial_edge_count_buf: Option<DeviceBuffer<i32>>,
    spatial_edges_buf: Option<DeviceBuffer<i32>>,
    spatial_edge_weights_buf: Option<DeviceBuffer<f64>>,
    spatial_readout_weights_buf: Option<DeviceBuffer<f64>>,
    spatial_readout_biases_buf: Option<DeviceBuffer<f64>>,
    spatial_landmark_sectors_buf: Option<DeviceBuffer<u8>>,
    spatial_learn_rate_buf: Option<DeviceBuffer<f64>>,
    spatial_message_passes_buf: Option<DeviceBuffer<i32>>,
    spatial_runtime_buf: Option<DeviceBuffer<f64>>,
    goo_node_count_buf: Option<DeviceBuffer<i32>>,
    goo_ops_buf: Option<DeviceBuffer<u8>>,
    goo_parent_count_buf: Option<DeviceBuffer<u8>>,
    goo_parents_buf: Option<DeviceBuffer<i32>>,
    goo_params_buf: Option<DeviceBuffer<f64>>,
    dish_neighbors_buf: Option<DeviceBuffer<i32>>,
    decision_buf: Option<DeviceBuffer<DecisionRowGpu>>,
}

impl GpuForwardState {
    pub fn new() -> Self {
        GpuForwardState {
            capacity: 0,
            env_buf: None,
            occupancy_buf: None,
            positions_buf: None,
            resources_buf: None,
            brain_kind_buf: None,
            layer1_weights_buf: None,
            layer1_biases_buf: None,
            multi_weights_buf: None,
            layer2_weights_buf: None,
            layer2_biases_buf: None,
            spatial_landmark_count_buf: None,
            spatial_edge_count_buf: None,
            spatial_edges_buf: None,
            spatial_edge_weights_buf: None,
            spatial_readout_weights_buf: None,
            spatial_readout_biases_buf: None,
            spatial_landmark_sectors_buf: None,
            spatial_learn_rate_buf: None,
            spatial_message_passes_buf: None,
            spatial_runtime_buf: None,
            goo_node_count_buf: None,
            goo_ops_buf: None,
            goo_parent_count_buf: None,
            goo_parents_buf: None,
            goo_params_buf: None,
            dish_neighbors_buf: None,
            decision_buf: None,
        }
    }
}

unsafe impl DeviceCopy for DecisionRowGpu {}

/// GPU forward with CPU fallback using shared step_engine semantics.
pub fn run_brain_forward(
    tensor: &mut PopulationBrainTensor,
    individuals: &mut [Individual],
    env_resources: &[f64],
    occupancy: &[i32],
    config: &Config,
    hidden_buf: &mut [f64],
    out2_buf: &mut [f64],
    gpu_state: &mut GpuForwardState,
    dish: Option<&HexDishLayout>,
) -> (Vec<DecisionRow>, bool) {
    let count = tensor.active_count();
    if count == 0 {
        return (Vec::new(), false);
    }

    cuda_debug!("[cuda] brain forward for {} rows", count);
    match run_brain_forward_gpu(tensor, env_resources, occupancy, config, gpu_state, count, dish) {
        Ok(decisions) => (decisions, true),
        Err(err) => {
            static GPU_FAIL_LOGS: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            let n = GPU_FAIL_LOGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 5 || n % 1000 == 0 {
                eprintln!(
                    "brain_step_kernel failed ({}); falling back to CPU forward (occurrence #{})",
                    err,
                    n + 1
                );
            }
            (
                run_brain_forward_cpu(
                    individuals,
                    env_resources,
                    occupancy,
                    config,
                    hidden_buf,
                    out2_buf,
                    dish,
                ),
                false,
            )
        }
    }
}

fn run_brain_forward_cpu(
    individuals: &mut [Individual],
    env_resources: &[f64],
    occupancy: &[i32],
    config: &Config,
    hidden_buf: &mut [f64],
    out2_buf: &mut [f64],
    dish: Option<&HexDishLayout>,
) -> Vec<DecisionRow> {
    let cell_count = grid_cell_count(config);
    let env = make_env_view(env_resources, cell_count);
    let mut decisions = Vec::with_capacity(individuals.len());
    for ind in individuals.iter_mut() {
        let pos = ind.position % cell_count;
        let local_pop = occupancy.get(pos).copied().unwrap_or(1);
        let edges =
            step_engine::edge_resources_for_position(config, &env, pos, cell_count, dish);
        let center = env.resources.get(pos).copied().unwrap_or(0.0);
        decisions.push(step_engine::forward_decision(
            ind,
            edges,
            center,
            local_pop,
            config.movement_cost_factor,
            config.resolved_movement_output_mode(),
            hidden_buf,
            out2_buf,
        ));
    }
    decisions
}

fn make_env_view(resources: &[f64], width: usize) -> Environment {
    let w = width.max(1);
    Environment {
        width: w,
        resources: resources.to_vec(),
        perlin_base: vec![0.0; w],
        perlin_time: 0.0,
    }
}

fn run_brain_forward_gpu(
    tensor: &mut PopulationBrainTensor,
    env_resources: &[f64],
    occupancy: &[i32],
    config: &Config,
    gpu_state: &mut GpuForwardState,
    count: usize,
    dish: Option<&HexDishLayout>,
) -> Result<Vec<DecisionRow>, TensorKernelGpuError> {
    let module = cached_cuda_module()?;
    let function = module.get_function("brain_step_kernel")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let base = &tensor.base;
    let cell_count = grid_cell_count(config);
    let grid_topology = if dish.is_some() {
        2i32
    } else {
        match config.grid_topology {
            GridTopology::Hex2d => 1i32,
            GridTopology::Ring1d => 0i32,
        }
    };
    let grid_height = if let Some(d) = dish {
        d.bounding_size
    } else if config.grid_height > 0 {
        config.grid_height
    } else {
        config.grid_width
    };
    let grid_width = dish.map(|d| d.bounding_size).unwrap_or_else(|| config.grid_width.max(1));

    let env_buf = sync_device_buffer(&mut gpu_state.env_buf, env_resources)?;
    let occupancy_buf = sync_device_buffer(&mut gpu_state.occupancy_buf, occupancy)?;
    let positions_host = base.positions_host_u64(count);
    let positions_buf = sync_device_buffer(&mut gpu_state.positions_buf, &positions_host)?;
    let resources_buf =
        sync_device_buffer(&mut gpu_state.resources_buf, &base.resources[..count])?;
    let brain_kind_buf =
        sync_device_buffer(&mut gpu_state.brain_kind_buf, &tensor.brain_kind[..count])?;

    let layer1_weights_buf = sync_device_buffer_opt(
        &mut gpu_state.layer1_weights_buf,
        base.layer1_weights_slice(),
    )?;
    let layer1_biases_buf =
        sync_device_buffer_opt(&mut gpu_state.layer1_biases_buf, base.layer1_biases_slice())?;
    let multi_weights_buf =
        sync_device_buffer_opt(&mut gpu_state.multi_weights_buf, base.multi_weights_slice())?;
    let layer2_weights_buf = sync_device_buffer_opt(
        &mut gpu_state.layer2_weights_buf,
        base.layer2_weights_slice(),
    )?;
    let layer2_biases_buf =
        sync_device_buffer_opt(&mut gpu_state.layer2_biases_buf, base.layer2_biases_slice())?;

    let spatial_landmark_count_buf = sync_device_buffer(
        &mut gpu_state.spatial_landmark_count_buf,
        &tensor.spatial_landmark_count[..count],
    )?;
    let spatial_edge_count_buf = sync_device_buffer(
        &mut gpu_state.spatial_edge_count_buf,
        &tensor.spatial_edge_count[..count],
    )?;
    let spatial_edges_buf = sync_device_buffer(
        &mut gpu_state.spatial_edges_buf,
        &tensor.spatial_edges[..count * tensor.max_edges * 3],
    )?;
    let spatial_edge_weights_buf = sync_device_buffer(
        &mut gpu_state.spatial_edge_weights_buf,
        &tensor.spatial_edge_weights[..count * tensor.max_edges],
    )?;
    let spatial_readout_weights_buf = sync_device_buffer(
        &mut gpu_state.spatial_readout_weights_buf,
        &tensor.spatial_readout_weights[..count * tensor.max_readout_features * 2],
    )?;
    let spatial_readout_biases_buf = sync_device_buffer(
        &mut gpu_state.spatial_readout_biases_buf,
        &tensor.spatial_readout_biases[..count * 2],
    )?;
    let spatial_landmark_sectors_buf = sync_device_buffer(
        &mut gpu_state.spatial_landmark_sectors_buf,
        &tensor.spatial_landmark_sectors[..count * tensor.max_brain_nodes],
    )?;
    let spatial_learn_rate_buf = sync_device_buffer(
        &mut gpu_state.spatial_learn_rate_buf,
        &tensor.spatial_learn_rate[..count],
    )?;
    let spatial_message_passes_buf = sync_device_buffer(
        &mut gpu_state.spatial_message_passes_buf,
        &tensor.spatial_message_passes[..count],
    )?;
    let spatial_runtime_buf = sync_device_buffer(
        &mut gpu_state.spatial_runtime_buf,
        &tensor.spatial_runtime[..count * tensor.max_brain_nodes * crate::brain::EMBED_DIM],
    )?;

    let goo_node_count_buf =
        sync_device_buffer(&mut gpu_state.goo_node_count_buf, &tensor.goo_node_count[..count])?;
    let goo_ops_buf = sync_device_buffer(
        &mut gpu_state.goo_ops_buf,
        &tensor.goo_ops[..count * tensor.max_brain_nodes],
    )?;
    let goo_parent_count_buf = sync_device_buffer(
        &mut gpu_state.goo_parent_count_buf,
        &tensor.goo_parent_count[..count * tensor.max_brain_nodes],
    )?;
    let goo_parents_buf = sync_device_buffer(
        &mut gpu_state.goo_parents_buf,
        &tensor.goo_parents[..count * tensor.max_brain_nodes * crate::brain::gpu_pack::MAX_GOO_PARENTS],
    )?;
    let goo_params_buf = sync_device_buffer(
        &mut gpu_state.goo_params_buf,
        &tensor.goo_params[..count * tensor.max_goo_params],
    )?;

    let dish_neighbors_buf = if let Some(d) = dish {
        let neighbors_i32: Vec<i32> = d
            .neighbors_flat
            .iter()
            .map(|&n| n as i32)
            .collect();
        Some(sync_device_buffer(
            &mut gpu_state.dish_neighbors_buf,
            &neighbors_i32,
        )?)
    } else {
        None
    };

    let mut decision_host = vec![DecisionRowGpu::default(); count];
    let decision_buf = sync_device_buffer(&mut gpu_state.decision_buf, &decision_host)?;

    unsafe {
        launch!(
            function<<<
                ((count as u32 + 127) / 128).max(1),
                128,
                0,
                stream
            >>>(
                usize_to_i32(count),
                usize_to_i32(cell_count),
                usize_to_i32(grid_width.max(1)),
                usize_to_i32(grid_height.max(1)),
                grid_topology,
                env_buf.as_device_ptr(),
                occupancy_buf.as_device_ptr(),
                config.movement_cost_factor,
                base.nodes as i32,
                base.inputs as i32,
                base.nodes2 as i32,
                base.layers,
                usize_to_i32(base.layer1_stride()),
                usize_to_i32(base.multi_stride()),
                usize_to_i32(base.layer2_weights_stride()),
                usize_to_i32(base.layer2_bias_stride()),
                usize_to_i32(tensor.max_brain_nodes),
                usize_to_i32(tensor.max_edges),
                usize_to_i32(tensor.max_goo_params),
                usize_to_i32(tensor.max_readout_features),
                brain_kind_buf.as_device_ptr(),
                positions_buf.as_device_ptr(),
                resources_buf.as_device_ptr(),
                device_ptr_or_null(layer1_weights_buf),
                device_ptr_or_null(layer1_biases_buf),
                device_ptr_or_null(multi_weights_buf),
                device_ptr_or_null(layer2_weights_buf),
                device_ptr_or_null(layer2_biases_buf),
                spatial_landmark_count_buf.as_device_ptr(),
                spatial_edge_count_buf.as_device_ptr(),
                spatial_edges_buf.as_device_ptr(),
                spatial_edge_weights_buf.as_device_ptr(),
                spatial_readout_weights_buf.as_device_ptr(),
                spatial_readout_biases_buf.as_device_ptr(),
                spatial_landmark_sectors_buf.as_device_ptr(),
                spatial_learn_rate_buf.as_device_ptr(),
                spatial_message_passes_buf.as_device_ptr(),
                spatial_runtime_buf.as_device_ptr(),
                goo_node_count_buf.as_device_ptr(),
                goo_ops_buf.as_device_ptr(),
                goo_parent_count_buf.as_device_ptr(),
                goo_parents_buf.as_device_ptr(),
                goo_params_buf.as_device_ptr(),
                device_ptr_or_null(dish_neighbors_buf),
                decision_buf.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;
    decision_buf.copy_to(&mut decision_host)?;
    spatial_runtime_buf.copy_to(
        &mut tensor.spatial_runtime[..count * tensor.max_brain_nodes * crate::brain::EMBED_DIM],
    )?;

    gpu_state.capacity = gpu_state.capacity.max(count);
    Ok(decision_host.into_iter().map(DecisionRow::from).collect())
}

fn sync_device_buffer<'a, T: DeviceCopy>(
    slot: &'a mut Option<DeviceBuffer<T>>,
    data: &[T],
) -> Result<&'a DeviceBuffer<T>, TensorKernelGpuError> {
    if data.is_empty() {
        return Err(TensorKernelGpuError::Init("empty device buffer upload".into()));
    }
    let needs_alloc = slot.as_ref().map(|b| b.len() != data.len()).unwrap_or(true);
    if needs_alloc {
        *slot = Some(DeviceBuffer::from_slice(data)?);
    } else if let Some(buf) = slot.as_mut() {
        buf.copy_from(data)?;
    }
    Ok(slot.as_ref().unwrap())
}

fn sync_device_buffer_opt<'a, T: DeviceCopy>(
    slot: &'a mut Option<DeviceBuffer<T>>,
    data: &[T],
) -> Result<Option<&'a DeviceBuffer<T>>, TensorKernelGpuError> {
    if data.is_empty() {
        Ok(None)
    } else {
        sync_device_buffer(slot, data).map(Some)
    }
}

#[derive(Debug)]
enum TensorKernelGpuError {
    MissingPtx,
    Init(String),
    Cuda(CudaError),
}

impl From<CudaError> for TensorKernelGpuError {
    fn from(err: CudaError) -> Self {
        TensorKernelGpuError::Cuda(err)
    }
}

impl fmt::Display for TensorKernelGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorKernelGpuError::MissingPtx => write!(f, "TENSOR_KERNEL_PTX is not set"),
            TensorKernelGpuError::Init(msg) => write!(f, "CUDA init failed: {}", msg),
            TensorKernelGpuError::Cuda(e) => write!(f, "CUDA error: {}", e),
        }
    }
}

impl std::error::Error for TensorKernelGpuError {}

fn tensor_kernel_ptx_path() -> Result<PathBuf, TensorKernelGpuError> {
    if let Ok(env_path) = env::var("TENSOR_KERNEL_PTX") {
        return Ok(PathBuf::from(env_path));
    }
    let fallback = PathBuf::from(BUILT_TENSOR_KERNEL_PTX);
    if fallback.exists() {
        Ok(fallback)
    } else {
        Err(TensorKernelGpuError::MissingPtx)
    }
}

fn device_ptr_or_null<T: DeviceCopy>(buffer: Option<&DeviceBuffer<T>>) -> DevicePointer<T> {
    buffer
        .map(|buf| buf.as_device_ptr())
        .unwrap_or(DevicePointer::null())
}

fn usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}
