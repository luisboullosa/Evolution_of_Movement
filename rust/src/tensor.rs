use crate::config::Config;
use crate::individual::Individual;
use crate::neural_network::NeuralNetwork;
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

include!(concat!(env!("OUT_DIR"), "/tensor_kernel_path.rs"));

/// Tensor-backed population metadata for future GPU/CUDA integration.
/// The structure now holds contiguous buffers for every per-individual field so
/// the data can be copied directly to a CUDA tensor without visiting
/// `Individual` instances.
pub struct PopulationTensor {
    max_population: usize,
    active_count: usize,
    base_seed: u64,
    nodes: usize,
    inputs: usize,
    nodes2: usize,
    layers: i32,
    layer1_stride: usize,
    layer2_weights_stride: usize,
    layer2_bias_stride: usize,
    multi_stride: usize,

    ids: Vec<i32>,
    positions: Vec<usize>,
    resources: Vec<f64>,
    ages: Vec<i32>,

    nn_weights: Vec<f64>,
    nn_weights_multi: Vec<f64>,
    nn_biases: Vec<f64>,
    nn_weights2: Vec<f64>,
    nn_biases2: Vec<f64>,
}

impl PopulationTensor {
    /// Allocate a tensor descriptor sized for the configured maximum population.
    pub fn new(cfg: &Config) -> Self {
        let safe_max = cfg.max_population.max(cfg.initial_population).max(1) as usize;
        let nodes = cfg.nn_nodes.max(0) as usize;
        let inputs = cfg.nn_inputs.max(0) as usize;
        let nodes2 = cfg.nn_nodes2.max(0) as usize;
        let layers = cfg.nn_layers;
        let layer1_stride = nodes;
        let multi_stride = nodes.saturating_mul(inputs);
        let layer2_weights_stride = nodes.saturating_mul(nodes2);
        let layer2_bias_stride = nodes2;
        PopulationTensor {
            max_population: safe_max,
            active_count: 0,
            base_seed: cfg.seed as u64,
            nodes,
            inputs,
            nodes2,
            layers,
            layer1_stride,
            layer2_weights_stride,
            layer2_bias_stride,
            multi_stride,
            ids: vec![0; safe_max],
            positions: vec![0; safe_max],
            resources: vec![0.0; safe_max],
            ages: vec![0; safe_max],
            nn_weights: vec![0.0; safe_max.saturating_mul(layer1_stride)],
            nn_weights_multi: vec![0.0; safe_max.saturating_mul(multi_stride)],
            nn_biases: vec![0.0; safe_max.saturating_mul(layer1_stride)],
            nn_weights2: vec![0.0; safe_max.saturating_mul(layer2_weights_stride)],
            nn_biases2: vec![0.0; safe_max.saturating_mul(layer2_bias_stride)],
        }
    }

    /// Record how many rows (individuals) are currently active.
    pub fn record_active(&mut self, count: usize) {
        self.active_count = count.min(self.max_population);
    }

    /// Return the last-known active population size.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Return the configured maximum population.
    pub fn max_population(&self) -> usize {
        self.max_population
    }

    pub fn nodes(&self) -> usize {
        self.nodes
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn nodes2(&self) -> usize {
        self.nodes2
    }

    pub fn layers(&self) -> i32 {
        self.layers
    }

    /// Prepare the tensor buffers from the supplied slice of individuals.
    pub fn sync_from_individuals(&mut self, individuals: &[Individual]) {
        let count = individuals.len().min(self.max_population);
        self.record_active(count);
        for (row, ind) in individuals.iter().take(count).enumerate() {
            self.ids[row] = ind.id;
            self.positions[row] = ind.position;
            self.resources[row] = ind.resources;
            self.ages[row] = ind.age;
            self.copy_nn_weights(row, &ind.nn);
        }
        for row in count..self.max_population {
            self.ids[row] = 0;
            self.positions[row] = 0;
            self.resources[row] = 0.0;
            self.ages[row] = 0;
        }
    }

    pub fn ids_slice(&self) -> &[i32] {
        &self.ids[..self.active_count]
    }

    pub fn positions_slice(&self) -> &[usize] {
        &self.positions[..self.active_count]
    }

    pub fn resources_slice(&self) -> &[f64] {
        &self.resources[..self.active_count]
    }

    pub fn ages_slice(&self) -> &[i32] {
        &self.ages[..self.active_count]
    }

    pub fn layer1_weights_slice(&self) -> &[f64] {
        &self.nn_weights[..self.active_count.saturating_mul(self.layer1_stride)]
    }

    pub fn layer1_biases_slice(&self) -> &[f64] {
        &self.nn_biases[..self.active_count.saturating_mul(self.layer1_stride)]
    }

    pub fn multi_weights_slice(&self) -> &[f64] {
        &self.nn_weights_multi[..self.active_count.saturating_mul(self.multi_stride)]
    }

    pub fn layer2_weights_slice(&self) -> &[f64] {
        &self.nn_weights2[..self.active_count.saturating_mul(self.layer2_weights_stride)]
    }

    pub fn layer2_biases_slice(&self) -> &[f64] {
        &self.nn_biases2[..self.active_count.saturating_mul(self.layer2_bias_stride)]
    }

    /// Copy tensor state back into the provided individuals slice.
    pub fn apply_to_individuals(&self, individuals: &mut [Individual]) {
        let count = self.active_count.min(individuals.len());
        for row in 0..count {
            let ind = &mut individuals[row];
            ind.id = self.ids[row];
            ind.position = self.positions[row];
            ind.resources = self.resources[row];
            ind.age = self.ages[row];
            Self::copy_back_nn_weights(ind, self, row);
        }
    }

    fn copy_back_nn_weights(ind: &mut Individual, tensor: &PopulationTensor, row: usize) {
        copy_tensor_to_nn(&mut ind.nn, tensor, row);
    }
    fn copy_nn_weights(&mut self, row: usize, nn: &NeuralNetwork) {
        if self.layer1_stride > 0 {
            let base = row.saturating_mul(self.layer1_stride);
            let target = &mut self.nn_weights[base..base + self.layer1_stride];
            target.copy_from_slice(&nn.weights[..self.layer1_stride]);
            let bias_target = &mut self.nn_biases[base..base + self.layer1_stride];
            bias_target.copy_from_slice(&nn.biases[..self.layer1_stride]);
        }
        if self.multi_stride > 0 {
            let base = row.saturating_mul(self.multi_stride);
            let target = &mut self.nn_weights_multi[base..base + self.multi_stride];
            target.copy_from_slice(&nn.weights_multi[..self.multi_stride]);
        }
        if self.layer2_weights_stride > 0 {
            let base = row.saturating_mul(self.layer2_weights_stride);
            let target = &mut self.nn_weights2[base..base + self.layer2_weights_stride];
            target.copy_from_slice(&nn.weights2[..self.layer2_weights_stride]);
        }
        if self.layer2_bias_stride > 0 {
            let base = row.saturating_mul(self.layer2_bias_stride);
            let target = &mut self.nn_biases2[base..base + self.layer2_bias_stride];
            target.copy_from_slice(&nn.biases2[..self.layer2_bias_stride]);
        }
    }
}

/// Helper that supplies deterministic RNG seeds for every tensor row.
pub struct RowRngManager {
    base_seed: u64,
    seeds: Vec<u64>,
}

impl RowRngManager {
    pub fn new(base_seed: u64) -> Self {
        RowRngManager {
            base_seed,
            seeds: Vec::new(),
        }
    }

    /// Precompute seeds for every row that will run this step.
    pub fn prepare(&mut self, row_count: usize, step: i32) {
        self.seeds.clear();
        self.seeds.reserve(row_count);
        for index in 0..row_count {
            let seed = self.compute_seed(index, step);
            self.seeds.push(seed);
        }
    }

    fn compute_seed(&self, index: usize, step: i32) -> u64 {
        let mut value = self.base_seed.wrapping_add(index as u64);
        value = value.wrapping_mul(0x9E3779B97F4A7C15);
        value = value.rotate_left(21);
        value = value.wrapping_add(step as u64);
        value
    }

    /// Retrieve the seed for the given row.
    pub fn seed_for(&self, index: usize) -> Option<u64> {
        self.seeds.get(index).copied()
    }

    /// Access the precomputed seeds so the GPU can read them in bulk.
    pub fn seeds_slice(&self) -> &[u64] {
        &self.seeds
    }

    /// Instantiate a `StdRng` seeded deterministically for the requested row.
    pub fn rng_for(&self, index: usize) -> Option<StdRng> {
        self.seed_for(index).map(StdRng::seed_from_u64)
    }
}

fn copy_tensor_to_nn(nn: &mut NeuralNetwork, tensor: &PopulationTensor, row: usize) {
    if tensor.layer1_stride > 0 {
        let base = row.saturating_mul(tensor.layer1_stride);
        let src = &tensor.nn_weights[base..base + tensor.layer1_stride];
        nn.weights[..tensor.layer1_stride].copy_from_slice(src);
        let bias_src = &tensor.nn_biases[base..base + tensor.layer1_stride];
        nn.biases[..tensor.layer1_stride].copy_from_slice(bias_src);
    }
    if tensor.multi_stride > 0 {
        let base = row.saturating_mul(tensor.multi_stride);
        let src = &tensor.nn_weights_multi[base..base + tensor.multi_stride];
        nn.weights_multi[..tensor.multi_stride].copy_from_slice(src);
    }
    if tensor.layer2_weights_stride > 0 {
        let base = row.saturating_mul(tensor.layer2_weights_stride);
        let src = &tensor.nn_weights2[base..base + tensor.layer2_weights_stride];
        nn.weights2[..tensor.layer2_weights_stride].copy_from_slice(src);
    }
    if tensor.layer2_bias_stride > 0 {
        let base = row.saturating_mul(tensor.layer2_bias_stride);
        let src = &tensor.nn_biases2[base..base + tensor.layer2_bias_stride];
        nn.biases2[..tensor.layer2_bias_stride].copy_from_slice(src);
    }
}

fn build_individual_from_row(tensor: &PopulationTensor, row: usize) -> Individual {
    let mut nn = NeuralNetwork::new(tensor.nodes, tensor.inputs, tensor.layers, tensor.nodes2);
    copy_tensor_to_nn(&mut nn, tensor, row);
    let mut ind = Individual::new(nn, tensor.positions[row], tensor.ids[row]);
    ind.resources = tensor.resources[row];
    ind.age = tensor.ages[row];
    ind
}

/// Stubbed "GPU" kernel that reads from the tensor buffers, performs the NN
/// decision (currently on the CPU), and writes the outcomes back into the
/// same tensor so `apply_to_individuals` can sync the results to the CPU
/// `Vec<Individual>` before the rest of the run uses them.
///
/// The eventual CUDA entry point is declared in `rust/cuda/tensor_kernel.cu`
/// with the following signature so the Rust side can launch it with the same
/// parameters that `run_tensor_kernel` accepts now:
///
/// extern "C" __global__ void tensor_step_kernel(
///     int active_count,
///     int env_width,
///     const double* env_resources,
///     const int32_t* occupancy,
///     bool use_occupancy,
///     double movement_cost_factor,
///     int nodes,
///     int inputs,
///     int nodes2,
///     int layers,
///     const uint64_t* row_seeds,
///     int layer1_stride,
///     int multi_stride,
///     int layer2_weights_stride,
///     int layer2_bias_stride,
///     const int32_t* ids,
///     uint64_t* positions,
///     double* resources,
///     int32_t* ages,
///     double* layer1_weights,
///     double* layer1_biases,
///     double* multi_weights,
///     double* layer2_weights,
///     double* layer2_biases,
///     double* hidden_buf,
///     double* out2_buf,
///     double* last_spend,
///     int32_t* last_delta,
///     int32_t* last_movers_out
/// );
pub fn run_tensor_kernel(
    tensor: &mut PopulationTensor,
    env_resources: &[f64],
    occupancy: &[i32],
    config: &Config,
    row_rng: &RowRngManager,
    hidden_buf: &mut [f64],
    out2_buf: &mut [f64],
    last_spend: &mut [f64],
    last_delta: &mut [i32],
    last_movers: &mut usize,
) {
    let count = tensor.active_count();
    if count == 0 {
        *last_movers = 0;
        return;
    }
    let width = env_resources.len().max(1);
    for slot in last_spend.iter_mut().take(count) {
        *slot = 0.0;
    }
    for slot in last_delta.iter_mut().take(count) {
        *slot = 0;
    }
    *last_movers = 0;

    eprintln!("[DEBUG] attempting GPU path for {} rows", count);
    if let Err(err) = run_tensor_kernel_gpu(
        tensor,
        env_resources,
        occupancy,
        config,
        row_rng,
        hidden_buf,
        out2_buf,
        last_spend,
        last_delta,
        last_movers,
        count,
        width,
    ) {
        if let TensorKernelGpuError::Cuda(cuda_err) = &err {
            eprintln!(
                "[DEBUG] raw CUDA error: {:?} ({})",
                cuda_err, cuda_err
            );
        }
        eprintln!(
            "tensor_step_kernel launch failed ({}); falling back to CPU stub",
            err
        );
        run_tensor_kernel_cpu(
            tensor,
            env_resources,
            occupancy,
            config,
            row_rng,
            hidden_buf,
            out2_buf,
            last_spend,
            last_delta,
            last_movers,
            count,
            width,
        );
    }
}

fn run_tensor_kernel_cpu(
    tensor: &mut PopulationTensor,
    env_resources: &[f64],
    occupancy: &[i32],
    config: &Config,
    row_rng: &RowRngManager,
    hidden_buf: &mut [f64],
    out2_buf: &mut [f64],
    last_spend: &mut [f64],
    last_delta: &mut [i32],
    last_movers: &mut usize,
    count: usize,
    width: usize,
) {
    let width = width.max(1);
    for row in 0..count {
        let mut ind = build_individual_from_row(tensor, row);
        let pos = ind.position % width;
        let left_idx = if pos == 0 { width - 1 } else { pos - 1 };
        let right_idx = (pos + 1) % width;
        let center = env_resources.get(pos).copied().unwrap_or(0.0);
        let left = env_resources.get(left_idx).copied().unwrap_or(center);
        let right = env_resources.get(right_idx).copied().unwrap_or(center);
        let local_population = if config.nn_use_occupancy {
            occupancy[pos]
        } else {
            0
        };
        let mut rng = row_rng
            .rng_for(row)
            .unwrap_or_else(|| StdRng::seed_from_u64(tensor.base_seed.wrapping_add(row as u64)));
        let out = ind.act(
            left,
            center,
            right,
            local_population,
            config.movement_cost_factor,
            &mut rng,
            hidden_buf,
            out2_buf,
        );
        if out != 0.0 {
            let delta = out.round() as i32;
            last_spend[row] = out.abs();
            last_delta[row] = delta;
            *last_movers = last_movers.saturating_add(1);
            let width_i64 = width as i64;
            let mut new_position = ind.position as i64 + delta as i64;
            new_position %= width_i64;
            if new_position < 0 {
                new_position += width_i64;
            }
            ind.position = new_position as usize;
        }
        tensor.resources[row] = ind.resources;
        tensor.positions[row] = ind.position;
    }
}

fn run_tensor_kernel_gpu(
    tensor: &mut PopulationTensor,
    env_resources: &[f64],
    occupancy: &[i32],
    config: &Config,
    row_rng: &RowRngManager,
    hidden_buf: &mut [f64],
    out2_buf: &mut [f64],
    last_spend: &mut [f64],
    last_delta: &mut [i32],
    last_movers: &mut usize,
    count: usize,
    width: usize,
) -> Result<(), TensorKernelGpuError> {
    let ptx_path = tensor_kernel_ptx_path()?;
    eprintln!("[DEBUG] loading PTX from {:?}", ptx_path);
    let _context = quick_init()?;
    eprintln!("[DEBUG] CUDA context initialized");
    let module = Module::from_file(&ptx_path)?;
    eprintln!("[DEBUG] module loaded");
    let function = module.get_function("tensor_step_kernel")?;
    eprintln!("[DEBUG] function resolved");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    eprintln!("[DEBUG] stream created");

    let row_seeds = row_rng.seeds_slice();
    if row_seeds.len() < count {
        return Err(TensorKernelGpuError::InsufficientSeeds {
            requested: count,
            available: row_seeds.len(),
        });
    }

    let env_buf = DeviceBuffer::from_slice(env_resources)?;
    eprintln!("[DEBUG] env buffer allocated");
    let occupancy_buf = DeviceBuffer::from_slice(occupancy)?;
    eprintln!("[DEBUG] occupancy buffer allocated");
    let row_seeds_buf = DeviceBuffer::from_slice(&row_seeds[..count])?;
    eprintln!("[DEBUG] row seeds buffer allocated");
    let ids_buf = DeviceBuffer::from_slice(&tensor.ids[..count])?;
    eprintln!("[DEBUG] ids buffer allocated");
    let mut positions_host: Vec<u64> = tensor
        .positions
        .iter()
        .take(count)
        .map(|&p| p as u64)
        .collect();
    let positions_buf = DeviceBuffer::from_slice(&positions_host)?;
    eprintln!("[DEBUG] positions buffer allocated");
    let resources_buf = DeviceBuffer::from_slice(&tensor.resources[..count])?;
    eprintln!("[DEBUG] resources buffer allocated");
    let ages_buf = DeviceBuffer::from_slice(&tensor.ages[..count])?;
    eprintln!("[DEBUG] ages buffer allocated");
    let last_spend_buf = DeviceBuffer::from_slice(&last_spend[..count])?;
    eprintln!("[DEBUG] last spend buffer allocated");
    let last_delta_buf = DeviceBuffer::from_slice(&last_delta[..count])?;
    eprintln!("[DEBUG] last delta buffer allocated");
    let last_movers_buf = DeviceBuffer::from_slice(&[0i32])?;
    eprintln!("[DEBUG] last movers buffer allocated");

    let layer1_weights_buf = optional_device_buffer(&tensor.layer1_weights_slice())?;
    let layer1_biases_buf = optional_device_buffer(&tensor.layer1_biases_slice())?;
    let multi_weights_buf = optional_device_buffer(&tensor.multi_weights_slice())?;
    let layer2_weights_buf = optional_device_buffer(&tensor.layer2_weights_slice())?;
    let layer2_biases_buf = optional_device_buffer(&tensor.layer2_biases_slice())?;
    let hidden_buf_dev = optional_device_buffer(hidden_buf)?;
    let out2_buf_dev = optional_device_buffer(out2_buf)?;

    if let Err(err) = unsafe {
        launch!(
            function<<<
                ((count as u32 + 127) / 128).max(1),
                128,
                0,
                stream
            >>>(
                usize_to_i32(count),
                usize_to_i32(width),
                env_buf.as_device_ptr(),
                occupancy_buf.as_device_ptr(),
                config.nn_use_occupancy,
                config.movement_cost_factor,
                tensor.nodes as i32,
                tensor.inputs as i32,
                tensor.nodes2 as i32,
                tensor.layers,
                row_seeds_buf.as_device_ptr(),
                usize_to_i32(tensor.layer1_stride),
                usize_to_i32(tensor.multi_stride),
                usize_to_i32(tensor.layer2_weights_stride),
                usize_to_i32(tensor.layer2_bias_stride),
                ids_buf.as_device_ptr(),
                positions_buf.as_device_ptr(),
                resources_buf.as_device_ptr(),
                ages_buf.as_device_ptr(),
                device_ptr_or_null(layer1_weights_buf.as_ref()),
                device_ptr_or_null(layer1_biases_buf.as_ref()),
                device_ptr_or_null(multi_weights_buf.as_ref()),
                device_ptr_or_null(layer2_weights_buf.as_ref()),
                device_ptr_or_null(layer2_biases_buf.as_ref()),
                device_ptr_or_null(hidden_buf_dev.as_ref()),
                device_ptr_or_null(out2_buf_dev.as_ref()),
                last_spend_buf.as_device_ptr(),
                last_delta_buf.as_device_ptr(),
                last_movers_buf.as_device_ptr()
            )
        )
    } {
        eprintln!("tensor_step_kernel launch failed inside helper: {}", err);
        return Err(err.into());
    }

    if let Err(err) = stream.synchronize() {
        eprintln!("tensor_step_kernel stream sync failed: {}", err);
        return Err(err.into());
    }

    ids_buf.copy_to(&mut tensor.ids[..count])?;
    resources_buf.copy_to(&mut tensor.resources[..count])?;
    ages_buf.copy_to(&mut tensor.ages[..count])?;
    positions_buf.copy_to(&mut positions_host)?;
    for (slot, &value) in tensor
        .positions
        .iter_mut()
        .take(count)
        .zip(positions_host.iter())
    {
        *slot = value as usize;
    }
    last_spend_buf.copy_to(&mut last_spend[..count])?;
    last_delta_buf.copy_to(&mut last_delta[..count])?;
    let mut last_movers_host = [0i32];
    last_movers_buf.copy_to(&mut last_movers_host)?;
    *last_movers = last_movers_host[0].max(0) as usize;

    if let Some(buf) = hidden_buf_dev {
        buf.copy_to(hidden_buf)?;
    }
    if let Some(buf) = out2_buf_dev {
        buf.copy_to(out2_buf)?;
    }

    Ok(())
}

#[derive(Debug)]
enum TensorKernelGpuError {
    MissingPtx,
    InsufficientSeeds { requested: usize, available: usize },
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
            TensorKernelGpuError::InsufficientSeeds {
                requested,
                available,
            } => write!(
                f,
                "{} seeds requested but only {} precomputed",
                requested, available
            ),
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

fn optional_device_buffer<T: DeviceCopy>(
    slice: &[T],
) -> cust::error::CudaResult<Option<DeviceBuffer<T>>> {
    if slice.is_empty() {
        Ok(None)
    } else {
        DeviceBuffer::from_slice(slice).map(Some)
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
