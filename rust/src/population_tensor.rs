//! CPU-side population tensor buffers shared by GPU packing and CUDA forward.

use crate::config::Config;
use crate::individual::Individual;
use crate::neural_network::NeuralNetwork;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Tensor-backed population metadata for GPU/CUDA integration.
/// Holds contiguous buffers for every per-individual field so data can be copied
/// directly without visiting `Individual` instances one by one.
pub struct PopulationTensor {
    max_population: usize,
    active_count: usize,
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

    /// Prepare metadata and MLP weights from individuals (non-MLP weights packed separately).
    pub fn sync_metadata_from_individuals(&mut self, individuals: &[Individual]) {
        let count = individuals.len().min(self.max_population);
        self.record_active(count);
        for (row, ind) in individuals.iter().take(count).enumerate() {
            self.ids[row] = ind.id;
            self.positions[row] = ind.position;
            self.resources[row] = ind.resources;
            self.ages[row] = ind.age;
            if let Some(mlp) = ind.brain.as_mlp() {
                self.copy_nn_weights(row, &mlp.nn);
            }
        }
        for row in count..self.max_population {
            self.ids[row] = 0;
            self.positions[row] = 0;
            self.resources[row] = 0.0;
            self.ages[row] = 0;
        }
    }

    /// Prepare the tensor buffers from the supplied slice of individuals.
    pub fn sync_from_individuals(&mut self, individuals: &[Individual]) {
        self.sync_metadata_from_individuals(individuals);
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
        if let Some(nn) = ind.nn_mut() {
            copy_tensor_to_nn(nn, tensor, row);
        }
    }

    fn copy_padded(dst: &mut [f64], src: &[f64]) {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        if n < dst.len() {
            dst[n..].fill(0.0);
        }
    }

    pub fn copy_nn_weights(&mut self, row: usize, nn: &NeuralNetwork) {
        if self.layer1_stride > 0 {
            let base = row.saturating_mul(self.layer1_stride);
            let target = &mut self.nn_weights[base..base + self.layer1_stride];
            Self::copy_padded(target, &nn.weights);
            let bias_target = &mut self.nn_biases[base..base + self.layer1_stride];
            Self::copy_padded(bias_target, &nn.biases);
        }
        if self.multi_stride > 0 {
            let base = row.saturating_mul(self.multi_stride);
            let target = &mut self.nn_weights_multi[base..base + self.multi_stride];
            Self::copy_padded(target, &nn.weights_multi);
        }
        if self.layer2_weights_stride > 0 {
            let base = row.saturating_mul(self.layer2_weights_stride);
            let target = &mut self.nn_weights2[base..base + self.layer2_weights_stride];
            Self::copy_padded(target, &nn.weights2);
        }
        if self.layer2_bias_stride > 0 {
            let base = row.saturating_mul(self.layer2_bias_stride);
            let target = &mut self.nn_biases2[base..base + self.layer2_bias_stride];
            Self::copy_padded(target, &nn.biases2);
        }
    }

    pub fn set_perceptron_row(
        &mut self,
        row: usize,
        nodes: usize,
        inputs: usize,
        weights: &[f64],
        biases: &[f64],
    ) {
        let nodes = nodes.min(self.nodes);
        let inputs = inputs.min(self.inputs);
        let layer1_base = row * self.layer1_stride;
        let multi_base = row * self.multi_stride;
        for i in 0..nodes {
            self.nn_biases[layer1_base + i] = biases.get(i).copied().unwrap_or(0.0);
            self.nn_weights[layer1_base + i] = 0.0;
            for j in 0..inputs {
                let idx = i * inputs + j;
                self.nn_weights_multi[multi_base + idx] = weights.get(idx).copied().unwrap_or(0.0);
            }
        }
    }

    pub fn layer1_stride(&self) -> usize {
        self.layer1_stride
    }

    pub fn multi_stride(&self) -> usize {
        self.multi_stride
    }

    pub fn layer2_weights_stride(&self) -> usize {
        self.layer2_weights_stride
    }

    pub fn layer2_bias_stride(&self) -> usize {
        self.layer2_bias_stride
    }

    pub fn positions_host_u64(&self, count: usize) -> Vec<u64> {
        self.positions
            .iter()
            .take(count)
            .map(|&p| p as u64)
            .collect()
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
