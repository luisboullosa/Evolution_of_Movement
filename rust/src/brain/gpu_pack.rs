//! GPU packing for all brain kinds and spatial runtime state.
use crate::brain::{
    BrainGenome, BrainInner, BrainKind, EMBED_DIM, GooGraphBrain, GooOp, PerceptronBrain,
    SpatialGraphBrain, SENSOR_DIM, THOUGHT_DIM,
};
use crate::config::Config;
use crate::individual::Individual;
use crate::step_engine::DecisionRow;
use crate::population_tensor::PopulationTensor;

pub const MAX_GOO_PARENTS: usize = 8;
/// Hex edge directions (matches `step_engine` / CUDA sensor layout).
pub const SPATIAL_SECTOR_COUNT: usize = 6;
pub const GOO_OP_GENERATE: u8 = 0;
pub const GOO_OP_AGGREGATE: u8 = 1;
pub const GOO_OP_SCORE: u8 = 2;
pub const GOO_OP_KEEP_BEST: u8 = 3;

pub fn brain_kind_id(kind: BrainKind) -> u8 {
    match kind {
        BrainKind::Perceptron => 0,
        BrainKind::Mlp => 1,
        BrainKind::SpatialGraph => 2,
        BrainKind::GooGraph => 3,
        BrainKind::SpatialBlob => 4,
    }
}

pub fn goo_op_id(op: GooOp) -> u8 {
    match op {
        GooOp::Generate => GOO_OP_GENERATE,
        GooOp::Aggregate => GOO_OP_AGGREGATE,
        GooOp::Score => GOO_OP_SCORE,
        GooOp::KeepBest => GOO_OP_KEEP_BEST,
    }
}

/// Extended population tensor with brain structures for GPU forward.
pub struct PopulationBrainTensor {
    pub base: PopulationTensor,
    pub max_brain_nodes: usize,
    pub max_edges: usize,
    pub max_goo_params: usize,
    pub max_readout_features: usize,

    pub brain_kind: Vec<u8>,
    pub gpu_compatible: Vec<bool>,

    pub spatial_landmark_count: Vec<i32>,
    pub spatial_edge_count: Vec<i32>,
    pub spatial_edges: Vec<i32>,
    pub spatial_edge_weights: Vec<f64>,
    pub spatial_readout_weights: Vec<f64>,
    pub spatial_readout_biases: Vec<f64>,
    pub spatial_landmark_sectors: Vec<u8>,
    pub spatial_learn_rate: Vec<f64>,
    pub spatial_message_passes: Vec<i32>,
    pub spatial_runtime: Vec<f64>,

    pub goo_node_count: Vec<i32>,
    pub goo_ops: Vec<u8>,
    pub goo_parent_count: Vec<u8>,
    pub goo_parents: Vec<i32>,
    pub goo_params: Vec<f64>,

    pub decision_out: Vec<DecisionRowGpu>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionRowGpu {
    pub move_signal: f64,
    pub direction_degrees: f64,
    pub movement_spend: f64,
    pub gathered: f64,
    pub move_one_step: i32,
}

impl From<DecisionRowGpu> for DecisionRow {
    fn from(g: DecisionRowGpu) -> Self {
        DecisionRow {
            move_one_step: g.move_one_step != 0,
            move_signal: g.move_signal,
            direction_degrees: g.direction_degrees,
            movement_spend: g.movement_spend,
            gathered: g.gathered,
        }
    }
}

impl PopulationBrainTensor {
    pub fn new(cfg: &Config) -> Self {
        let max_pop = cfg.max_population.max(cfg.initial_population).max(1) as usize;
        let max_brain_nodes = cfg.max_brain_nodes.max(1) as usize;
        let max_edges = max_brain_nodes * 4;
        let max_goo_params = max_brain_nodes * (THOUGHT_DIM * THOUGHT_DIM + THOUGHT_DIM + 8);
        let max_readout_features = max_brain_nodes + SENSOR_DIM;

        PopulationBrainTensor {
            base: PopulationTensor::new(cfg),
            max_brain_nodes,
            max_edges,
            max_goo_params,
            max_readout_features,
            brain_kind: vec![0; max_pop],
            gpu_compatible: vec![true; max_pop],
            spatial_landmark_count: vec![0; max_pop],
            spatial_edge_count: vec![0; max_pop],
            spatial_edges: vec![0; max_pop * max_edges * 3],
            spatial_edge_weights: vec![0.0; max_pop * max_edges],
            spatial_readout_weights: vec![0.0; max_pop * max_readout_features * 2],
            spatial_readout_biases: vec![0.0; max_pop * 2],
            spatial_landmark_sectors: vec![0x3F; max_pop * max_brain_nodes],
            spatial_learn_rate: vec![0.05; max_pop],
            spatial_message_passes: vec![2; max_pop],
            spatial_runtime: vec![0.0; max_pop * max_brain_nodes * EMBED_DIM],
            goo_node_count: vec![0; max_pop],
            goo_ops: vec![0; max_pop * max_brain_nodes],
            goo_parent_count: vec![0; max_pop * max_brain_nodes],
            goo_parents: vec![0; max_pop * max_brain_nodes * MAX_GOO_PARENTS],
            goo_params: vec![0.0; max_pop * max_goo_params],
            decision_out: vec![DecisionRowGpu::default(); max_pop],
        }
    }

    pub fn active_count(&self) -> usize {
        self.base.active_count()
    }

    pub fn record_active(&mut self, count: usize) {
        self.base.record_active(count);
    }

    pub fn sync_from_individuals(&mut self, individuals: &[Individual], cfg: &Config) {
        self.base.sync_metadata_from_individuals(individuals);
        let count = individuals.len().min(self.base.max_population());
        for row in 0..count {
            let ind = &individuals[row];
            let compatible = pack_row(self, row, &ind.brain, ind.spatial_runtime.as_ref(), cfg);
            self.gpu_compatible[row] = compatible;
        }
    }

    pub fn apply_spatial_runtime_to_individuals(&self, individuals: &mut [Individual]) {
        let count = self.active_count().min(individuals.len());
        let max_lm = self.max_brain_nodes;
        for row in 0..count {
            if self.brain_kind[row] != brain_kind_id(BrainKind::SpatialGraph) {
                continue;
            }
            let ind = &mut individuals[row];
            let Some(runtime) = ind.spatial_runtime.as_mut() else {
                continue;
            };
            let lm = self.spatial_landmark_count[row].max(0) as usize;
            runtime.node_embeddings.truncate(lm);
            for i in 0..lm {
                if runtime.node_embeddings.len() <= i {
                    runtime.node_embeddings.push(vec![0.0; EMBED_DIM]);
                }
                let base = row * max_lm * EMBED_DIM + i * EMBED_DIM;
                for d in 0..EMBED_DIM {
                    runtime.node_embeddings[i][d] = self.spatial_runtime[base + d];
                }
            }
        }
    }

    pub fn population_is_gpu_compatible(&self) -> bool {
        let n = self.active_count();
        self.gpu_compatible.iter().take(n).all(|&c| c)
    }
}

fn pack_row(
    tensor: &mut PopulationBrainTensor,
    row: usize,
    brain: &BrainGenome,
    spatial_runtime: Option<&crate::brain::SpatialRuntime>,
    cfg: &Config,
) -> bool {
    let _max_n = tensor.max_brain_nodes;
    let _max_e = tensor.max_edges;
    tensor.brain_kind[row] = brain_kind_id(brain.kind);

    match &brain.inner {
        BrainInner::Perceptron(p) => {
            if !pack_perceptron(tensor, row, p) {
                return false;
            }
        }
        BrainInner::Mlp(m) => {
            tensor.base.copy_nn_weights(row, &m.nn);
        }
        BrainInner::SpatialGraph(sg) => {
            if !pack_spatial(tensor, row, sg, spatial_runtime) {
                return false;
            }
        }
        BrainInner::SpatialBlob(_) => return false,
        BrainInner::GooGraph(g) => {
            if !pack_goo(tensor, row, g) {
                return false;
            }
        }
    }

    let _ = cfg;
    true
}

fn pack_perceptron(tensor: &mut PopulationBrainTensor, row: usize, p: &PerceptronBrain) -> bool {
    if p.nodes > tensor.base.nodes() || p.inputs > tensor.base.inputs() {
        return false;
    }
    tensor.base.set_perceptron_row(row, p.nodes, p.inputs, &p.weights, &p.biases);
    true
}

fn pack_spatial(
    tensor: &mut PopulationBrainTensor,
    row: usize,
    sg: &SpatialGraphBrain,
    runtime: Option<&crate::brain::SpatialRuntime>,
) -> bool {
    if sg.landmark_count > tensor.max_brain_nodes || sg.edges.len() > tensor.max_edges {
        return false;
    }
    tensor.spatial_landmark_count[row] = sg.landmark_count as i32;
    tensor.spatial_edge_count[row] = sg.edges.len() as i32;
    tensor.spatial_learn_rate[row] = sg.learn_rate;
    tensor.spatial_message_passes[row] = sg.message_passes as i32;

    let edge_base = row * tensor.max_edges * 3;
    for (i, e) in sg.edges.iter().enumerate().take(tensor.max_edges) {
        tensor.spatial_edges[edge_base + i * 3] = e.from as i32;
        tensor.spatial_edges[edge_base + i * 3 + 1] = e.to as i32;
        tensor.spatial_edges[edge_base + i * 3 + 2] = e.weight_idx as i32;
    }
    let ew_base = row * tensor.max_edges;
    for (i, &w) in sg.edge_weights.iter().enumerate().take(tensor.max_edges) {
        tensor.spatial_edge_weights[ew_base + i] = w;
    }
    let rw_cap = tensor.max_readout_features * 2;
    let rw_base = row * rw_cap;
    for (i, &w) in sg
        .readout_weights
        .iter()
        .enumerate()
        .take(rw_cap)
    {
        tensor.spatial_readout_weights[rw_base + i] = w;
    }
    let rb_base = row * 2;
    tensor.spatial_readout_biases[rb_base] = sg.readout_biases.first().copied().unwrap_or(0.0);
    tensor.spatial_readout_biases[rb_base + 1] = sg.readout_biases.get(1).copied().unwrap_or(0.0);

    let sector_base = row * tensor.max_brain_nodes;
    for i in 0..sg.landmark_count.min(tensor.max_brain_nodes) {
        let mask = sg.landmark_sectors.get(i).copied().unwrap_or(0x3F);
        tensor.spatial_landmark_sectors[sector_base + i] = if mask == 0 { 0x3F } else { mask };
    }

    let rt_base = row * tensor.max_brain_nodes * EMBED_DIM;
    if let Some(rt) = runtime {
        for (i, emb) in rt.node_embeddings.iter().enumerate().take(tensor.max_brain_nodes) {
            for (d, &v) in emb.iter().enumerate().take(EMBED_DIM) {
                tensor.spatial_runtime[rt_base + i * EMBED_DIM + d] = v;
            }
        }
    }
    true
}

fn pack_goo(tensor: &mut PopulationBrainTensor, row: usize, g: &GooGraphBrain) -> bool {
    if g.nodes.len() > tensor.max_brain_nodes || g.params.len() > tensor.max_goo_params {
        return false;
    }
    tensor.goo_node_count[row] = g.nodes.len() as i32;
    let max_n = tensor.max_brain_nodes;
    for (i, node) in g.nodes.iter().enumerate().take(max_n) {
        let idx = row * max_n + i;
        tensor.goo_ops[idx] = goo_op_id(node.op);
        let pc = node.parents.len().min(MAX_GOO_PARENTS);
        tensor.goo_parent_count[idx] = pc as u8;
        let pbase = row * max_n * MAX_GOO_PARENTS + i * MAX_GOO_PARENTS;
        for (pi, &pid) in node.parents.iter().enumerate().take(MAX_GOO_PARENTS) {
            tensor.goo_parents[pbase + pi] = pid as i32;
        }
    }
    let param_base = row * tensor.max_goo_params;
    for (i, &p) in g.params.iter().enumerate().take(tensor.max_goo_params) {
        tensor.goo_params[param_base + i] = p;
    }
    true
}

pub fn individual_fits_gpu(ind: &Individual, max_brain_nodes: usize) -> bool {
    match &ind.brain.inner {
        BrainInner::Perceptron(p) => {
            p.nodes <= max_brain_nodes && p.inputs <= SENSOR_DIM + 8
        }
        BrainInner::Mlp(m) => {
            m.nn.nodes <= max_brain_nodes && m.nn.nodes2 <= max_brain_nodes
        }
        BrainInner::SpatialGraph(sg) => {
            sg.landmark_count <= max_brain_nodes && sg.edges.len() <= max_brain_nodes * 4
        }
        BrainInner::SpatialBlob(b) => b.layer_count <= max_brain_nodes,
        BrainInner::GooGraph(g) => {
            g.nodes.len() <= max_brain_nodes
                && g.params.len() <= max_brain_nodes * (THOUGHT_DIM * THOUGHT_DIM + THOUGHT_DIM + 8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{BrainGenome, BrainInner, BrainKind, PerceptronBrain};
    use crate::config::Config;
    use crate::individual::Individual;

    #[test]
    fn pack_perceptron_brain_kind() {
        let cfg = Config::default();
        let mut pt = PopulationBrainTensor::new(&cfg);
        let brain = BrainGenome {
            kind: BrainKind::Perceptron,
            inner: BrainInner::Perceptron(PerceptronBrain::new(8, 4)),
        };
        let ind = Individual::new(brain, 0, 1);
        pt.sync_from_individuals(&[ind], &cfg);
        assert_eq!(pt.brain_kind[0], 0);
        assert!(pt.gpu_compatible[0]);
    }
}
