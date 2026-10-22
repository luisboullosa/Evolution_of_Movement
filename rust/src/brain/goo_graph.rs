use super::types::{BrainOutput, SENSOR_DIM, THOUGHT_DIM, l2_slices, perturb_param};
use super::movement_output::{hidden_sum_output, split_channels_output};
use crate::config::MovementOutputMode;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GooOp {
    Generate,
    Aggregate,
    Score,
    KeepBest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GooNode {
    pub id: usize,
    pub op: GooOp,
    pub parents: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GooGraphBrain {
    pub nodes: Vec<GooNode>,
    pub params: Vec<f64>,
    pub thought_dim: usize,
}

impl GooGraphBrain {
    pub fn new_initial() -> Self {
        GooGraphBrain {
            nodes: vec![GooNode {
                id: 0,
                op: GooOp::Generate,
                parents: vec![],
            }],
            params: vec![0.0; THOUGHT_DIM * THOUGHT_DIM + THOUGHT_DIM],
            thought_dim: THOUGHT_DIM,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn complexity(&self) -> usize {
        self.nodes.len()
            + self
                .nodes
                .iter()
                .map(|n| n.parents.len())
                .sum::<usize>()
            + self.params.len() / self.thought_dim.max(1)
    }

    pub fn structure_signature(&self) -> String {
        let ops: Vec<String> = self
            .nodes
            .iter()
            .map(|n| format!("{:?}", n.op))
            .collect();
        format!("goo:{}:[{}]", self.nodes.len(), ops.join(","))
    }

    fn init_thought(sensors: &[f64; SENSOR_DIM]) -> Vec<f64> {
        let mut t = vec![0.0; THOUGHT_DIM];
        for i in 0..THOUGHT_DIM.min(SENSOR_DIM) {
            t[i] = sensors[i];
        }
        t
    }

    fn matvec(params: &[f64], dim: usize, input: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; dim];
        for i in 0..dim {
            let mut acc = params.get(dim * dim + i).copied().unwrap_or(0.0);
            for j in 0..dim {
                acc += params.get(i * dim + j).copied().unwrap_or(0.0) * input.get(j).copied().unwrap_or(0.0);
            }
            out[i] = acc.tanh();
        }
        out
    }

    fn topo_order(nodes: &[GooNode]) -> Vec<usize> {
        let mut indegree: HashMap<usize, usize> = HashMap::new();
        for n in nodes {
            indegree.entry(n.id).or_insert(0);
            for &p in &n.parents {
                *indegree.entry(n.id).or_default() += 1;
                indegree.entry(p).or_insert(0);
            }
        }
        let mut q: VecDeque<usize> = nodes
            .iter()
            .filter(|n| indegree.get(&n.id).copied().unwrap_or(0) == 0)
            .map(|n| n.id)
            .collect();
        let mut order = Vec::new();
        while let Some(id) = q.pop_front() {
            order.push(id);
            for n in nodes {
                if n.parents.contains(&id) {
                    let deg = indegree.get_mut(&n.id).unwrap();
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        q.push_back(n.id);
                    }
                }
            }
        }
        if order.len() != nodes.len() {
            return nodes.iter().map(|n| n.id).collect();
        }
        order
    }

    pub fn forward(
        &self,
        sensors: &[f64; SENSOR_DIM],
        _hidden_buf: &mut [f64],
        output_mode: MovementOutputMode,
    ) -> BrainOutput {
        let dim = self.thought_dim;
        let base = Self::init_thought(sensors);
        let mut values: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for id in Self::topo_order(&self.nodes) {
            let node = self.nodes.iter().find(|n| n.id == id).unwrap();
            match node.op {
                GooOp::Generate => {
                    let offset = (id % (self.params.len().saturating_sub(dim * dim + dim) + 1)).min(self.params.len().saturating_sub(1));
                    let slice = &self.params[offset..];
                    values.insert(id, Self::matvec(slice, dim, &base));
                }
                GooOp::Aggregate => {
                    let mut concat = vec![0.0; dim * node.parents.len().max(1)];
                    for (pi, &pid) in node.parents.iter().enumerate() {
                        let parent = values.get(&pid).cloned().unwrap_or_else(|| base.clone());
                        for j in 0..dim {
                            concat[pi * dim + j] = parent.get(j).copied().unwrap_or(0.0);
                        }
                    }
                    let mut agg = vec![0.0; dim];
                    for j in 0..dim {
                        let mut s = 0.0;
                        for pi in 0..node.parents.len().max(1) {
                            s += concat.get(pi * dim + j).copied().unwrap_or(0.0);
                        }
                        agg[j] = (s / node.parents.len().max(1) as f64).tanh();
                    }
                    values.insert(id, agg);
                }
                GooOp::Score => {
                    let input = node
                        .parents
                        .first()
                        .and_then(|p| values.get(p))
                        .cloned()
                        .unwrap_or_else(|| base.clone());
                    let score: f64 = input.iter().enumerate().map(|(i, v)| {
                        v * self.params.get(i % self.params.len()).copied().unwrap_or(0.0)
                    }).sum();
                    scores.insert(id, score);
                    values.insert(id, input);
                }
                GooOp::KeepBest => {
                    let mut best_parent = node.parents.first().copied().unwrap_or(0);
                    let mut best_score = f64::NEG_INFINITY;
                    for (rank, &pid) in node.parents.iter().enumerate() {
                        let sc = scores.get(&pid).copied().unwrap_or_else(|| {
                            values
                                .get(&pid)
                                .map(|v| v.iter().sum())
                                .unwrap_or(0.0)
                        });
                        if sc > best_score || (sc == best_score && pid < best_parent) {
                            best_score = sc;
                            best_parent = pid;
                        }
                        let _ = rank;
                    }
                    values.insert(
                        id,
                        values.get(&best_parent).cloned().unwrap_or_else(|| base.clone()),
                    );
                }
            }
        }

        let last_id = self.nodes.last().map(|n| n.id).unwrap_or(0);
        let final_thought = values.get(&last_id).cloned().unwrap_or(base);
        match output_mode {
            MovementOutputMode::SplitChannels => split_channels_output(&final_thought),
            _ => hidden_sum_output(&final_thought),
        }
    }

    pub fn flatten_params(&self) -> Vec<f64> {
        self.params.clone()
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let struct_penalty = if self.nodes.len() != other.nodes.len() { 0.3 } else { 0.0 };
        struct_penalty + l2_slices(&self.flatten_params(), &other.flatten_params())
    }

    pub fn add_generate_node(&mut self, rng: &mut impl Rng, parent_id: Option<usize>) {
        self.add_generate_node_ext(rng, parent_id, false, 0.1);
    }

    pub fn duplicate_generate_node(&mut self, rng: &mut impl Rng, parent_id: Option<usize>, sigma: f64) {
        self.add_generate_node_ext(rng, parent_id, true, sigma);
    }

    fn add_generate_node_ext(
        &mut self,
        rng: &mut impl Rng,
        parent_id: Option<usize>,
        duplicate: bool,
        sigma: f64,
    ) {
        let id = self.nodes.len();
        let mut parents = Vec::new();
        if let Some(p) = parent_id {
            parents.push(p);
        }
        self.nodes.push(GooNode {
            id,
            op: GooOp::Generate,
            parents,
        });
        let block = THOUGHT_DIM * THOUGHT_DIM + THOUGHT_DIM;
        if duplicate {
            if let Some(p) = parent_id {
                let max_offset = self.params.len().saturating_sub(block);
                let offset = (p % (max_offset + 1)).min(max_offset);
                let template: Vec<f64> = self.params[offset..offset + block].to_vec();
                self.params
                    .extend(template.iter().map(|v| perturb_param(*v, rng, sigma)));
                return;
            }
        }
        self.params
            .extend((0..block).map(|_| rng.gen::<f64>() * 0.1));
    }

    /// Pick `count` distinct parent node ids from `0..n` (for branch/merge mutations).
    pub fn sample_parent_ids(rng: &mut impl Rng, n: usize, count: usize) -> Vec<usize> {
        if n == 0 || count == 0 {
            return Vec::new();
        }
        let take = count.min(n);
        let mut ids: Vec<usize> = (0..n).collect();
        for i in 0..take {
            let j = rng.gen_range(i..n);
            ids.swap(i, j);
        }
        let mut out = ids[..take].to_vec();
        out.sort_unstable();
        out
    }

    pub fn add_aggregate_node(&mut self, rng: &mut impl Rng, parents: Vec<usize>) {
        if parents.len() < 2 {
            return;
        }
        let id = self.nodes.len();
        self.nodes.push(GooNode {
            id,
            op: GooOp::Aggregate,
            parents,
        });
        let _ = rng;
    }

    pub fn add_score_node(&mut self, parent_id: usize) {
        let id = self.nodes.len();
        self.nodes.push(GooNode {
            id,
            op: GooOp::Score,
            parents: vec![parent_id],
        });
    }

    pub fn add_keep_best_node(&mut self, parents: Vec<usize>) {
        if parents.len() < 2 {
            return;
        }
        let id = self.nodes.len();
        self.nodes.push(GooNode {
            id,
            op: GooOp::KeepBest,
            parents,
        });
    }

    pub fn change_op_type(&mut self, node_id: usize, rng: &mut impl Rng) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == node_id) {
            node.op = match node.op {
                GooOp::Generate => GooOp::Score,
                GooOp::Score => GooOp::Aggregate,
                GooOp::Aggregate => GooOp::KeepBest,
                GooOp::KeepBest => GooOp::Generate,
            };
            let _ = rng;
        }
    }

    /// Grow the graph to `target_nodes` using the same structural ops as reproduction mutations.
    pub fn grow_to_size(&mut self, rng: &mut impl Rng, target_nodes: usize) {
        while self.nodes.len() < target_nodes.max(1) {
            let n = self.nodes.len();
            match rng.gen_range(0..5) {
                0 => {
                    let parent_id = if n > 0 {
                        Some(rng.gen_range(0..n))
                    } else {
                        None
                    };
                    self.add_generate_node(rng, parent_id);
                }
                1 if n >= 2 => {
                    let parents = Self::sample_parent_ids(rng, n, 2);
                    if parents.len() >= 2 {
                        self.add_aggregate_node(rng, parents);
                    }
                }
                2 if n >= 1 => {
                    self.add_score_node(rng.gen_range(0..n));
                }
                3 if n >= 2 => {
                    let parent_count = rng.gen_range(2..=n.min(4));
                    let parents = Self::sample_parent_ids(rng, n, parent_count);
                    if parents.len() >= 2 {
                        self.add_keep_best_node(parents);
                    }
                }
                _ => {
                    if n > 0 {
                        self.change_op_type(rng.gen_range(0..n), rng);
                    } else {
                        self.add_generate_node(rng, None);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn grow_to_size_can_produce_mixed_ops() {
        let mut goo = GooGraphBrain::new_initial();
        let mut rng = StdRng::seed_from_u64(99);
        goo.grow_to_size(&mut rng, 8);
        let ops: Vec<GooOp> = goo.nodes.iter().map(|n| n.op).collect();
        assert!(
            ops.iter().any(|o| *o != GooOp::Generate),
            "expected non-Generate ops, got {ops:?}"
        );
    }
}
