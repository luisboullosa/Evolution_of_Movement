//! Intracellular "macromolecule blob" spatial brain: layered nodes tuned to hex edges
//! or cardinal (N/S/E/W) pools of neighbor resources.
use super::types::{BrainOutput, SENSOR_DIM, l2_slices, perturb_param};
use super::movement_output::{
    hidden_sum_output, hex_compass_output, readout_output, CARDINAL_ANGLES_DEG,
    HEX_SECTOR_ANGLES_DEG,
};
use super::spatial_graph::SPATIAL_SECTOR_COUNT;
use crate::config::MovementOutputMode;
use rand::Rng;
use rand::distributions::Distribution;
use serde::{Deserialize, Serialize};

/// Flat-top hex neighbor order (matches `HexCoord::neighbors`).
pub const SECTOR_E: usize = 0;
pub const SECTOR_NE: usize = 1;
pub const SECTOR_NW: usize = 2;
pub const SECTOR_W: usize = 3;
pub const SECTOR_SW: usize = 4;
pub const SECTOR_SE: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialBlobLayout {
    /// One spatial node per hex edge per layer (6 nodes/layer).
    HexEdge,
    /// Four cardinal macromolecule pools per layer (N/S/E/W).
    Cardinal4,
}

impl SpatialBlobLayout {
    pub fn nodes_per_layer(self) -> usize {
        match self {
            SpatialBlobLayout::HexEdge => SPATIAL_SECTOR_COUNT,
            SpatialBlobLayout::Cardinal4 => 4,
        }
    }

    pub fn from_config_str(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cardinal" | "cardinal_4" | "cardinal4" | "nsew" => SpatialBlobLayout::Cardinal4,
            _ => SpatialBlobLayout::HexEdge,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SpatialBlobLayout::HexEdge => "hex_edge",
            SpatialBlobLayout::Cardinal4 => "cardinal_4",
        }
    }
}

/// Eight compass tokens derived from six hex edge sensors (N/S are bisectors).
#[derive(Debug, Clone, Copy)]
pub struct Compass8 {
    pub n: f64,
    pub ne: f64,
    pub nw: f64,
    pub e: f64,
    pub se: f64,
    pub sw: f64,
    pub w: f64,
    pub s: f64,
}

pub fn hex_edges_from_sensors(sensors: &[f64; SENSOR_DIM]) -> [f64; SPATIAL_SECTOR_COUNT] {
    let mut e = [0.0; SPATIAL_SECTOR_COUNT];
    for i in 0..SPATIAL_SECTOR_COUNT {
        e[i] = sensors[i];
    }
    e
}

pub fn hex_to_compass8(edges: &[f64; SPATIAL_SECTOR_COUNT]) -> Compass8 {
    Compass8 {
        ne: edges[SECTOR_NE],
        nw: edges[SECTOR_NW],
        e: edges[SECTOR_E],
        se: edges[SECTOR_SE],
        sw: edges[SECTOR_SW],
        w: edges[SECTOR_W],
        n: (edges[SECTOR_NE] + edges[SECTOR_NW]) * 0.5,
        s: (edges[SECTOR_SW] + edges[SECTOR_SE]) * 0.5,
    }
}

/// Fixed cardinal pools: Ni+NWi+NEi→N, etc. (user recipe on compass tokens).
pub fn cardinal_pool_inputs(c: Compass8) -> [f64; 4] {
    [
        c.n + c.nw + c.ne,
        c.s + c.sw + c.se,
        c.w + c.nw + c.sw,
        c.e + c.ne + c.se,
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialBlobBrain {
    pub layout: SpatialBlobLayout,
    /// Number of spatial macromolecule layers (≥1).
    pub layer_count: usize,
    /// Layer 0: `nodes × 6` weights from hex edge / cardinal inputs.
    pub layer0_input_weights: Vec<f64>,
    /// Layers 1+: flat `layer_count-1` blocks of `nodes × nodes`.
    pub layer_recurrent_weights: Vec<f64>,
    /// `layer_count × nodes` biases.
    pub layer_biases: Vec<f64>,
    pub readout_weights: Vec<f64>,
    pub readout_biases: Vec<f64>,
}

impl SpatialBlobBrain {
    pub fn new(layout: SpatialBlobLayout, layer_count: usize) -> Self {
        let layer_count = layer_count.max(1);
        let nodes = layout.nodes_per_layer();
        SpatialBlobBrain {
            layout,
            layer_count,
            layer0_input_weights: vec![0.0; nodes * SPATIAL_SECTOR_COUNT],
            layer_recurrent_weights: vec![0.0; layer_count.saturating_sub(1) * nodes * nodes],
            layer_biases: vec![0.0; layer_count * nodes],
            readout_weights: vec![0.0; nodes * 2],
            readout_biases: vec![0.0; 2],
        }
    }

    pub fn nodes_per_layer(&self) -> usize {
        self.layout.nodes_per_layer()
    }

    pub fn complexity(&self) -> usize {
        // Metabolic cost scales with spatial nodes/layers, not dense weight parameter count.
        let nodes = self.nodes_per_layer();
        self.layer_count * nodes + self.layer_recurrent_weights.len() / nodes.max(1)
    }

    pub fn structure_signature(&self) -> String {
        format!(
            "blob:{}:{}:{}",
            self.layout.as_str(),
            self.layer_count,
            self.nodes_per_layer()
        )
    }

    pub fn flatten_params(&self) -> Vec<f64> {
        let mut out = self.layer0_input_weights.clone();
        out.extend(&self.layer_recurrent_weights);
        out.extend(&self.layer_biases);
        out.extend(&self.readout_weights);
        out.extend(&self.readout_biases);
        out
    }

    /// Structured layer wiring; readouts stay zero (gather-first founders).
    pub fn init_founder_layers(&mut self, rng: &mut impl Rng) {
        let nodes = self.nodes_per_layer();
        let wrand = rand::distributions::Uniform::new(-0.15, 0.15);
        for i in 0..nodes {
            for j in 0..SPATIAL_SECTOR_COUNT {
                let idx = i * SPATIAL_SECTOR_COUNT + j;
                let mut w = wrand.sample(rng);
                if i == j % nodes {
                    w += if rng.gen_bool(0.5) { 0.25 } else { -0.25 };
                }
                self.layer0_input_weights[idx] = w;
            }
        }
        for w in self.layer_recurrent_weights.iter_mut() {
            *w = wrand.sample(rng) * 0.5;
        }
        for b in self.layer_biases.iter_mut() {
            *b = wrand.sample(rng) * 0.05;
        }
        self.readout_weights.fill(0.0);
        self.readout_biases.fill(0.0);
    }

    /// Small structured founder: diagonal-ish edge→node wiring, not a flat perceptron.
    pub fn init_founder_weights(&mut self, rng: &mut impl Rng) {
        self.init_founder_layers(rng);
        let wrand = rand::distributions::Uniform::new(-0.15, 0.15);
        for w in self.readout_weights.iter_mut() {
            *w = wrand.sample(rng) * 0.1;
        }
    }

    fn layer0_inputs(&self, sensors: &[f64; SENSOR_DIM]) -> Vec<f64> {
        let edges = hex_edges_from_sensors(sensors);
        match self.layout {
            SpatialBlobLayout::HexEdge => edges.to_vec(),
            SpatialBlobLayout::Cardinal4 => cardinal_pool_inputs(hex_to_compass8(&edges)).to_vec(),
        }
    }

    fn activate_layer(
        &self,
        layer_idx: usize,
        prev: &[f64],
        layer0_inputs: &[f64],
    ) -> Vec<f64> {
        let nodes = self.nodes_per_layer();
        let mut out = vec![0.0; nodes];
        let bias_base = layer_idx * nodes;

        if layer_idx == 0 {
            let in_dim = layer0_inputs.len();
            for i in 0..nodes {
                let mut acc = self.layer_biases.get(bias_base + i).copied().unwrap_or(0.0);
                for j in 0..in_dim.min(SPATIAL_SECTOR_COUNT) {
                    let w = self
                        .layer0_input_weights
                        .get(i * SPATIAL_SECTOR_COUNT + j)
                        .copied()
                        .unwrap_or(0.0);
                    acc += w * layer0_inputs[j];
                }
                if self.layout == SpatialBlobLayout::Cardinal4 {
                    for j in 0..nodes.min(layer0_inputs.len()) {
                        if j != i {
                            let w = self
                                .layer0_input_weights
                                .get(i * SPATIAL_SECTOR_COUNT + j)
                                .copied()
                                .unwrap_or(0.0);
                            acc += w * layer0_inputs[j];
                        }
                    }
                }
                out[i] = acc.tanh();
            }
        } else {
            let block = layer_idx - 1;
            let w_base = block * nodes * nodes;
            for i in 0..nodes {
                let mut acc = self.layer_biases.get(bias_base + i).copied().unwrap_or(0.0);
                for j in 0..nodes {
                    let w = self
                        .layer_recurrent_weights
                        .get(w_base + i * nodes + j)
                        .copied()
                        .unwrap_or(0.0);
                    acc += w * prev[j];
                }
                out[i] = acc.tanh();
            }
        }
        out
    }

    pub fn forward(
        &self,
        sensors: &[f64; SENSOR_DIM],
        hidden_buf: &mut [f64],
        output_mode: MovementOutputMode,
    ) -> BrainOutput {
        let layer0 = self.layer0_inputs(sensors);
        let mut activations = self.activate_layer(0, &[], &layer0);
        for layer in 1..self.layer_count {
            activations = self.activate_layer(layer, &activations, &layer0);
        }

        for (i, &a) in activations.iter().enumerate() {
            if i < hidden_buf.len() {
                hidden_buf[i] = a;
            }
        }

        if output_mode == MovementOutputMode::HiddenSum {
            return hidden_sum_output(&activations);
        }

        if output_mode == MovementOutputMode::HexCompass {
            let angles = match self.layout {
                SpatialBlobLayout::HexEdge => &HEX_SECTOR_ANGLES_DEG[..],
                SpatialBlobLayout::Cardinal4 => &CARDINAL_ANGLES_DEG[..],
            };
            return hex_compass_output(&activations, angles);
        }

        let mut move_raw = self.readout_biases.first().copied().unwrap_or(0.0);
        let mut direction_raw = self.readout_biases.get(1).copied().unwrap_or(0.0);
        for (i, &a) in activations.iter().enumerate() {
            let wi = i * 2;
            move_raw += self.readout_weights.get(wi).copied().unwrap_or(0.0) * a;
            direction_raw += self
                .readout_weights
                .get(wi + 1)
                .copied()
                .unwrap_or(0.0)
                * a;
        }

        readout_output(move_raw, direction_raw, output_mode)
    }

    pub fn add_layer(&mut self, rng: &mut impl Rng, duplicate: bool, sigma: f64) {
        let nodes = self.nodes_per_layer();
        let new_idx = self.layer_count;
        self.layer_count += 1;
        self.layer_biases.resize(self.layer_count * nodes, 0.0);

        if duplicate && new_idx > 0 {
            let src = new_idx - 1;
            let src_bias = src * nodes;
            let dst_bias = new_idx * nodes;
            for i in 0..nodes {
                self.layer_biases[dst_bias + i] =
                    perturb_param(self.layer_biases[src_bias + i], rng, sigma);
            }
            let block = new_idx - 1;
            let w_base = block * nodes * nodes;
            let block_len = nodes * nodes;
            let mut new_block = vec![0.0; block_len];
            if w_base + block_len <= self.layer_recurrent_weights.len() {
                for i in 0..block_len {
                    new_block[i] = perturb_param(
                        self.layer_recurrent_weights[w_base + i],
                        rng,
                        sigma,
                    );
                }
            } else {
                // First recurrent block: mirror layer-0 input wiring (no prior block to copy).
                for i in 0..nodes {
                    for j in 0..nodes {
                        let in_idx = i * SPATIAL_SECTOR_COUNT + (j % SPATIAL_SECTOR_COUNT);
                        let w = self.layer0_input_weights.get(in_idx).copied().unwrap_or(0.0);
                        new_block[i * nodes + j] = perturb_param(w, rng, sigma);
                    }
                }
            }
            self.layer_recurrent_weights.extend(new_block);
        } else {
            let wrand = rand::distributions::Uniform::new(-0.12, 0.12);
            for i in 0..nodes {
                self.layer_biases[new_idx * nodes + i] = wrand.sample(rng);
            }
            for _ in 0..(nodes * nodes) {
                self.layer_recurrent_weights.push(wrand.sample(rng));
            }
        }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let struct_diff = if self.layout != other.layout || self.layer_count != other.layer_count {
            0.5
        } else {
            0.0
        };
        let mut a = self.layer0_input_weights.clone();
        a.extend(&self.layer_recurrent_weights);
        a.extend(&self.layer_biases);
        a.extend(&self.readout_weights);
        a.extend(&self.readout_biases);
        let mut b = other.layer0_input_weights.clone();
        b.extend(&other.layer_recurrent_weights);
        b.extend(&other.layer_biases);
        b.extend(&other.readout_weights);
        b.extend(&other.readout_biases);
        struct_diff + l2_slices(&a, &b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MovementOutputMode;

    #[test]
    fn cardinal_pools_match_compass_recipe() {
        let edges = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = hex_to_compass8(&edges);
        let pools = cardinal_pool_inputs(c);
        assert!((pools[0] - (c.n + c.nw + c.ne)).abs() < 1e-9);
        assert!((pools[1] - (c.s + c.sw + c.se)).abs() < 1e-9);
        assert!((pools[2] - (c.w + c.nw + c.sw)).abs() < 1e-9);
        assert!((pools[3] - (c.e + c.ne + c.se)).abs() < 1e-9);
    }

    #[test]
    fn blob_forward_runs() {
        let mut brain = SpatialBlobBrain::new(SpatialBlobLayout::HexEdge, 2);
        brain.init_founder_weights(&mut rand::thread_rng());
        let sensors = [1.0, 0.5, 0.2, 0.1, 0.0, 0.3, 2.0, 10.0];
        let mut hidden = [0.0; 8];
        let out = brain.forward(&sensors, &mut hidden, MovementOutputMode::SeparateReadout);
        assert!(out.move_signal.is_finite());
        assert!(out.direction_signal.is_finite());
    }
}
