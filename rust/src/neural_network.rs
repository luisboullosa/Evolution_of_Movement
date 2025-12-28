use std::vec::Vec;

#[derive(Clone, Debug)]
pub enum NNType {
    SingleInput,
    MultiInput,
}

#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    pub weights: Vec<f64>,
    pub weights_multi: Vec<f64>,
    pub biases: Vec<f64>,
    pub weights2: Vec<f64>,
    pub biases2: Vec<f64>,
    pub nodes: usize,
    pub inputs: usize,
    pub nodes2: usize,
    pub layers: i32,
    pub nn_type: NNType,
}

impl NeuralNetwork {
    pub fn new(nodes: usize, inputs: usize, layers: i32, nodes2: usize) -> Self {
        let n = nodes;
        NeuralNetwork {
            weights: vec![0.0; n],
            weights_multi: vec![0.0; n.saturating_mul(inputs)],
            biases: vec![0.0; n],
            weights2: vec![0.0; n.saturating_mul(nodes2)],
            biases2: vec![0.0; nodes2],
            nodes: n,
            inputs,
            nodes2,
            layers,
            nn_type: if inputs <= 1 {
                NNType::SingleInput
            } else {
                NNType::MultiInput
            },
        }
    }

    #[inline]
    fn activate(x: f64) -> f64 {
        x.tanh()
    }

    pub fn forward_single(&self, input: f64) -> (f64, f64) {
        // compute first layer activations
        let mut hidden = vec![0.0; self.nodes];
        for (i, slot) in hidden.iter_mut().enumerate().take(self.nodes) {
            let v = self.weights.get(i).copied().unwrap_or(0.0) * input
                + self.biases.get(i).copied().unwrap_or(0.0);
            *slot = Self::activate(v);
        }

        // if two-layer network, compute second layer then sum
        if self.layers == 2 && self.nodes2 > 0 {
            let mut out2 = vec![0.0; self.nodes2];
            for (j, slot) in out2.iter_mut().enumerate().take(self.nodes2) {
                let mut acc = self.biases2.get(j).copied().unwrap_or(0.0);
                for (i, h) in hidden.iter().enumerate().take(self.nodes) {
                    let idx = i.saturating_mul(self.nodes2).saturating_add(j);
                    acc += self.weights2.get(idx).copied().unwrap_or(0.0) * h;
                }
                *slot = Self::activate(acc);
            }
            let sum: f64 = out2.iter().sum();
            (sum, 0.0)
        } else {
            let sum: f64 = hidden.iter().sum();
            (sum, 0.0)
        }
    }

    /// Forward pass using caller-provided buffers to avoid per-call allocations.
    pub fn forward_single_buf(
        &self,
        input: f64,
        hidden_buf: &mut [f64],
        out2_buf: &mut [f64],
    ) -> (f64, f64) {
        let nodes = self.nodes;
        for i in 0..nodes {
            let v = self.weights.get(i).copied().unwrap_or(0.0) * input
                + self.biases.get(i).copied().unwrap_or(0.0);
            hidden_buf[i] = Self::activate(v);
        }
        if self.layers == 2 && self.nodes2 > 0 {
            let nodes2 = self.nodes2;
            for j in 0..nodes2 {
                let mut acc = self.biases2.get(j).copied().unwrap_or(0.0);
                for i in 0..nodes {
                    let idx = i.saturating_mul(nodes2).saturating_add(j);
                    acc += self.weights2.get(idx).copied().unwrap_or(0.0) * hidden_buf[i];
                }
                out2_buf[j] = Self::activate(acc);
            }
            let mut sum = 0.0f64;
            for j in 0..self.nodes2 {
                sum += out2_buf[j];
            }
            (sum, 0.0)
        } else {
            let mut sum = 0.0f64;
            for i in 0..nodes {
                sum += hidden_buf[i];
            }
            (sum, 0.0)
        }
    }

    pub fn forward_multi(&self, input: &[f64]) -> (f64, f64) {
        // compute first layer activations using weights_multi
        let mut hidden = vec![0.0; self.nodes];
        for (i, slot) in hidden.iter_mut().enumerate().take(self.nodes) {
            let mut acc = self.biases.get(i).copied().unwrap_or(0.0);
            for (j, val) in input.iter().enumerate().take(self.inputs.min(input.len())) {
                let idx = i.saturating_mul(self.inputs).saturating_add(j);
                acc += self.weights_multi.get(idx).copied().unwrap_or(0.0) * (*val);
            }
            *slot = Self::activate(acc);
        }

        if self.layers == 2 && self.nodes2 > 0 {
            let mut out2 = vec![0.0; self.nodes2];
            for (j, slot) in out2.iter_mut().enumerate().take(self.nodes2) {
                let mut acc = self.biases2.get(j).copied().unwrap_or(0.0);
                for (i, h) in hidden.iter().enumerate().take(self.nodes) {
                    let idx = i.saturating_mul(self.nodes2).saturating_add(j);
                    acc += self.weights2.get(idx).copied().unwrap_or(0.0) * h;
                }
                *slot = Self::activate(acc);
            }
            let sum: f64 = out2.iter().sum();
            (sum, 0.0)
        } else {
            let sum: f64 = hidden.iter().sum();
            (sum, 0.0)
        }
    }

    /// Forward pass for multi-input nets using caller-provided buffers to avoid allocations.
    pub fn forward_multi_buf(
        &self,
        input: &[f64],
        hidden_buf: &mut [f64],
        out2_buf: &mut [f64],
    ) -> (f64, f64) {
        let nodes = self.nodes;
        let in_len = input.len();
        for i in 0..nodes {
            let mut acc = self.biases.get(i).copied().unwrap_or(0.0);
            let read_len = self.inputs.min(in_len);
            for j in 0..read_len {
                let idx = i.saturating_mul(self.inputs).saturating_add(j);
                acc += self.weights_multi.get(idx).copied().unwrap_or(0.0) * input[j];
            }
            hidden_buf[i] = Self::activate(acc);
        }
        if self.layers == 2 && self.nodes2 > 0 {
            let nodes2 = self.nodes2;
            for j in 0..nodes2 {
                let mut acc = self.biases2.get(j).copied().unwrap_or(0.0);
                for i in 0..nodes {
                    let idx = i.saturating_mul(nodes2).saturating_add(j);
                    acc += self.weights2.get(idx).copied().unwrap_or(0.0) * hidden_buf[i];
                }
                out2_buf[j] = Self::activate(acc);
            }
            let mut sum = 0.0f64;
            for j in 0..nodes2 {
                sum += out2_buf[j];
            }
            (sum, 0.0)
        } else {
            let mut sum = 0.0f64;
            for i in 0..nodes {
                sum += hidden_buf[i];
            }
            (sum, 0.0)
        }
    }

    /// Return a small preview of weights useful for quick debugging.
    pub fn weights_preview(&self, max: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for &w in self.weights.iter().take(max) {
            out.push(w);
        }
        for &w in self.weights_multi.iter().take(max) {
            out.push(w);
        }
        for &w in self.weights2.iter().take(max) {
            out.push(w);
        }
        out
    }

    /// Return the network shape: (nodes, inputs, nodes2, layers)
    pub fn shape(&self) -> (usize, usize, usize, i32) {
        (self.nodes, self.inputs, self.nodes2, self.layers)
    }
}
