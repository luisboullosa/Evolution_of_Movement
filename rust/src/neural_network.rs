use std::vec::Vec;

#[derive(Clone, Debug)]
pub enum NNType { SingleInput, MultiInput }

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
            nn_type: if inputs <= 1 { NNType::SingleInput } else { NNType::MultiInput },
        }
    }

    #[inline]
    fn activate(x: f64) -> f64 { x.tanh() }

    pub fn forward_single(&self, input: f64) -> (f64, f64) {
        // compute first layer activations
        let mut hidden = vec![0.0; self.nodes];
        for i in 0..self.nodes {
            let v = self.weights.get(i).copied().unwrap_or(0.0) * input + self.biases.get(i).copied().unwrap_or(0.0);
            hidden[i] = Self::activate(v);
        }

        // if two-layer network, compute second layer then sum
        if self.layers == 2 && self.nodes2 > 0 {
            let mut out2 = vec![0.0; self.nodes2];
            for j in 0..self.nodes2 {
                let mut acc = self.biases2.get(j).copied().unwrap_or(0.0);
                for i in 0..self.nodes {
                    let idx = i.saturating_mul(self.nodes2).saturating_add(j);
                    acc += self.weights2.get(idx).copied().unwrap_or(0.0) * hidden[i];
                }
                out2[j] = Self::activate(acc);
            }
            let sum: f64 = out2.iter().sum();
            (sum, 0.0)
        } else {
            let sum: f64 = hidden.iter().sum();
            (sum, 0.0)
        }
    }

    pub fn forward_multi(&self, input: &[f64]) -> (f64, f64) {
        // compute first layer activations using weights_multi
        let mut hidden = vec![0.0; self.nodes];
        for i in 0..self.nodes {
            let mut acc = self.biases.get(i).copied().unwrap_or(0.0);
            for j in 0..self.inputs.min(input.len()) {
                let idx = i.saturating_mul(self.inputs).saturating_add(j);
                acc += self.weights_multi.get(idx).copied().unwrap_or(0.0) * input[j];
            }
            hidden[i] = Self::activate(acc);
        }

        if self.layers == 2 && self.nodes2 > 0 {
            let mut out2 = vec![0.0; self.nodes2];
            for j in 0..self.nodes2 {
                let mut acc = self.biases2.get(j).copied().unwrap_or(0.0);
                for i in 0..self.nodes {
                    let idx = i.saturating_mul(self.nodes2).saturating_add(j);
                    acc += self.weights2.get(idx).copied().unwrap_or(0.0) * hidden[i];
                }
                out2[j] = Self::activate(acc);
            }
            let sum: f64 = out2.iter().sum();
            (sum, 0.0)
        } else {
            let sum: f64 = hidden.iter().sum();
            (sum, 0.0)
        }
    }
}
