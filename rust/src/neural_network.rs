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
            weights_multi: vec![0.0; n*inputs],
            biases: vec![0.0; n],
            weights2: vec![0.0; n * nodes2],
            biases2: vec![0.0; nodes2],
            nodes: n,
            inputs,
            nodes2,
            layers,
            nn_type: if inputs <= 1 { NNType::SingleInput } else { NNType::MultiInput },
        }
    }

    pub fn forward_single(&self, input: f64) -> (f64, f64) {
        // placeholder: sum weight*input + bias
        let mut s = 0.0;
        for i in 0..self.nodes {
            s += self.weights[i] * input + self.biases[i];
        }
        (s, 0.0)
    }

    pub fn forward_multi(&self, input: &[f64]) -> (f64, f64) {
        // placeholder: compute simple dot-product per node then sum
        let mut s = 0.0;
        for i in 0..self.nodes {
            let mut node = 0.0;
            for j in 0..self.inputs {
                let idx = i * self.inputs + j;
                node += self.weights_multi.get(idx).copied().unwrap_or(0.0) * input[j];
            }
            node += self.biases[i];
            s += node;
        }
        (s, 0.0)
    }
}
