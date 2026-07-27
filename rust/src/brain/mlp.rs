use super::genome::{BrainOutput, SENSOR_DIM};
use super::perceptron::{l2_slices, perturb_param};
use crate::neural_network::{NeuralNetwork, NNType};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpBrainSerde {
    pub weights: Vec<f64>,
    pub weights_multi: Vec<f64>,
    pub biases: Vec<f64>,
    pub weights2: Vec<f64>,
    pub biases2: Vec<f64>,
    pub nodes: usize,
    pub inputs: usize,
    pub nodes2: usize,
    pub layers: i32,
}

#[derive(Debug, Clone)]
pub struct MlpBrain {
    pub nn: NeuralNetwork,
}

impl MlpBrain {
    pub fn from_config(nodes: usize, inputs: usize, layers: i32, nodes2: usize) -> Self {
        MlpBrain {
            nn: NeuralNetwork::new(nodes, inputs.max(SENSOR_DIM), layers, nodes2),
        }
    }

    pub fn from_perceptron(inputs: usize, nodes: usize, weights: Vec<f64>, biases: Vec<f64>) -> Self {
        let mut nn = NeuralNetwork::new(nodes, inputs, 1, 0);
        nn.nn_type = NNType::MultiInput;
        nn.weights_multi = weights;
        nn.weights = vec![0.0; nodes];
        nn.biases = biases;
        MlpBrain { nn }
    }

    pub fn complexity(&self) -> usize {
        self.nn.nodes + self.nn.nodes2 + 1
    }

    pub fn forward(
        &self,
        sensors: &[f64; SENSOR_DIM],
        hidden_buf: &mut [f64],
        out2_buf: &mut [f64],
    ) -> BrainOutput {
        let (move_signal, direction_signal) = if self.nn.inputs <= 1 {
            let center = sensors.iter().sum::<f64>() / SENSOR_DIM as f64;
            self.nn.forward_single_buf(center, hidden_buf, out2_buf)
        } else {
            self.nn.forward_multi_buf(sensors, hidden_buf, out2_buf)
        };
        BrainOutput {
            move_signal,
            direction_signal,
        }
    }

    pub fn flatten_params(&self) -> Vec<f64> {
        let mut v = self.nn.weights.clone();
        v.extend_from_slice(&self.nn.weights_multi);
        v.extend_from_slice(&self.nn.biases);
        v.extend_from_slice(&self.nn.weights2);
        v.extend_from_slice(&self.nn.biases2);
        v
    }

    pub fn param_distance(&self, other: &Self) -> f64 {
        l2_slices(&self.flatten_params(), &other.flatten_params())
    }

    pub fn add_hidden_node_layer1(&mut self, rng: &mut impl Rng) {
        self.add_hidden_node_layer1_ext(rng, false, 0.1);
    }

    pub fn duplicate_hidden_node_layer1(&mut self, rng: &mut impl Rng, sigma: f64) {
        self.add_hidden_node_layer1_ext(rng, true, sigma);
    }

    fn add_hidden_node_layer1_ext(&mut self, rng: &mut impl Rng, duplicate: bool, sigma: f64) {
        let inputs = self.nn.inputs;

        if duplicate && self.nn.nodes >= 1 {
            let src = rng.gen_range(0..self.nn.nodes);
            let out_w = self.nn.weights.get(src).copied().unwrap_or(0.0);
            let bias = self.nn.biases.get(src).copied().unwrap_or(0.0);
            let multi_base = src * inputs;
            let multi: Vec<f64> = (0..inputs)
                .map(|j| {
                    self.nn
                        .weights_multi
                        .get(multi_base + j)
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();

            self.nn.nodes += 1;
            self.nn.weights.push(perturb_param(out_w, rng, sigma));
            for w in multi {
                self.nn.weights_multi.push(perturb_param(w, rng, sigma));
            }
            self.nn.biases.push(perturb_param(bias, rng, sigma));
            if self.nn.layers == 2 && self.nn.nodes2 > 0 {
                let nodes2 = self.nn.nodes2;
                for j in 0..nodes2 {
                    let idx = src * nodes2 + j;
                    let w = self.nn.weights2.get(idx).copied().unwrap_or(0.0);
                    self.nn.weights2.push(perturb_param(w, rng, sigma));
                }
            }
            return;
        }

        self.nn.nodes += 1;
        self.nn.weights.push(rng.gen::<f64>() * 0.1);
        self.nn.weights_multi.extend((0..inputs).map(|_| rng.gen::<f64>() * 0.1));
        self.nn.biases.push(rng.gen::<f64>() * 0.1);
        if self.nn.layers == 2 && self.nn.nodes2 > 0 {
            self.nn.weights2.extend((0..self.nn.nodes2).map(|_| rng.gen::<f64>() * 0.1));
        }
    }

    pub fn add_hidden_node_layer2(&mut self, rng: &mut impl Rng) {
        self.add_hidden_node_layer2_ext(rng, false, 0.1);
    }

    pub fn duplicate_hidden_node_layer2(&mut self, rng: &mut impl Rng, sigma: f64) {
        self.add_hidden_node_layer2_ext(rng, true, sigma);
    }

    fn add_hidden_node_layer2_ext(&mut self, rng: &mut impl Rng, duplicate: bool, sigma: f64) {
        if self.nn.layers != 2 {
            return;
        }

        if duplicate && self.nn.nodes2 >= 1 {
            let src = rng.gen_range(0..self.nn.nodes2);
            let bias = self.nn.biases2.get(src).copied().unwrap_or(0.0);
            let nodes = self.nn.nodes;
            let nodes2 = self.nn.nodes2;
            self.nn.nodes2 += 1;
            for i in 0..nodes {
                let idx = i * nodes2 + src;
                let w = self.nn.weights2.get(idx).copied().unwrap_or(0.0);
                self.nn.weights2.push(perturb_param(w, rng, sigma));
            }
            self.nn.biases2.push(perturb_param(bias, rng, sigma));
            return;
        }

        self.nn.nodes2 += 1;
        self.nn.weights2.extend((0..self.nn.nodes).map(|_| rng.gen::<f64>() * 0.1));
        self.nn.biases2.push(rng.gen::<f64>() * 0.1);
    }

    pub fn add_second_layer(&mut self, rng: &mut impl Rng, nodes2: usize) {
        self.add_second_layer_ext(rng, nodes2, false, 0.1);
    }

    pub fn duplicate_second_layer(&mut self, rng: &mut impl Rng, nodes2: usize, sigma: f64) {
        self.add_second_layer_ext(rng, nodes2, true, sigma);
    }

    fn add_second_layer_ext(&mut self, rng: &mut impl Rng, nodes2: usize, duplicate: bool, sigma: f64) {
        if self.nn.layers >= 2 {
            return;
        }
        let nodes2 = nodes2.max(1);

        self.nn.layers = 2;
        self.nn.nodes2 = nodes2;
        if duplicate && self.nn.nodes >= 1 {
            let src = rng.gen_range(0..self.nn.nodes);
            let template = self.nn.weights.get(src).copied().unwrap_or(0.0);
            let bias_template = self.nn.biases.get(src).copied().unwrap_or(0.0);
            self.nn.weights2 = (0..self.nn.nodes * nodes2)
                .map(|_| perturb_param(template, rng, sigma))
                .collect();
            self.nn.biases2 = (0..nodes2)
                .map(|_| perturb_param(bias_template, rng, sigma))
                .collect();
        } else {
            self.nn.weights2 = (0..self.nn.nodes * nodes2)
                .map(|_| rng.gen::<f64>() * 0.1)
                .collect();
            self.nn.biases2 = (0..nodes2).map(|_| rng.gen::<f64>() * 0.1).collect();
        }
    }

    pub fn remove_hidden_node_layer1(&mut self) -> bool {
        if self.nn.nodes <= 1 {
            return false;
        }
        self.nn.nodes -= 1;
        self.nn.weights.pop();
        for _ in 0..self.nn.inputs {
            self.nn.weights_multi.pop();
        }
        self.nn.biases.pop();
        if self.nn.layers == 2 && self.nn.nodes2 > 0 {
            self.nn.weights2.truncate(self.nn.nodes * self.nn.nodes2);
        }
        true
    }
}

impl Serialize for MlpBrain {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MlpBrainSerde {
            weights: self.nn.weights.clone(),
            weights_multi: self.nn.weights_multi.clone(),
            biases: self.nn.biases.clone(),
            weights2: self.nn.weights2.clone(),
            biases2: self.nn.biases2.clone(),
            nodes: self.nn.nodes,
            inputs: self.nn.inputs,
            nodes2: self.nn.nodes2,
            layers: self.nn.layers,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MlpBrain {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = MlpBrainSerde::deserialize(deserializer)?;
        let mut nn = NeuralNetwork::new(s.nodes, s.inputs, s.layers, s.nodes2);
        nn.weights = s.weights;
        nn.weights_multi = s.weights_multi;
        nn.biases = s.biases;
        nn.weights2 = s.weights2;
        nn.biases2 = s.biases2;
        nn.nn_type = if s.inputs <= 1 {
            NNType::SingleInput
        } else {
            NNType::MultiInput
        };
        Ok(MlpBrain { nn })
    }
}
