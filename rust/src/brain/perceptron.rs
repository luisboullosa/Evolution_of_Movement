use super::genome::{BrainOutput, PERCEPTRON_SENSOR_DIM, SENSOR_DIM};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptronBrain {
    pub inputs: usize,
    pub nodes: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

impl PerceptronBrain {
    pub fn new(inputs: usize, nodes: usize) -> Self {
        let inputs = if inputs == 0 {
            PERCEPTRON_SENSOR_DIM
        } else {
            inputs.min(SENSOR_DIM)
        };
        let nodes = nodes.max(1);
        PerceptronBrain {
            inputs,
            nodes,
            weights: vec![0.0; nodes * inputs],
            biases: vec![0.0; nodes],
        }
    }

    pub fn complexity(&self) -> usize {
        self.nodes + self.weights.len() / self.inputs.max(1)
    }

    pub fn forward(&self, sensors: &[f64; SENSOR_DIM], hidden_buf: &mut [f64]) -> BrainOutput {
        let mut hidden = vec![0.0; self.nodes];
        for i in 0..self.nodes {
            let mut acc = self.biases.get(i).copied().unwrap_or(0.0);
            for j in 0..self.inputs.min(SENSOR_DIM) {
                let idx = i * self.inputs + j;
                acc += self.weights.get(idx).copied().unwrap_or(0.0) * sensors[j];
            }
            hidden[i] = acc.tanh();
            if i < hidden_buf.len() {
                hidden_buf[i] = hidden[i];
            }
        }
        let move_signal = hidden.iter().sum::<f64>();
        let direction_signal = hidden.get(1).copied().unwrap_or_else(|| hidden[0]);
        BrainOutput {
            move_signal,
            direction_signal,
        }
    }

    pub fn flatten_params(&self) -> Vec<f64> {
        let mut v = self.weights.clone();
        v.extend_from_slice(&self.biases);
        v
    }

    pub fn param_distance(&self, other: &Self) -> f64 {
        l2_slices(&self.flatten_params(), &other.flatten_params())
    }
}

pub fn l2_slices(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len()).max(1);
    let mut sum = 0.0;
    for i in 0..n {
        let da = a.get(i).copied().unwrap_or(0.0);
        let db = b.get(i).copied().unwrap_or(0.0);
        sum += (da - db).powi(2);
    }
    (sum / n as f64).sqrt()
}

pub fn perturb_param(v: f64, rng: &mut impl rand::Rng, sigma: f64) -> f64 {
    use rand_distr::{Cauchy, Distribution};
    if sigma <= 0.0 {
        return v;
    }
    v + Cauchy::new(0.0, sigma).unwrap().sample(rng)
}
