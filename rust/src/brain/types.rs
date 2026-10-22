//! Shared brain data structures, constants, and math utilities.
use crate::config::MovementOutputMode;
use serde::{Deserialize, Serialize};

pub const SENSOR_DIM: usize = 8;
/// Perceptron uses current-cell sensors only (resource, competition, internal stores).
pub const PERCEPTRON_SENSOR_DIM: usize = 3;
pub const THOUGHT_DIM: usize = 8;
pub const EMBED_DIM: usize = 8;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpatialRuntime {
    pub node_embeddings: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct ForwardContext<'a> {
    pub sensors: [f64; SENSOR_DIM],
    pub position: usize,
    pub hidden_buf: &'a mut [f64],
    pub out2_buf: &'a mut [f64],
    pub spatial_runtime: Option<&'a mut SpatialRuntime>,
    pub movement_output_mode: MovementOutputMode,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BrainOutput {
    pub move_signal: f64,
    pub direction_signal: f64,
}

/// Normalized root-mean-squared error / Euclidean distance across slices of weights.
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

/// Cauchy mutation perturbation on continuous weights.
pub fn perturb_param(v: f64, rng: &mut impl rand::Rng, sigma: f64) -> f64 {
    use rand_distr::{Cauchy, Distribution};
    if sigma <= 0.0 {
        return v;
    }
    v + Cauchy::new(0.0, sigma).unwrap().sample(rng)
}
