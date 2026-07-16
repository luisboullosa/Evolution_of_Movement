use crate::environment::Environment;
use crate::individual::Individual;
use crate::brain::{BrainGenome, BrainInner, BrainKind};
use crate::neural_network::NeuralNetwork;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub const SNAPSHOT_VERSION_V1: u16 = 1;
pub const SNAPSHOT_VERSION: u16 = 2;

#[derive(Debug)]
pub enum SnapshotError {
    Io(std::io::Error),
    Codec(serde_cbor::Error),
    UnsupportedVersion { expected: u16, found: u16 },
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SnapshotError::Io(e) => write!(f, "snapshot I/O error: {}", e),
            SnapshotError::Codec(e) => write!(f, "snapshot codec error: {}", e),
            SnapshotError::UnsupportedVersion { expected, found } => {
                write!(
                    f,
                    "snapshot version {} not supported (expected {})",
                    found, expected
                )
            }
        }
    }
}

impl std::error::Error for SnapshotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SnapshotError::Io(e) => Some(e),
            SnapshotError::Codec(e) => Some(e),
            SnapshotError::UnsupportedVersion { .. } => None,
        }
    }
}

impl From<std::io::Error> for SnapshotError {
    fn from(value: std::io::Error) -> Self {
        SnapshotError::Io(value)
    }
}

impl From<serde_cbor::Error> for SnapshotError {
    fn from(value: serde_cbor::Error) -> Self {
        SnapshotError::Codec(value)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SnapshotIndividual {
    pub id: i32,
    pub position: u32,
    pub resources: f32,
    pub age: i32,
    // v1 legacy MLP fields (populated for backward compat)
    pub nn_nodes: u32,
    pub nn_nodes2: u32,
    pub nn_inputs: u32,
    pub nn_layers: i32,
    pub weights: Vec<f32>,
    pub weights_multi: Vec<f32>,
    pub biases: Vec<f32>,
    pub weights2: Vec<f32>,
    pub biases2: Vec<f32>,
    // v2 brain fields
    #[serde(default)]
    pub brain_kind: Option<String>,
    #[serde(default)]
    pub brain_json: Option<String>,
}

impl SnapshotIndividual {
    pub fn from_individual(ind: &Individual) -> Self {
        let brain_json = serde_json::to_string(&ind.brain).ok();
        let mut snap = SnapshotIndividual {
            id: ind.id,
            position: ind.position as u32,
            resources: ind.resources as f32,
            age: ind.age,
            nn_nodes: 0,
            nn_nodes2: 0,
            nn_inputs: 8,
            nn_layers: 1,
            weights: Vec::new(),
            weights_multi: Vec::new(),
            biases: Vec::new(),
            weights2: Vec::new(),
            biases2: Vec::new(),
            brain_kind: Some(ind.brain.kind().as_str().to_string()),
            brain_json,
        };

        if let Some(nn) = ind.brain.legacy_nn_view() {
            snap.nn_nodes = nn.nodes as u32;
            snap.nn_nodes2 = nn.nodes2 as u32;
            snap.nn_inputs = nn.inputs as u32;
            snap.nn_layers = nn.layers;
            snap.weights = nn.weights.iter().map(|v| *v as f32).collect();
            snap.weights_multi = nn.weights_multi.iter().map(|v| *v as f32).collect();
            snap.biases = nn.biases.iter().map(|v| *v as f32).collect();
            snap.weights2 = nn.weights2.iter().map(|v| *v as f32).collect();
            snap.biases2 = nn.biases2.iter().map(|v| *v as f32).collect();
        } else if let BrainInner::Perceptron(ref p) = ind.brain.inner {
            snap.nn_nodes = p.nodes as u32;
            snap.nn_inputs = p.inputs as u32;
            snap.weights_multi = p.weights.iter().map(|v| *v as f32).collect();
            snap.biases = p.biases.iter().map(|v| *v as f32).collect();
        }

        snap
    }

    pub fn into_individual(self) -> Individual {
        if let Some(ref json) = self.brain_json {
            if let Ok(brain) = serde_json::from_str::<BrainGenome>(json) {
                return Individual::new(brain, self.position as usize, self.id);
            }
        }

        // v1 fallback: reconstruct MLP/perceptron from flat weights
        if self.nn_layers >= 2 || self.weights2.iter().any(|&w| w != 0.0) {
            let mut nn = NeuralNetwork::new(
                self.nn_nodes as usize,
                self.nn_inputs as usize,
                self.nn_layers,
                self.nn_nodes2 as usize,
            );
            copy_weights(&mut nn, &self);
            let brain = BrainGenome {
                kind: BrainKind::Mlp,
                inner: BrainInner::Mlp(crate::brain::MlpBrain { nn }),
            };
            let mut ind = Individual::new(brain, self.position as usize, self.id);
            ind.resources = self.resources as f64;
            ind.age = self.age;
            return ind;
        }

        let mut p = crate::brain::PerceptronBrain::new(
            self.nn_inputs as usize,
            self.nn_nodes.max(1) as usize,
        );
        if !self.weights_multi.is_empty() {
            p.weights = self.weights_multi.iter().map(|v| *v as f64).collect();
        } else {
            p.weights = self.weights.iter().map(|v| *v as f64).collect();
        }
        p.biases = self.biases.iter().map(|v| *v as f64).collect();
        let brain = BrainGenome {
            kind: BrainKind::Perceptron,
            inner: BrainInner::Perceptron(p),
        };
        let mut ind = Individual::new(brain, self.position as usize, self.id);
        ind.resources = self.resources as f64;
        ind.age = self.age;
        ind
    }
}

fn copy_weights(nn: &mut NeuralNetwork, snap: &SnapshotIndividual) {
    for (dst, src) in nn.weights.iter_mut().zip(snap.weights.iter()) {
        *dst = *src as f64;
    }
    for (dst, src) in nn.weights_multi.iter_mut().zip(snap.weights_multi.iter()) {
        *dst = *src as f64;
    }
    for (dst, src) in nn.biases.iter_mut().zip(snap.biases.iter()) {
        *dst = *src as f64;
    }
    for (dst, src) in nn.weights2.iter_mut().zip(snap.weights2.iter()) {
        *dst = *src as f64;
    }
    for (dst, src) in nn.biases2.iter_mut().zip(snap.biases2.iter()) {
        *dst = *src as f64;
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PopulationSnapshot {
    pub version: u16,
    pub timestep: i32,
    pub rng_seed_hint: u64,
    pub next_id: i32,
    pub env_width: u32,
    #[serde(default)]
    pub env_height: u32,
    #[serde(default)]
    pub hex_dish_radius: u32,
    pub env_perlin_time: f64,
    pub env_resources: Vec<f32>,
    pub env_perlin_base: Vec<f32>,
    pub individuals: Vec<SnapshotIndividual>,
}

impl PopulationSnapshot {
    pub fn capture(
        env: &Environment,
        individuals: &[Individual],
        next_id: i32,
        rng_seed_hint: u64,
        timestep: i32,
        env_height: u32,
        hex_dish_radius: u32,
    ) -> Self {
        PopulationSnapshot {
            version: SNAPSHOT_VERSION,
            timestep,
            rng_seed_hint,
            next_id,
            env_width: env.width as u32,
            env_height,
            hex_dish_radius,
            env_perlin_time: env.perlin_time,
            env_resources: env.resources.iter().map(|v| *v as f32).collect(),
            env_perlin_base: env.perlin_base.iter().map(|v| *v as f32).collect(),
            individuals: individuals
                .iter()
                .map(SnapshotIndividual::from_individual)
                .collect(),
        }
    }

    pub fn write_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), SnapshotError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_cbor::to_writer(writer, self)?;
        Ok(())
    }

    pub fn read_from_path<P: AsRef<Path>>(path: P) -> Result<Self, SnapshotError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: PopulationSnapshot = serde_cbor::from_reader(reader)?;
        if snapshot.version != SNAPSHOT_VERSION && snapshot.version != SNAPSHOT_VERSION_V1 {
            return Err(SnapshotError::UnsupportedVersion {
                expected: SNAPSHOT_VERSION,
                found: snapshot.version,
            });
        }
        Ok(snapshot)
    }

    pub fn individuals(&self) -> &[SnapshotIndividual] {
        &self.individuals
    }
}
