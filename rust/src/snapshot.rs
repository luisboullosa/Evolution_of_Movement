use crate::environment::Environment;
use crate::individual::Individual;
use crate::neural_network::NeuralNetwork;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

const SNAPSHOT_VERSION: u16 = 1;

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
                write!(f, "snapshot version {} not supported (expected {})", found, expected)
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
    pub nn_nodes: u32,
    pub nn_nodes2: u32,
    pub nn_inputs: u32,
    pub nn_layers: i32,
    pub weights: Vec<f32>,
    pub weights_multi: Vec<f32>,
    pub biases: Vec<f32>,
    pub weights2: Vec<f32>,
    pub biases2: Vec<f32>,
}

impl SnapshotIndividual {
    pub fn from_individual(ind: &Individual) -> Self {
        SnapshotIndividual {
            id: ind.id,
            position: ind.position as u32,
            resources: ind.resources as f32,
            age: ind.age,
            nn_nodes: ind.nn.nodes as u32,
            nn_nodes2: ind.nn.nodes2 as u32,
            nn_inputs: ind.nn.inputs as u32,
            nn_layers: ind.nn.layers,
            weights: ind.nn.weights.iter().map(|v| *v as f32).collect(),
            weights_multi: ind.nn.weights_multi.iter().map(|v| *v as f32).collect(),
            biases: ind.nn.biases.iter().map(|v| *v as f32).collect(),
            weights2: ind.nn.weights2.iter().map(|v| *v as f32).collect(),
            biases2: ind.nn.biases2.iter().map(|v| *v as f32).collect(),
        }
    }

    pub fn into_individual(self) -> Individual {
        let mut nn = NeuralNetwork::new(
            self.nn_nodes as usize,
            self.nn_inputs as usize,
            self.nn_layers,
            self.nn_nodes2 as usize,
        );
        for (dst, src) in nn.weights.iter_mut().zip(self.weights.iter()) {
            *dst = *src as f64;
        }
        for (dst, src) in nn.weights_multi.iter_mut().zip(self.weights_multi.iter()) {
            *dst = *src as f64;
        }
        for (dst, src) in nn.biases.iter_mut().zip(self.biases.iter()) {
            *dst = *src as f64;
        }
        for (dst, src) in nn.weights2.iter_mut().zip(self.weights2.iter()) {
            *dst = *src as f64;
        }
        for (dst, src) in nn.biases2.iter_mut().zip(self.biases2.iter()) {
            *dst = *src as f64;
        }
        let mut ind = Individual::new(nn, self.position as usize, self.id);
        ind.resources = self.resources as f64;
        ind.age = self.age;
        ind
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PopulationSnapshot {
    pub version: u16,
    pub timestep: i32,
    pub rng_seed_hint: u64,
    pub next_id: i32,
    pub env_width: u32,
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
    ) -> Self {
        PopulationSnapshot {
            version: SNAPSHOT_VERSION,
            timestep,
            rng_seed_hint,
            next_id,
            env_width: env.width as u32,
            env_perlin_time: env.perlin_time,
            env_resources: env.resources.iter().map(|v| *v as f32).collect(),
            env_perlin_base: env.perlin_base.iter().map(|v| *v as f32).collect(),
            individuals: individuals.iter().map(SnapshotIndividual::from_individual).collect(),
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
        if snapshot.version != SNAPSHOT_VERSION {
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
