use super::genome::{BrainOutput, EMBED_DIM, SENSOR_DIM};
use super::genome::SpatialRuntime;
use super::movement_output::{hidden_sum_output, hex_compass_output, readout_output, sector_mask_centroid_degrees};
use super::perceptron::{l2_slices, perturb_param};
use crate::config::MovementOutputMode;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const SPATIAL_SECTOR_COUNT: usize = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialEdge {
    pub from: usize,
    pub to: usize,
    pub weight_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialGraphBrain {
    pub landmark_count: usize,
    pub edges: Vec<SpatialEdge>,
    pub edge_weights: Vec<f64>,
    pub readout_weights: Vec<f64>,
    pub readout_biases: Vec<f64>,
    /// Per-landmark bitmask over hex directions (bit `s` = sector `s` resource input).
    #[serde(default = "default_landmark_sectors")]
    pub landmark_sectors: Vec<u8>,
    pub learn_rate: f64,
    pub message_passes: usize,
}

fn default_landmark_sectors() -> Vec<u8> {
    Vec::new()
}

pub fn all_sectors_mask() -> u8 {
    (1u8 << SPATIAL_SECTOR_COUNT) - 1
}

/// Random contiguous arc of 1–3 hex directions (head-direction–like tuning).
pub fn random_sector_mask(rng: &mut impl Rng) -> u8 {
    let start = rng.gen_range(0..SPATIAL_SECTOR_COUNT as u8);
    let width = rng.gen_range(1..=3u8);
    let mut mask = 0u8;
    for k in 0..width {
        mask |= 1 << ((start + k) % SPATIAL_SECTOR_COUNT as u8);
    }
    mask.max(1)
}

impl SpatialGraphBrain {
    pub fn new(initial_landmarks: usize) -> Self {
        let landmark_count = initial_landmarks.max(1);
        let mut rng = rand::thread_rng();
        let landmark_sectors = (0..landmark_count)
            .map(|_| random_sector_mask(&mut rng))
            .collect();
        SpatialGraphBrain {
            landmark_count,
            edges: Vec::new(),
            edge_weights: vec![0.0],
            readout_weights: vec![0.0; 2 * (landmark_count + SENSOR_DIM)],
            readout_biases: vec![0.0; 2],
            landmark_sectors,
            learn_rate: 0.05,
            message_passes: 2,
        }
    }

    pub fn ensure_landmark_sectors(&mut self) {
        if self.landmark_sectors.len() < self.landmark_count {
            let mut rng = rand::thread_rng();
            while self.landmark_sectors.len() < self.landmark_count {
                self.landmark_sectors.push(random_sector_mask(&mut rng));
            }
        }
        self.landmark_sectors.truncate(self.landmark_count);
        for mask in &mut self.landmark_sectors {
            if *mask == 0 {
                *mask = all_sectors_mask();
            }
        }
    }

    pub fn landmark_count(&self) -> usize {
        self.landmark_count
    }

    pub fn complexity(&self) -> usize {
        self.landmark_count + self.edges.len() + self.edge_weights.len()
    }

    pub fn structure_signature(&self) -> String {
        let masks: Vec<String> = self
            .landmark_sectors
            .iter()
            .take(self.landmark_count)
            .map(|m| format!("{m:02x}"))
            .collect();
        format!(
            "sg:{}:{}:{}:[{}]",
            self.landmark_count,
            self.edges.len(),
            self.message_passes,
            masks.join(",")
        )
    }

    pub fn init_runtime(&self) -> SpatialRuntime {
        SpatialRuntime {
            node_embeddings: (0..self.landmark_count)
                .map(|_| vec![0.0; EMBED_DIM])
                .collect(),
        }
    }

    /// Build a landmark-specific observation from masked hex edge sensors (+ optional context).
    pub fn landmark_observation(
        sensors: &[f64; SENSOR_DIM],
        position: usize,
        sector_mask: u8,
    ) -> Vec<f64> {
        let mut obs = vec![0.0; EMBED_DIM];
        let mut active = 0usize;
        for s in 0..SPATIAL_SECTOR_COUNT {
            if (sector_mask >> s) & 1 != 0 {
                obs[s.min(EMBED_DIM - 1)] += sensors[s];
                active += 1;
            }
        }
        if active > 0 {
            let inv = 1.0 / active as f64;
            for s in 0..SPATIAL_SECTOR_COUNT.min(EMBED_DIM) {
                if (sector_mask >> s) & 1 != 0 {
                    obs[s] *= inv;
                }
            }
        }
        if EMBED_DIM > SPATIAL_SECTOR_COUNT {
            obs[SPATIAL_SECTOR_COUNT] = sensors[6];
        }
        if EMBED_DIM > SPATIAL_SECTOR_COUNT + 1 {
            obs[SPATIAL_SECTOR_COUNT + 1] = sensors[7];
        }
        if EMBED_DIM > SENSOR_DIM {
            obs[SENSOR_DIM] = (position as f64 % 1000.0) / 1000.0;
        }
        obs
    }

    pub fn forward(
        &self,
        sensors: &[f64; SENSOR_DIM],
        position: usize,
        runtime: &mut SpatialRuntime,
        hidden_buf: &mut [f64],
        output_mode: MovementOutputMode,
    ) -> BrainOutput {
        let sectors: Vec<u8> = if self.landmark_sectors.len() >= self.landmark_count {
            self.landmark_sectors[..self.landmark_count].to_vec()
        } else {
            vec![all_sectors_mask(); self.landmark_count]
        };

        if !runtime.node_embeddings.is_empty() {
            let mut best_idx = 0usize;
            let mut best_dist = f64::MAX;
            for (i, emb) in runtime.node_embeddings.iter().enumerate() {
                let observe = Self::landmark_observation(sensors, position, sectors[i]);
                let dist: f64 = emb
                    .iter()
                    .zip(observe.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            let observe =
                Self::landmark_observation(sensors, position, sectors[best_idx]);
            let emb = &mut runtime.node_embeddings[best_idx];
            for (e, o) in emb.iter_mut().zip(observe.iter()) {
                *e += self.learn_rate * (o - *e);
            }
        }

        let mut activations: Vec<f64> = runtime
            .node_embeddings
            .iter()
            .map(|emb| emb.iter().sum::<f64>() / EMBED_DIM as f64)
            .collect();

        for _ in 0..self.message_passes {
            let mut next = activations.clone();
            for edge in &self.edges {
                if edge.from >= activations.len() || edge.to >= next.len() {
                    continue;
                }
                let w = self
                    .edge_weights
                    .get(edge.weight_idx)
                    .copied()
                    .unwrap_or(0.5);
                next[edge.to] += w * activations[edge.from].tanh();
            }
            activations = next.into_iter().map(|v| v.tanh()).collect();
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
            let sectors: Vec<u8> = if self.landmark_sectors.len() >= self.landmark_count {
                self.landmark_sectors[..self.landmark_count].to_vec()
            } else {
                vec![all_sectors_mask(); self.landmark_count]
            };
            let angles: Vec<f64> = sectors
                .iter()
                .take(activations.len())
                .map(|&mask| sector_mask_centroid_degrees(mask, SPATIAL_SECTOR_COUNT))
                .collect();
            return hex_compass_output(&activations, &angles);
        }

        let mut features = activations;
        for s in 0..SPATIAL_SECTOR_COUNT.min(SENSOR_DIM) {
            features.push(sensors[s]);
        }
        while features.len() < self.landmark_count + SENSOR_DIM {
            features.push(0.0);
        }

        let mut move_raw = self.readout_biases.first().copied().unwrap_or(0.0);
        let mut direction_raw = self.readout_biases.get(1).copied().unwrap_or(0.0);
        let feat_len = self.landmark_count + SENSOR_DIM;
        for (i, &f) in features.iter().take(feat_len).enumerate() {
            let wi = i * 2;
            move_raw += self.readout_weights.get(wi).copied().unwrap_or(0.0) * f;
            direction_raw += self
                .readout_weights
                .get(wi + 1)
                .copied()
                .unwrap_or(0.0)
                * f;
        }

        readout_output(move_raw, direction_raw, output_mode)
    }

    pub fn flatten_params(&self) -> Vec<f64> {
        let mut v = self.edge_weights.clone();
        v.extend_from_slice(&self.readout_weights);
        v.extend_from_slice(&self.readout_biases);
        v.push(self.learn_rate);
        v
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let struct_diff = if self.landmark_count != other.landmark_count
            || self.edges.len() != other.edges.len()
            || self.landmark_sectors != other.landmark_sectors
        {
            0.5
        } else {
            0.0
        };
        struct_diff + l2_slices(&self.flatten_params(), &other.flatten_params())
    }

    pub fn add_landmark(&mut self, rng: &mut impl Rng) {
        self.add_landmark_ext(rng, false, 0.1);
    }

    pub fn duplicate_landmark(&mut self, rng: &mut impl Rng, sigma: f64) {
        self.add_landmark_ext(rng, true, sigma);
    }

    fn add_landmark_ext(&mut self, rng: &mut impl Rng, duplicate: bool, sigma: f64) {
        let new_idx = self.landmark_count;
        self.landmark_count += 1;
        let insert_at = new_idx * 2;
        let (w0, w1) = if duplicate && new_idx > 0 {
            let src = rng.gen_range(0..new_idx);
            (
                perturb_param(self.readout_weights[src * 2], rng, sigma),
                perturb_param(self.readout_weights[src * 2 + 1], rng, sigma),
            )
        } else {
            (rng.gen::<f64>() * 0.1, rng.gen::<f64>() * 0.1)
        };
        self.readout_weights.insert(insert_at + 1, w1);
        self.readout_weights.insert(insert_at, w0);
        let sector_mask = if duplicate && !self.landmark_sectors.is_empty() {
            let src = rng.gen_range(0..self.landmark_sectors.len());
            let base = self.landmark_sectors[src];
            if rng.gen_bool(0.35) {
                flip_random_sector(base, rng)
            } else {
                base
            }
        } else {
            random_sector_mask(rng)
        };
        self.landmark_sectors.push(sector_mask);
    }

    pub fn mutate_sector_mask(&mut self, rng: &mut impl Rng) -> bool {
        self.ensure_landmark_sectors();
        if self.landmark_sectors.is_empty() {
            return false;
        }
        let idx = rng.gen_range(0..self.landmark_sectors.len());
        self.landmark_sectors[idx] = flip_random_sector(self.landmark_sectors[idx], rng);
        true
    }

    pub fn add_edge(&mut self, rng: &mut impl Rng) {
        self.add_edge_ext(rng, false, 0.1);
    }

    pub fn duplicate_edge(&mut self, rng: &mut impl Rng, sigma: f64) {
        self.add_edge_ext(rng, true, sigma);
    }

    fn add_edge_ext(&mut self, rng: &mut impl Rng, duplicate: bool, sigma: f64) {
        if self.landmark_count < 2 {
            return;
        }
        let from = rng.gen_range(0..self.landmark_count);
        let mut to = rng.gen_range(0..self.landmark_count);
        if to == from {
            to = (to + 1) % self.landmark_count;
        }
        let weight_idx = self.edge_weights.len();
        let w = if duplicate && !self.edge_weights.is_empty() {
            let src = self.edge_weights[rng.gen_range(0..self.edge_weights.len())];
            perturb_param(src, rng, sigma)
        } else {
            rng.gen::<f64>() * 0.2
        };
        self.edge_weights.push(w);
        self.edges.push(SpatialEdge {
            from,
            to,
            weight_idx,
        });
    }

    pub fn remove_edge(&mut self) -> bool {
        if self.edges.is_empty() {
            return false;
        }
        self.edges.pop();
        if !self.edge_weights.is_empty() {
            self.edge_weights.pop();
        }
        true
    }
}

fn flip_random_sector(mask: u8, rng: &mut impl Rng) -> u8 {
    let bit = rng.gen_range(0..SPATIAL_SECTOR_COUNT as u8);
    let flipped = mask ^ (1 << bit);
    if flipped == 0 {
        all_sectors_mask()
    } else {
        flipped
    }
}
