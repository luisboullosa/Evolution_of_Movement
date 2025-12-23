use crate::perlin::Perlin;

pub struct Environment {
    pub width: usize,
    pub resources: Vec<f64>,
    pub perlin_base: Vec<f64>,
    pub perlin_time: f64,
}

impl Environment {
    pub fn new(width: usize) -> Self {
        Environment {
            width,
            resources: vec![0.0; width],
            perlin_base: vec![0.0; width],
            perlin_time: 0.0,
        }
    }

    pub fn init_perlin(&mut self, seed: u64) {
        self.perlin_base = Perlin::generate(self.width, seed);
        self.resources = self.perlin_base.clone();
    }

    pub fn set_perlin_base(&mut self, base: &[f64]) {
        self.perlin_base = base.to_vec();
        self.resources = self.perlin_base.clone();
        self.width = self.perlin_base.len();
    }

    pub fn gather_at(&self, pos: usize, _local_population: i32) -> f64 {
        self.resources.get(pos % self.width).copied().unwrap_or(0.0)
    }

    pub fn step(&mut self, change_rate: f64, seed: u64, use_perlin_time: bool, time_speed: f64, time_amplitude: f64, spatial_scale: f64, time_scale: f64) {
        if use_perlin_time {
            self.perlin_time += time_speed;
            let slice = Perlin::sample_time_slice(self.width, seed, self.perlin_time, spatial_scale, time_scale);
            for i in 0..self.width {
                self.resources[i] = self.perlin_base[i] + slice[i] * time_amplitude;
            }
        } else {
            // simple additive noise
            for i in 0..self.width {
                self.resources[i] += change_rate;
            }
        }
    }
}
