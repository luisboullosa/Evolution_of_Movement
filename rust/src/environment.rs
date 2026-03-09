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

    pub fn init_perlin(&mut self, seed: u64, resource_scale: f64) {
        self.perlin_base = Perlin::generate(self.width, seed);
        // apply resource scale to the perlin base and initialize resources
        for v in self.perlin_base.iter_mut().take(self.width) {
            *v *= resource_scale;
        }
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

    pub fn step(
        &mut self,
        seed: u64,
        time_speed: f64,
        time_amplitude: f64,
        spatial_scale: f64,
        time_scale: f64,
        resource_scale: f64,
    ) {
        // Always advance Perlin temporal evolution slice
        self.perlin_time += time_speed;
        let slice = Perlin::sample_time_slice(
            self.width,
            seed,
            self.perlin_time,
            spatial_scale,
            time_scale,
        );
        for (i, v) in slice.iter().enumerate().take(self.width) {
            // apply temporal variation and then scale resources by resource_scale
            self.resources[i] = (self.perlin_base[i] + v * time_amplitude) * resource_scale;
        }
    }
}
