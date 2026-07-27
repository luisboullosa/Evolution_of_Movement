use crate::perlin::Perlin;
use crate::hex_dish::HexDishLayout;

#[inline]
fn scaled_resource(raw: f64, resource_scale: f64, contrast: f64) -> f64 {
    Perlin::shape_resource_contrast(raw, contrast) * resource_scale
}

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

    pub fn init_perlin(&mut self, seed: u64, resource_scale: f64, contrast: f64) {
        self.perlin_base = Perlin::generate(self.width, seed);
        self.resources = self
            .perlin_base
            .iter()
            .map(|&raw| scaled_resource(raw, resource_scale, contrast))
            .collect();
    }

    pub fn set_perlin_base(&mut self, base: &[f64]) {
        self.perlin_base = base.to_vec();
        self.resources = self.perlin_base.clone();
        self.width = self.perlin_base.len();
    }

    pub fn gather_at(&self, pos: usize, _local_population: i32) -> f64 {
        self.resources.get(pos % self.width).copied().unwrap_or(0.0)
    }

    pub fn init_perlin_hex_dish(
        &mut self,
        dish: &HexDishLayout,
        seed: u64,
        resource_scale: f64,
        spatial_scale: f64,
        time_scale: f64,
        contrast: f64,
    ) {
        let n = dish.cell_count();
        self.width = n;
        self.perlin_base = Vec::with_capacity(n);
        for coord in &dish.active_coords {
            let v = Perlin::sample_2d_time(
                coord.q as f64,
                coord.r as f64,
                0.0,
                seed,
                spatial_scale,
                time_scale,
            );
            self.perlin_base.push(v);
        }
        self.resources = self
            .perlin_base
            .iter()
            .map(|&raw| scaled_resource(raw, resource_scale, contrast))
            .collect();
    }

    pub fn step_hex_dish(
        &mut self,
        dish: &HexDishLayout,
        seed: u64,
        time_speed: f64,
        time_amplitude: f64,
        spatial_scale: f64,
        time_scale: f64,
        resource_scale: f64,
        contrast: f64,
    ) {
        self.perlin_time += time_speed;
        for (idx, coord) in dish.active_coords.iter().enumerate() {
            let temporal = Perlin::sample_2d_time(
                coord.q as f64,
                coord.r as f64,
                self.perlin_time,
                seed,
                spatial_scale,
                time_scale,
            );
            let base_raw = self.perlin_base.get(idx).copied().unwrap_or(0.0);
            let combined_raw =
                (base_raw + temporal * time_amplitude).clamp(0.0, 1.0 + time_amplitude);
            if let Some(slot) = self.resources.get_mut(idx) {
                *slot = scaled_resource(combined_raw, resource_scale, contrast);
            }
        }
    }

    pub fn step(
        &mut self,
        seed: u64,
        time_speed: f64,
        time_amplitude: f64,
        spatial_scale: f64,
        time_scale: f64,
        resource_scale: f64,
        contrast: f64,
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
            let combined_raw =
                (self.perlin_base[i] + v * time_amplitude).clamp(0.0, 1.0 + time_amplitude);
            self.resources[i] = scaled_resource(combined_raw, resource_scale, contrast);
        }
    }
}
