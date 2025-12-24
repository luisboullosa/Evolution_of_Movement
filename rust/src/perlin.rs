use rand::Rng;
use rand::SeedableRng;

/// Simple placeholder Perlin-like generator. Returns values in [0,1].
pub struct Perlin;

impl Perlin {
    pub fn generate(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen::<f64>()).collect()
    }

    pub fn sample_time_slice(
        n: usize,
        seed: u64,
        time_offset: f64,
        _spatial_scale: f64,
        _time_scale: f64,
    ) -> Vec<f64> {
        // naive implementation that mixes seed and time_offset for deterministic output
        let tseed = ((time_offset * 1e6) as u64).wrapping_add(seed);
        Perlin::generate(n, tseed)
    }
}
