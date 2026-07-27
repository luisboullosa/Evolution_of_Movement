use rand::Rng;
use rand::SeedableRng;

/// Simple placeholder Perlin-like generator. Returns values in [0,1].
pub struct Perlin;

impl Perlin {
    #[inline]
    fn hash3(seed: u64, xi: i64, yi: i64, zi: i64) -> u64 {
        let mut h = seed
            ^ (xi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (yi as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (zi as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        h ^= h >> 33;
        h
    }

    #[inline]
    fn hash_unit(seed: u64, xi: i64, yi: i64, zi: i64) -> f64 {
        let v = Self::hash3(seed, xi, yi, zi);
        (v as f64) / (u64::MAX as f64)
    }

    #[inline]
    fn smoothstep(t: f64) -> f64 {
        t * t * (3.0 - 2.0 * t)
    }

    #[inline]
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }

    /// Deterministic 2D spatial + 1D temporal value-noise sample in [0, 1].
    pub fn sample_2d_time(
        x: f64,
        y: f64,
        time: f64,
        seed: u64,
        spatial_scale: f64,
        time_scale: f64,
    ) -> f64 {
        let s = spatial_scale.max(1e-9);
        let ts = time_scale.max(1e-9);

        let xf = x * s;
        let yf = y * s;
        let tf = time * ts;

        let x0 = xf.floor() as i64;
        let y0 = yf.floor() as i64;
        let t0 = tf.floor() as i64;

        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let t1 = t0 + 1;

        let ux = Self::smoothstep(xf - x0 as f64);
        let uy = Self::smoothstep(yf - y0 as f64);
        let ut = Self::smoothstep(tf - t0 as f64);

        let c000 = Self::hash_unit(seed, x0, y0, t0);
        let c100 = Self::hash_unit(seed, x1, y0, t0);
        let c010 = Self::hash_unit(seed, x0, y1, t0);
        let c110 = Self::hash_unit(seed, x1, y1, t0);
        let c001 = Self::hash_unit(seed, x0, y0, t1);
        let c101 = Self::hash_unit(seed, x1, y0, t1);
        let c011 = Self::hash_unit(seed, x0, y1, t1);
        let c111 = Self::hash_unit(seed, x1, y1, t1);

        let x00 = Self::lerp(c000, c100, ux);
        let x10 = Self::lerp(c010, c110, ux);
        let y0v = Self::lerp(x00, x10, uy);

        let x01 = Self::lerp(c001, c101, ux);
        let x11 = Self::lerp(c011, c111, ux);
        let y1v = Self::lerp(x01, x11, uy);

        Self::lerp(y0v, y1v, ut)
    }

    pub fn generate(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen::<f64>()).collect()
    }

    pub fn sample_time_slice(
        n: usize,
        seed: u64,
        time_offset: f64,
        spatial_scale: f64,
        time_scale: f64,
    ) -> Vec<f64> {
        (0..n)
            .map(|i| Self::sample_2d_time(i as f64, 0.0, time_offset, seed, spatial_scale, time_scale))
            .collect()
    }

    /// Stretch Perlin samples away from 0.5 so rich/poor patches contrast more.
    /// `contrast` = 1 leaves the sample unchanged; values > 1 sharpen the landscape.
    pub fn shape_resource_contrast(raw: f64, contrast: f64) -> f64 {
        let r = raw.clamp(0.0, 1.0);
        if contrast <= 1.0 + 1e-9 {
            return r;
        }
        (0.5 + (r - 0.5) * contrast).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Perlin;

    #[test]
    fn sample_2d_time_is_deterministic() {
        let a = Perlin::sample_2d_time(3.2, -1.7, 0.5, 42, 0.1, 0.2);
        let b = Perlin::sample_2d_time(3.2, -1.7, 0.5, 42, 0.1, 0.2);
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn sample_2d_time_varies_with_time() {
        let a = Perlin::sample_2d_time(8.0, 4.0, 0.0, 7, 0.05, 0.1);
        let b = Perlin::sample_2d_time(8.0, 4.0, 10.0, 7, 0.05, 0.1);
        assert!((a - b).abs() > 1e-9);
    }

    #[test]
    fn sample_2d_time_varies_with_y_dimension() {
        let a = Perlin::sample_2d_time(8.0, 1.0, 2.0, 7, 0.05, 0.1);
        let b = Perlin::sample_2d_time(8.0, 9.0, 2.0, 7, 0.05, 0.1);
        assert!((a - b).abs() > 1e-9);
    }

    #[test]
    fn resource_contrast_spreads_around_midpoint() {
        let flat = Perlin::shape_resource_contrast(0.5, 1.4);
        assert!((flat - 0.5).abs() < 1e-9);
        let rich = Perlin::shape_resource_contrast(0.7, 1.4);
        let poor = Perlin::shape_resource_contrast(0.3, 1.4);
        assert!(rich > 0.7);
        assert!(poor < 0.3);
    }
}
