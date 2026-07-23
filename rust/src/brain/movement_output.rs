use crate::config::MovementOutputMode;
use super::genome::BrainOutput;

pub fn default_movement_output_mode(
    policy: crate::config::ArchitecturePolicy,
    legacy_spatial_graph: bool,
) -> MovementOutputMode {
    use crate::config::ArchitecturePolicy;
    match policy {
        ArchitecturePolicy::EvolveSpatialGraph if legacy_spatial_graph => {
            MovementOutputMode::SeparateReadout
        }
        ArchitecturePolicy::EvolveSpatialGraph => MovementOutputMode::SeparateReadout,
        ArchitecturePolicy::EvolveGoo => MovementOutputMode::HiddenSum,
        _ => MovementOutputMode::HiddenSum,
    }
}

/// Inverse of `step_engine::direction_signal_to_degrees` (ignores outer tanh compression edge cases).
pub fn direction_signal_from_degrees(degrees: f64) -> f64 {
    let normalized = degrees.rem_euclid(360.0) / 360.0;
    let t = (2.0 * normalized - 1.0).clamp(-0.999_999, 0.999_999);
    t.atanh()
}

pub const HEX_SECTOR_ANGLES_DEG: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
/// N, S, W, E pool order from `cardinal_pool_inputs`.
pub const CARDINAL_ANGLES_DEG: [f64; 4] = [90.0, 270.0, 180.0, 0.0];

const COMPASS_EPS: f64 = 1e-12;

/// Unit-vector sum of centered architecture outputs. Uniform activations cancel to zero.
pub fn hex_compass_vector(values: &[f64], angles_deg: &[f64]) -> (f64, f64) {
    let n = values.len().min(angles_deg.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().take(n).sum::<f64>() / n as f64;
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for i in 0..n {
        let centered = values[i] - mean;
        let rad = angles_deg[i].to_radians();
        sin_sum += centered * rad.sin();
        cos_sum += centered * rad.cos();
    }
    (sin_sum, cos_sum)
}

pub fn hex_compass_magnitude(values: &[f64], angles_deg: &[f64]) -> f64 {
    let (sin_sum, cos_sum) = hex_compass_vector(values, angles_deg);
    (sin_sum * sin_sum + cos_sum * cos_sum).sqrt()
}

/// Direction + move from a vector sum of architecture node outputs (centered).
/// Equal activations on all nodes → zero vector → gather (no move).
pub fn hex_compass_output(values: &[f64], angles_deg: &[f64]) -> BrainOutput {
    let (sin_sum, cos_sum) = hex_compass_vector(values, angles_deg);
    let magnitude = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt();
    if magnitude <= COMPASS_EPS {
        return BrainOutput {
            move_signal: 0.0,
            direction_signal: 0.0,
        };
    }
    BrainOutput {
        move_signal: magnitude,
        direction_signal: direction_signal_from_degrees(sin_sum.atan2(cos_sum).to_degrees()),
    }
}

pub fn sector_mask_centroid_degrees(mask: u8, sector_count: usize) -> f64 {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    let mut any = false;
    for s in 0..sector_count {
        if mask & (1u8 << s) != 0 {
            let rad = HEX_SECTOR_ANGLES_DEG
                .get(s)
                .copied()
                .unwrap_or(0.0)
                .to_radians();
            sin_sum += rad.sin();
            cos_sum += rad.cos();
            any = true;
        }
    }
    if !any {
        return 0.0;
    }
    sin_sum.atan2(cos_sum).to_degrees()
}

pub fn hidden_sum_output(activations: &[f64]) -> BrainOutput {
    BrainOutput {
        move_signal: activations.iter().sum(),
        direction_signal: activations.get(1).copied().unwrap_or_else(|| activations[0]),
    }
}

pub fn split_channels_output(thought: &[f64]) -> BrainOutput {
    BrainOutput {
        move_signal: thought.first().copied().unwrap_or(0.0),
        direction_signal: thought.get(1).copied().unwrap_or_else(|| thought[0]),
    }
}

pub fn readout_output(move_raw: f64, direction_raw: f64, mode: MovementOutputMode) -> BrainOutput {
    match mode {
        MovementOutputMode::SeparateReadout => BrainOutput {
            move_signal: move_raw.tanh(),
            direction_signal: direction_raw.tanh(),
        },
        MovementOutputMode::LinearReadout => BrainOutput {
            move_signal: move_raw,
            direction_signal: direction_raw,
        },
        _ => BrainOutput {
            move_signal: move_raw,
            direction_signal: direction_raw,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_roundtrip_near_equator() {
        for deg in [0.0, 60.0, 120.0, 180.0, 240.0, 300.0] {
            let signal = direction_signal_from_degrees(deg);
            let back = crate::step_engine::direction_signal_to_degrees(signal);
            assert!(
                (back - deg).abs() < 1.0,
                "deg {deg} -> signal {signal} -> {back}"
            );
        }
    }

    #[test]
    fn hex_compass_uniform_activations_do_not_move() {
        let uniform = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let out = hex_compass_output(&uniform, &HEX_SECTOR_ANGLES_DEG);
        assert!(out.move_signal.abs() <= COMPASS_EPS);
        assert!(!out.move_signal.is_nan());
    }

    #[test]
    fn hex_compass_favors_east_activation() {
        let activations = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = hex_compass_output(&activations, &HEX_SECTOR_ANGLES_DEG);
        assert!(out.move_signal > 0.1);
        let deg = crate::step_engine::direction_signal_to_degrees(out.direction_signal);
        assert!(deg < 30.0 || deg > 330.0, "expected ~0°, got {deg}");
    }

    #[test]
    fn hex_compass_cardinal_uniform_is_flat() {
        let uniform = [0.5, 0.5, 0.5, 0.5];
        let out = hex_compass_output(&uniform, &CARDINAL_ANGLES_DEG);
        assert!(out.move_signal.abs() <= COMPASS_EPS);
    }
}
