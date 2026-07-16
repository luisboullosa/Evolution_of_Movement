//! Ecology and locomotion presets for stable, evolvable populations.
//!
//! Tuned for hex-dish runs (~10k cells, `hex_dish_radius` 58) without depletable
//! stock regrowth (step 1 in the ecology roadmap). `max_population` is a safety
//! margin above the expected soft equilibrium, not the primary regulator.

use crate::config::{ArchitecturePolicy, Config, MovementSpeed};

/// Ecology knobs shared across architecture policies for a given locomotion speed.
#[derive(Debug, Clone, Copy)]
pub struct EcologyTuning {
    pub movement_interval: u32,
    pub movement_cost_factor: f64,
    pub env_time_speed: f64,
    pub resource_scale: f64,
    pub reproduction_threshold: f64,
    pub reproduction_cost: f64,
    pub basal_metabolic_cost: f64,
    pub maintenance_cost_per_node: f64,
    pub death_risk: f64,
    pub initial_population: i32,
    /// Safety cap ~35–50% above expected equilibrium.
    pub max_population: i32,
}

impl EcologyTuning {
    pub fn fast() -> Self {
        EcologyTuning {
            movement_interval: 1,
            movement_cost_factor: 1.05,
            env_time_speed: 0.018,
            resource_scale: 0.58,
            reproduction_threshold: 40.0,
            reproduction_cost: 10.0,
            basal_metabolic_cost: 0.095,
            maintenance_cost_per_node: 0.042,
            death_risk: 8.0e-8,
            initial_population: 3000,
            max_population: 10000,
        }
    }

    pub fn medium() -> Self {
        EcologyTuning {
            movement_interval: 5,
            movement_cost_factor: 0.72,
            env_time_speed: 0.0036,
            resource_scale: 0.64,
            reproduction_threshold: 38.0,
            reproduction_cost: 9.0,
            basal_metabolic_cost: 0.088,
            maintenance_cost_per_node: 0.040,
            death_risk: 5.0e-8,
            initial_population: 2500,
            max_population: 9000,
        }
    }

    pub fn slow() -> Self {
        EcologyTuning {
            movement_interval: 25,
            movement_cost_factor: 0.40,
            env_time_speed: 0.00072,
            resource_scale: 0.66,
            reproduction_threshold: 38.0,
            reproduction_cost: 8.5,
            basal_metabolic_cost: 0.082,
            maintenance_cost_per_node: 0.034,
            death_risk: 3.0e-8,
            initial_population: 2000,
            max_population: 8000,
        }
    }

    /// Smaller equilibrium band (~70–85% of `max_population`), meaningful node maintenance.
    pub fn plateau80_fast() -> Self {
        EcologyTuning {
            movement_interval: 1,
            movement_cost_factor: 1.12,
            env_time_speed: 0.018,
            resource_scale: 0.34,
            reproduction_threshold: 49.0,
            reproduction_cost: 11.5,
            basal_metabolic_cost: 0.100,
            maintenance_cost_per_node: 0.030,
            death_risk: 8.0e-8,
            initial_population: 800,
            max_population: 5000,
        }
    }

    pub fn plateau80_medium() -> Self {
        EcologyTuning {
            movement_interval: 5,
            movement_cost_factor: 0.90,
            env_time_speed: 0.0036,
            resource_scale: 0.355,
            reproduction_threshold: 48.0,
            reproduction_cost: 11.2,
            basal_metabolic_cost: 0.097,
            maintenance_cost_per_node: 0.028,
            death_risk: 5.0e-8,
            initial_population: 700,
            max_population: 4500,
        }
    }

    pub fn plateau80_slow() -> Self {
        EcologyTuning {
            movement_interval: 25,
            movement_cost_factor: 0.50,
            env_time_speed: 0.00072,
            resource_scale: 0.38,
            reproduction_threshold: 47.0,
            reproduction_cost: 10.5,
            basal_metabolic_cost: 0.090,
            maintenance_cost_per_node: 0.026,
            death_risk: 3.0e-8,
            initial_population: 600,
            max_population: 4000,
        }
    }

    pub fn plateau80_for_speed(speed: MovementSpeed) -> Self {
        match speed {
            MovementSpeed::Fast => Self::plateau80_fast(),
            MovementSpeed::Medium => Self::plateau80_medium(),
            MovementSpeed::Slow => Self::plateau80_slow(),
        }
    }

    pub fn for_speed(speed: MovementSpeed) -> Self {
        match speed {
            MovementSpeed::Fast => Self::fast(),
            MovementSpeed::Medium => Self::medium(),
            MovementSpeed::Slow => Self::slow(),
        }
    }
}

/// Apply locomotion interval and ecology tuning for a movement-speed preset.
/// Population counts (`initial_population`, `max_population`) are left to each
/// run config so interactive sessions can keep small founder pools.
pub fn apply_movement_speed(cfg: &mut Config, speed: MovementSpeed) {
    let tune = EcologyTuning::for_speed(speed);
    apply_ecology_tuning(cfg, &tune);
    cfg.movement_speed = Some(speed);
}

/// Plateau-80 profile: smaller populations, sharper landscape, tighter carrying capacity.
pub fn apply_plateau80_movement_speed(cfg: &mut Config, speed: MovementSpeed) {
    let tune = EcologyTuning::plateau80_for_speed(speed);
    apply_ecology_tuning(cfg, &tune);
    apply_plateau80_landscape(cfg);
    cfg.movement_speed = Some(speed);
    // Heavy architectures get gentler offsets in `apply_architecture_ecology_offset`;
    // add a final trim so spatial / GoO do not pin `max_population`.
    match cfg.architecture_policy {
        ArchitecturePolicy::EvolveSpatialGraph if cfg.uses_legacy_spatial_graph() => {
            cfg.resource_scale *= 0.88;
            cfg.reproduction_threshold += 2.0;
        }
        ArchitecturePolicy::EvolveSpatialGraph => {
            // Macromolecule blob: lighter founders; speed-tuned trim (fast burns more stores).
            let (resource_trim, repro_bump) = match speed {
                MovementSpeed::Fast => (0.92, 1.0),
                MovementSpeed::Medium => (0.94, 0.5),
                MovementSpeed::Slow => (0.96, 0.0),
            };
            cfg.resource_scale *= resource_trim;
            cfg.reproduction_threshold += repro_bump;
        }
        ArchitecturePolicy::EvolveGoo => {
            cfg.resource_scale *= 0.82;
            cfg.reproduction_threshold += 4.0;
        }
        _ => {}
    }
}

/// Sharper rich/poor patches and slightly stronger temporal swings (plateau-80 only).
pub fn apply_plateau80_landscape(cfg: &mut Config) {
    cfg.env_time_amplitude = 1.45;
    cfg.env_perlin_spatial_scale = 0.038;
    cfg.resource_contrast = 1.4;
}

/// Apply ecology knobs without changing architecture-specific fields or population size.
pub fn apply_ecology_tuning(cfg: &mut Config, tune: &EcologyTuning) {
    cfg.movement_interval = tune.movement_interval;
    cfg.movement_cost_factor = tune.movement_cost_factor;
    cfg.env_time_speed = tune.env_time_speed;
    cfg.resource_scale = tune.resource_scale;
    cfg.reproduction_threshold = tune.reproduction_threshold;
    cfg.reproduction_cost = tune.reproduction_cost;
    cfg.basal_metabolic_cost = tune.basal_metabolic_cost;
    if !cfg.lock_maintenance_cost {
        cfg.maintenance_cost_per_node = tune.maintenance_cost_per_node;
    }
    cfg.death_risk = tune.death_risk;
    apply_architecture_ecology_offset(cfg);
}

/// Compensate for higher founder `brain.complexity()` on spatial / GoO policies.
pub fn apply_architecture_ecology_offset(cfg: &mut Config) {
    let plateau80 = matches!(
        cfg.ecology_profile,
        crate::config::EcologyProfile::Plateau80
    );
    match cfg.architecture_policy {
        ArchitecturePolicy::PerceptronOnly | ArchitecturePolicy::EvolveMlp => {}
        ArchitecturePolicy::EvolveSpatialGraph if cfg.uses_legacy_spatial_graph() => {
            if plateau80 {
                cfg.resource_scale *= 1.08;
                cfg.maintenance_cost_per_node *= 0.85;
                cfg.basal_metabolic_cost *= 0.95;
                cfg.reproduction_threshold = (cfg.reproduction_threshold - 1.0).max(28.0);
                cfg.reproduction_cost = (cfg.reproduction_cost - 0.5).max(6.0);
            } else {
                cfg.resource_scale *= 1.20;
                cfg.maintenance_cost_per_node *= 0.50;
                cfg.basal_metabolic_cost *= 0.90;
                cfg.reproduction_threshold = (cfg.reproduction_threshold - 4.0).max(20.0);
                cfg.reproduction_cost = (cfg.reproduction_cost - 1.0).max(3.0);
            }
        }
        ArchitecturePolicy::EvolveSpatialGraph => {
            // Spatial macromolecule blob: low founder complexity, layer growth over time.
            if plateau80 {
                cfg.resource_scale *= 1.05;
                cfg.maintenance_cost_per_node *= 0.90;
                cfg.basal_metabolic_cost *= 0.97;
                cfg.reproduction_threshold = (cfg.reproduction_threshold - 0.5).max(28.0);
                cfg.reproduction_cost = (cfg.reproduction_cost - 0.3).max(6.0);
            } else {
                cfg.resource_scale *= 1.12;
                cfg.maintenance_cost_per_node *= 0.62;
                cfg.basal_metabolic_cost *= 0.93;
                cfg.reproduction_threshold = (cfg.reproduction_threshold - 2.0).max(20.0);
                cfg.reproduction_cost = (cfg.reproduction_cost - 0.5).max(3.0);
            }
        }
        ArchitecturePolicy::EvolveGoo => {
            if plateau80 {
                cfg.resource_scale *= 1.12;
                cfg.maintenance_cost_per_node *= 0.75;
                cfg.basal_metabolic_cost *= 0.92;
                cfg.reproduction_threshold = (cfg.reproduction_threshold - 2.0).max(26.0);
                cfg.reproduction_cost = (cfg.reproduction_cost - 0.5).max(6.0);
            } else {
                cfg.resource_scale *= 1.45;
                cfg.maintenance_cost_per_node *= 0.28;
                cfg.basal_metabolic_cost *= 0.85;
                cfg.reproduction_threshold = (cfg.reproduction_threshold - 8.0).max(18.0);
                cfg.reproduction_cost = (cfg.reproduction_cost - 1.5).max(3.0);
            }
        }
    }
}

/// Suggested founder count and safety cap for batch / pipeline runs.
pub fn population_defaults(speed: MovementSpeed) -> (i32, i32) {
    let tune = EcologyTuning::for_speed(speed);
    (tune.initial_population, tune.max_population)
}

/// Playback multiplier for interactive env-time advance, relative to the medium preset.
pub fn env_time_playback_multiplier(speed: MovementSpeed) -> f32 {
    let tune = EcologyTuning::for_speed(speed);
    let medium = EcologyTuning::medium().env_time_speed;
    (tune.env_time_speed / medium) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movement_speed_sets_interval() {
        let mut cfg = Config::default();
        cfg.initial_population = 42;
        apply_movement_speed(&mut cfg, MovementSpeed::Medium);
        assert_eq!(cfg.movement_interval, 5);
        assert_eq!(cfg.initial_population, 42);
    }

    #[test]
    fn goo_offset_reduces_maintenance() {
        let mut cfg = Config::default();
        cfg.architecture_policy = ArchitecturePolicy::EvolveGoo;
        cfg.maintenance_cost_per_node = 0.04;
        let before = cfg.maintenance_cost_per_node;
        apply_architecture_ecology_offset(&mut cfg);
        assert!(cfg.maintenance_cost_per_node < before);
        assert!(cfg.resource_scale > 1.0);
    }

    #[test]
    fn spatial_blob_gets_gentler_plateau80_trim_than_legacy_graph() {
        let mut legacy = Config::default();
        legacy.architecture_policy = ArchitecturePolicy::EvolveSpatialGraph;
        legacy.ecology_profile = crate::config::EcologyProfile::Plateau80;
        legacy.spatial_blob_layout = "legacy_graph".to_string();
        apply_plateau80_movement_speed(&mut legacy, MovementSpeed::Medium);

        let mut blob = Config::default();
        blob.architecture_policy = ArchitecturePolicy::EvolveSpatialGraph;
        blob.ecology_profile = crate::config::EcologyProfile::Plateau80;
        blob.spatial_blob_layout = "hex_edge".to_string();
        apply_plateau80_movement_speed(&mut blob, MovementSpeed::Medium);

        assert!(blob.resource_scale > legacy.resource_scale);
        assert!(blob.reproduction_threshold < legacy.reproduction_threshold);
    }
}
