//! Shared reproduction, metabolism, and population-cap logic for headless and interactive.

use crate::architecture_lineage::ArchitectureLineageEvent;
use crate::brain::StructuralMutation;
use crate::cladogram::Cladogram;
use crate::config::Config;
use crate::individual::Individual;
use rand::rngs::StdRng;
use rand::Rng;

/// True when no more births are allowed this step.
pub fn pop_cap_reached(population_len: usize, max_population: i32) -> bool {
    max_population > 0 && (population_len as i32) >= max_population
}

/// Clamp requested founder count to `max_population` when capped.
pub fn clamp_initial_population(requested: usize, max_population: i32) -> usize {
    if max_population > 0 {
        requested.min(max_population as usize)
    } else {
        requested
    }
}

pub struct BirthRecord {
    pub parent: Individual,
    pub child: Individual,
    pub genetic_distance: f64,
    pub structural_mutation: StructuralMutation,
}

/// Run reproduction for all eligible individuals (index order) until pop cap.
pub fn run_reproduction_phase(
    individuals: &mut Vec<Individual>,
    config: &Config,
    rng: &mut StdRng,
    next_id: &mut i32,
    mut customize_child: impl FnMut(&mut Individual),
) -> Vec<BirthRecord> {
    let threshold = config.reproduction_threshold;
    let mut records = Vec::new();
    let scan_count = individuals.len();
    for idx in 0..scan_count {
        if idx >= individuals.len() {
            break;
        }
        if pop_cap_reached(individuals.len(), config.max_population) {
            break;
        }
        if individuals[idx].resources < threshold {
            continue;
        }
        let parent = individuals[idx].clone();
        let Some(repro) = individuals[idx].reproduce(
            rng,
            next_id,
            threshold,
            config.reproduction_cost,
            config,
        ) else {
            continue;
        };
        let mut child = repro.child;
        customize_child(&mut child);
        records.push(BirthRecord {
            parent,
            child: child.clone(),
            genetic_distance: repro.genetic_distance,
            structural_mutation: repro.structural_mutation,
        });
        individuals.push(child);
    }
    records
}

pub fn apply_metabolism_and_cull(
    individuals: &mut Vec<Individual>,
    config: &Config,
    rng: &mut StdRng,
) -> Vec<i32> {
    for ind in individuals.iter_mut() {
        let total_nodes = ind.complexity();
        ind.resources -= config.basal_metabolic_cost
            + (total_nodes as f64) * config.maintenance_cost_per_node;
    }

    use rand::distributions::{Distribution, Uniform};
    let ud = Uniform::new(0.0f64, 1.0f64);
    let mut dead_ids = Vec::new();
    individuals.retain(|ind| {
        if ind.resources <= 0.0 {
            dead_ids.push(ind.id);
            return false;
        }
        if config.death_risk > 0.0 && ind.age > 0 {
            let p = 1.0 - (-(config.death_risk * (ind.age as f64))).exp();
            if ud.sample(rng) < p {
                dead_ids.push(ind.id);
                return false;
            }
        }
        true
    });
    dead_ids
}

pub fn record_births_in_cladogram(clado: &mut Cladogram, records: &[BirthRecord]) {
    for birth in records {
        clado.add_offspring(
            birth.parent.id,
            birth.child.id,
            birth.child.lineage_color,
            birth.genetic_distance,
        );
    }
}

pub fn record_births_in_lineage(
    lineage: &mut Vec<ArchitectureLineageEvent>,
    step_count: i32,
    records: &[BirthRecord],
) {
    for birth in records {
        lineage.push(ArchitectureLineageEvent::from_reproduction(
            step_count,
            &birth.parent,
            &birth.child,
            birth.genetic_distance,
            &birth.structural_mutation,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::BrainGenome;
    use crate::config::ArchitecturePolicy;
    use rand::SeedableRng;

    #[test]
    fn pop_cap_blocks_births() {
        let mut cfg = Config::default();
        cfg.max_population = 2;
        cfg.reproduction_threshold = 0.0;
        cfg.reproduction_cost = 0.0;
        cfg.architecture_policy = ArchitecturePolicy::PerceptronOnly;
        let mut rng = StdRng::seed_from_u64(1);
        let mut next_id = 10;
        let mut inds = vec![
            Individual::new(BrainGenome::new_founder(&cfg), 0, 1),
            Individual::new(BrainGenome::new_founder(&cfg), 0, 2),
        ];
        for ind in &mut inds {
            ind.resources = 100.0;
        }
        let births = run_reproduction_phase(&mut inds, &cfg, &mut rng, &mut next_id, |_| {});
        assert_eq!(births.len(), 0);
        assert_eq!(inds.len(), 2);
    }

    #[test]
    fn clamp_initial_respects_cap() {
        assert_eq!(clamp_initial_population(10_000, 12000), 10_000);
        assert_eq!(clamp_initial_population(15_000, 12000), 12000);
        assert_eq!(clamp_initial_population(15_000, 0), 15_000);
    }
}
