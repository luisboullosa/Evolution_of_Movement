//! Hex dish state for the lab interactive simulator (single-dish, manual controls).

use crate::brain_playground::random_founder_for_policy;
use crate::cladogram::Cladogram;
use crate::config::{ArchitecturePolicy, Config, GridTopology};
use crate::hex_grid::{HexCell, HexCoord, HexGrid};
use crate::individual::Individual;
use crate::perlin::Perlin;
use crate::simulation::Simulation;
use crate::step_engine::{
    forward_decision, gather_amount_for_decision, move_allowed, start_move_cooldown, tick_move_wait,
};
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};

const GOLDEN_ANGLE: f32 = 137.508;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = h.rem_euclid(360.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

pub fn genotype_signature(ind: &Individual) -> String {
    ind.brain.structure_signature()
}

pub fn genotype_color(sig: &str) -> [u8; 3] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    sig.hash(&mut hasher);
    let hue = (hasher.finish() % 360) as f32;
    hsv_to_rgb(hue * GOLDEN_ANGLE % 360.0, 0.65, 0.85)
}

pub struct LabDish {
    pub hex_grid: HexGrid,
    pub simulation: Simulation,
    pub cladogram: Cladogram,
    pub timestep: i32,
    pub radius_cells: i32,
    pub active_coords: Vec<HexCoord>,
    coord_to_index: HashMap<HexCoord, usize>,
    dish_center: HexCoord,
    resource_base: Vec<f64>,
    pub last_avg_move_signal: f64,
}

impl LabDish {
    pub fn new(radius_cells: i32, hex_size: f32, mut config: Config) -> Self {
        let radius_cells = radius_cells.max(1);
        let hex_grid = HexGrid::new_hex_dish(radius_cells, hex_size);
        let mut active_coords: Vec<HexCoord> = hex_grid.cells.keys().copied().collect();
        active_coords.sort_by_key(|c| (c.r, c.q));
        let mut coord_to_index = HashMap::with_capacity(active_coords.len());
        for (idx, coord) in active_coords.iter().enumerate() {
            coord_to_index.insert(*coord, idx);
        }
        let dish_center = HexCoord::new(radius_cells, radius_cells);

        config.grid_topology = GridTopology::Hex2d;
        if config.architecture_policy == ArchitecturePolicy::PerceptronOnly {
            config.nn_inputs = crate::brain::PERCEPTRON_SENSOR_DIM as i32;
        } else {
            config.nn_inputs = config.nn_inputs.max(8);
        }
        config.grid_width = active_coords.len().max(1);
        config.hex_dish_radius = 0;

        let seed = config.seed as u64;
        let spatial_scale = config.env_perlin_spatial_scale;
        let time_scale = config.env_perlin_time_scale;
        let mut resource_base = Vec::with_capacity(active_coords.len());
        for coord in &active_coords {
            let v = Perlin::sample_2d_time(
                coord.q as f64,
                coord.r as f64,
                0.0,
                seed,
                spatial_scale,
                time_scale,
            );
            resource_base.push(v);
        }

        let mut dish = Self {
            hex_grid,
            simulation: Simulation::new(&config),
            cladogram: Cladogram::new(),
            timestep: 0,
            radius_cells,
            active_coords,
            coord_to_index,
            dish_center,
            resource_base,
            last_avg_move_signal: 0.0,
        };
        dish.update_resource_grid();
        dish
    }

    pub fn cell_count(&self) -> usize {
        self.active_coords.len().max(1)
    }

    pub fn reset(&mut self, config: Config) {
        let radius = self.radius_cells;
        let hex_size = self.hex_grid.hex_size;
        *self = Self::new(radius, hex_size, config);
    }

    fn cells_within_insert_radius(&self, insert_radius: i32) -> Vec<usize> {
        let max_dist = if insert_radius <= 0 {
            self.radius_cells
        } else {
            insert_radius.min(self.radius_cells)
        };
        self.active_coords
            .iter()
            .enumerate()
            .filter(|(_, c)| c.distance_to(self.dish_center) <= max_dist)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn insert_individuals(
        &mut self,
        count: usize,
        architecture: ArchitecturePolicy,
        insert_radius: i32,
        starting_resources: f64,
    ) {
        if count == 0 {
            return;
        }
        let candidates = self.cells_within_insert_radius(insert_radius);
        if candidates.is_empty() {
            return;
        }
        let mut cfg = self.simulation.config.clone();
        cfg.architecture_policy = architecture;

        for i in 0..count {
            let seed = self
                .simulation
                .config
                .seed
                .wrapping_add(self.simulation.next_id as u32)
                .wrapping_add(i as u32);
            let brain = random_founder_for_policy(architecture, seed, &cfg);
            let pos = *candidates
                .choose(&mut self.simulation.rng)
                .unwrap_or(&0);
            let mut ind = Individual::new(brain, pos, self.simulation.next_id);
            ind.resources = starting_resources;
            ind.parent_id = None;
            let sig = genotype_signature(&ind);
            ind.lineage_color = genotype_color(&sig);
            self.cladogram.add_founder(ind.id, ind.lineage_color);
            self.simulation.next_id += 1;
            self.simulation.individuals.push(ind);
        }
        self.rebuild_hex_occupancy();
    }

    pub fn clear_population(&mut self) {
        self.simulation.individuals.clear();
        self.cladogram = Cladogram::new();
        self.rebuild_hex_occupancy();
    }

    pub fn update_resource_grid(&mut self) {
        let seed = self.simulation.config.seed as u64;
        let spatial_scale = self.simulation.config.env_perlin_spatial_scale;
        let time_scale = self.simulation.config.env_perlin_time_scale;
        let time_amplitude = self.simulation.config.env_time_amplitude;
        let resource_scale = self.simulation.config.resource_scale;

        for (idx, coord) in self.active_coords.iter().enumerate() {
            let temporal = Perlin::sample_2d_time(
                coord.q as f64,
                coord.r as f64,
                self.simulation.env.perlin_time,
                seed,
                spatial_scale,
                time_scale,
            );
            let centered_temporal = temporal - 0.5;
            let base = self.resource_base.get(idx).copied().unwrap_or(0.0);
            let resource = ((base + centered_temporal * time_amplitude) * resource_scale).max(0.0);

            if let Some(slot) = self.simulation.env.resources.get_mut(idx) {
                *slot = resource;
            }
            if let Some(cell) = self.hex_grid.get_cell_mut(*coord) {
                cell.resource_level = resource;
            }
        }
    }

    pub fn rebuild_hex_occupancy(&mut self) {
        for cell in self.hex_grid.cells.values_mut() {
            cell.individual_indices.clear();
        }
        let width = self.active_coords.len().max(1);
        for (idx, ind) in self.simulation.individuals.iter_mut().enumerate() {
            let pos = ind.position % width;
            ind.position = pos;
            let coord = self.active_coords[pos];
            if let Some(cell) = self.hex_grid.get_cell_mut(coord) {
                cell.individual_indices.push(idx);
            }
        }
    }

    pub fn sync_config(&mut self) {
        let w = self.active_coords.len().max(1);
        self.simulation.config.grid_width = w;
    }

    pub fn step(&mut self, move_threshold: f64, env_speed_multiplier: f32, auto_reproduce: bool) {
        if self.active_coords.is_empty() {
            return;
        }

        let before_ids: HashSet<i32> = self
            .simulation
            .individuals
            .iter()
            .map(|ind| ind.id)
            .collect();

        self.timestep += 1;
        self.cladogram.advance_time();

        let dt = (self.simulation.config.env_time_speed * env_speed_multiplier as f64).max(0.0);
        self.simulation.env.perlin_time += dt;
        self.update_resource_grid();

        let cell_count = self.active_coords.len();
        let mut move_signal_sum = 0.0;
        let mut move_signal_count = 0usize;

        for i in 0..self.simulation.individuals.len() {
            let pos = self.simulation.individuals[i].position % cell_count;
            self.simulation.individuals[i].position = pos;
            let coord = self.active_coords[pos];
            let neighbors = coord.neighbors();
            let mut edge_resources = [0.0f64; 6];
            for (sector, ncoord) in neighbors.iter().enumerate() {
                if let Some(&nidx) = self.coord_to_index.get(ncoord) {
                    edge_resources[sector] = self.simulation.env.resources.get(nidx).copied().unwrap_or(0.0);
                }
            }
            let local_population = self
                .hex_grid
                .get_cell(coord)
                .map(|c| c.individual_indices.len() as i32)
                .unwrap_or(0);

            let center = self
                .simulation
                .env
                .resources
                .get(pos)
                .copied()
                .unwrap_or(0.0);
            let ind = &mut self.simulation.individuals[i];
            let decision = forward_decision(
                ind,
                edge_resources,
                center,
                local_population,
                self.simulation.config.movement_cost_factor,
                self.simulation.config.resolved_movement_output_mode(),
                &mut self.simulation.nn_hidden_buf,
                &mut self.simulation.nn_out2_buf,
            );
            move_signal_sum += decision.move_signal;
            move_signal_count = move_signal_count.saturating_add(1);

            let wants_move = decision.move_one_step && decision.move_signal > move_threshold;
            if wants_move && !move_allowed(ind, true, &self.simulation.config) {
                tick_move_wait(ind);
                let gathered = gather_amount_for_decision(
                    decision,
                    &self.simulation.env,
                    pos,
                    local_population.max(1),
                );
                if gathered > 0.0 {
                    ind.resources += gathered;
                    if let Some(resource_slot) = self.simulation.env.resources.get_mut(pos) {
                        *resource_slot = (*resource_slot - gathered).max(0.0);
                    }
                }
            } else if wants_move {
                ind.resources -= decision.movement_spend;
                let mut sector =
                    ((decision.direction_degrees.rem_euclid(360.0) / 60.0).floor() as usize) % 6;
                if sector >= 6 {
                    sector = 0;
                }
                let target_coord = neighbors[sector];
                if let Some(target_idx) = self.coord_to_index.get(&target_coord).copied() {
                    ind.position = target_idx;
                    start_move_cooldown(ind, &self.simulation.config);
                }
            } else if decision.gathered > 0.0 {
                ind.resources += decision.gathered;
                if let Some(resource_slot) = self.simulation.env.resources.get_mut(pos) {
                    *resource_slot = (*resource_slot - decision.gathered).max(0.0);
                }
            }
        }

        self.last_avg_move_signal = if move_signal_count > 0 {
            move_signal_sum / move_signal_count as f64
        } else {
            0.0
        };

        for ind in &mut self.simulation.individuals {
            ind.age += 1;
        }

        if auto_reproduce {
            self.reproduce_eligible();
        }

        let dead_ids = crate::population_dynamics::apply_metabolism_and_cull(
            &mut self.simulation.individuals,
            &self.simulation.config,
            &mut self.simulation.rng,
        );

        let _after_ids: HashSet<i32> = self
            .simulation
            .individuals
            .iter()
            .map(|ind| ind.id)
            .collect();

        for ind in &self.simulation.individuals {
            if !before_ids.contains(&ind.id) {
                if let Some(parent_id) = ind.parent_id {
                    if self.cladogram.nodes.contains_key(&parent_id) {
                        let genetic_distance_to_parent = self
                            .simulation
                            .individuals
                            .iter()
                            .find(|p| p.id == parent_id)
                            .map(|p| Cladogram::genetic_distance_brain(&p.brain, &ind.brain))
                            .unwrap_or(0.0);
                        self.cladogram.add_offspring(
                            parent_id,
                            ind.id,
                            ind.lineage_color,
                            genetic_distance_to_parent,
                        );
                    } else {
                        self.cladogram.add_founder(ind.id, ind.lineage_color);
                    }
                } else {
                    self.cladogram.add_founder(ind.id, ind.lineage_color);
                }
            }
        }
        for dead_id in dead_ids {
            self.cladogram.mark_death(dead_id);
        }

        self.rebuild_hex_occupancy();
    }

    pub fn reproduce_eligible(&mut self) -> usize {
        use crate::population_dynamics::run_reproduction_phase;
        let born = run_reproduction_phase(
            &mut self.simulation.individuals,
            &self.simulation.config,
            &mut self.simulation.rng,
            &mut self.simulation.next_id,
            |child| {
                let sig = genotype_signature(child);
                child.lineage_color = genotype_color(&sig);
            },
        )
        .len();
        if born > 0 {
            self.rebuild_hex_occupancy();
        }
        born
    }
}

pub fn max_cell_occupancy(cells: &HashMap<HexCoord, HexCell>) -> usize {
    cells
        .values()
        .map(|c| c.individual_indices.len())
        .max()
        .unwrap_or(0)
}

pub fn resource_background_color(level: f64) -> [u8; 3] {
    let t = (level / 2.0).clamp(0.0, 1.0) as f32;
    let v = (40.0 + t * 180.0) as u8;
    [v, (v as f32 * 0.92) as u8, (v as f32 * 0.55) as u8]
}
