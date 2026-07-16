//! Shared simulation step semantics for CPU and GPU paths.
use crate::brain::{BrainKind, BrainOutput, ForwardContext, SENSOR_DIM};
use crate::config::{Config, GridTopology};
use crate::environment::Environment;
use crate::hex_dish::HexDishLayout;
use crate::hex_simulation::{
    direction_to_sector, edge_resources_hex, edge_resources_hex_dish, edge_resources_ring,
    grid_cell_count, neighbor_flat, neighbor_flat_dish,
};
use crate::individual::{ActionDecision, Individual};
use rand::Rng;

/// Decision output from forward pass only (before apply mutates state).
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionRow {
    pub move_one_step: bool,
    pub move_signal: f64,
    pub direction_degrees: f64,
    pub movement_spend: f64,
    pub gathered: f64,
}

impl From<ActionDecision> for DecisionRow {
    fn from(d: ActionDecision) -> Self {
        DecisionRow {
            move_one_step: d.move_one_step,
            move_signal: d.move_signal,
            direction_degrees: d.direction_degrees,
            movement_spend: d.movement_spend,
            gathered: d.gathered,
        }
    }
}

impl From<DecisionRow> for ActionDecision {
    fn from(d: DecisionRow) -> Self {
        ActionDecision {
            move_one_step: d.move_one_step,
            move_signal: d.move_signal,
            direction_degrees: d.direction_degrees,
            movement_spend: d.movement_spend,
            gathered: d.gathered,
        }
    }
}

#[inline]
pub fn direction_signal_to_degrees(signal: f64) -> f64 {
    let normalized = ((signal.tanh() + 1.0) * 0.5).clamp(0.0, 1.0);
    normalized * 360.0
}

pub fn edge_resources_for_position(
    cfg: &Config,
    env: &Environment,
    pos: usize,
    cell_count: usize,
    dish: Option<&HexDishLayout>,
) -> [f64; 6] {
    if let Some(dish) = dish {
        return edge_resources_hex_dish(env, pos % cell_count, dish);
    }
    let width = cfg.grid_width.max(1);
    let height = if cfg.grid_height > 0 {
        cfg.grid_height
    } else {
        width
    };
    match cfg.grid_topology {
        GridTopology::Hex2d => edge_resources_hex(env, pos % cell_count, width, height),
        GridTopology::Ring1d => edge_resources_ring(env, pos % cell_count),
    }
}

pub fn build_sensor_inputs(
    edge_resources: [f64; 6],
    local_population: i32,
    individual_resources: f64,
) -> [f64; SENSOR_DIM] {
    let mut stack_inputs = [0.0f64; SENSOR_DIM];
    stack_inputs[0] = edge_resources[0];
    stack_inputs[1] = edge_resources[1];
    stack_inputs[2] = edge_resources[2];
    stack_inputs[3] = edge_resources[3];
    stack_inputs[4] = edge_resources[4];
    stack_inputs[5] = edge_resources[5];
    stack_inputs[6] = local_population as f64;
    stack_inputs[7] = individual_resources;
    stack_inputs
}

/// Perceptron sensors: current hex resource, local competition, internal stores.
pub fn build_perceptron_sensor_inputs(
    center_resource: f64,
    local_population: i32,
    individual_resources: f64,
) -> [f64; SENSOR_DIM] {
    let mut stack_inputs = [0.0f64; SENSOR_DIM];
    stack_inputs[0] = center_resource;
    stack_inputs[1] = local_population as f64;
    stack_inputs[2] = individual_resources;
    stack_inputs
}

pub fn build_sensor_inputs_for_kind(
    kind: BrainKind,
    edge_resources: [f64; 6],
    center_resource: f64,
    local_population: i32,
    individual_resources: f64,
) -> [f64; SENSOR_DIM] {
    match kind {
        BrainKind::Perceptron => build_perceptron_sensor_inputs(
            center_resource,
            local_population,
            individual_resources,
        ),
        _ => build_sensor_inputs(edge_resources, local_population, individual_resources),
    }
}

/// Run brain forward and compute decision without mutating resources/position/env.
pub fn forward_decision(
    ind: &mut Individual,
    edge_resources: [f64; 6],
    center_resource: f64,
    local_population: i32,
    movement_cost_factor: f64,
    hidden_buf: &mut [f64],
    out2_buf: &mut [f64],
) -> DecisionRow {
    let sensors = build_sensor_inputs_for_kind(
        ind.brain.kind(),
        edge_resources,
        center_resource,
        local_population,
        ind.resources,
    );

    let ctx = ForwardContext {
        sensors,
        position: ind.position,
        hidden_buf,
        out2_buf,
        spatial_runtime: ind.spatial_runtime.as_mut(),
    };
    let output = ind.brain.forward(ctx);

    decision_from_output(
        output,
        center_resource,
        local_population,
        ind.resources,
        movement_cost_factor,
    )
}

pub fn decision_from_output(
    output: BrainOutput,
    center: f64,
    local_population: i32,
    individual_resources: f64,
    movement_cost_factor: f64,
) -> DecisionRow {
    let direction_degrees = direction_signal_to_degrees(output.direction_signal);
    if output.move_signal <= 0.0 {
        let denom = (local_population.max(1)) as f64;
        let gathered = center / denom;
        return DecisionRow {
            move_one_step: false,
            move_signal: output.move_signal,
            direction_degrees,
            movement_spend: 0.0,
            gathered,
        };
    }

    let spend = output.move_signal.abs() * movement_cost_factor;
    let movement_spend = if spend.is_nan() {
        0.0
    } else {
        spend.min(individual_resources)
    };
    DecisionRow {
        move_one_step: true,
        move_signal: output.move_signal,
        direction_degrees,
        movement_spend,
        gathered: 0.0,
    }
}

/// Apply a decision to individual state and optionally deplete environment.
pub fn apply_decision(
    ind: &mut Individual,
    decision: DecisionRow,
    cfg: &Config,
    env: &mut Environment,
    cell_count: usize,
    occupancy: &mut [i32],
    dish: Option<&HexDishLayout>,
) -> i32 {
    let pos = ind.position % cell_count;
    let interval = cfg.movement_interval.max(1);
    let local_pop = occupancy.get(pos).copied().unwrap_or(1).max(1);

    if decision.move_one_step {
        if ind.move_cooldown > 0 {
            ind.move_cooldown -= 1;
            apply_gather_at(ind, decision, env, pos, local_pop);
            return 0;
        }
        ind.resources -= decision.movement_spend;
        let new_pos = compute_new_position(pos, decision.direction_degrees, cfg, cell_count, dish);
        let delta = new_pos as i32 - pos as i32;
        ind.position = new_pos;
        occupancy[pos] = occupancy[pos].saturating_sub(1);
        occupancy[new_pos % cell_count] = occupancy[new_pos % cell_count].saturating_add(1);
        if interval > 1 {
            ind.move_cooldown = interval - 1;
        }
        delta
    } else {
        apply_gather_at(ind, decision, env, pos, local_pop);
        0
    }
}

fn apply_gather_at(
    ind: &mut Individual,
    decision: DecisionRow,
    env: &mut Environment,
    pos: usize,
    local_pop: i32,
) {
    let gathered = gather_amount_for_decision(decision, env, pos, local_pop);
    if gathered > 0.0 {
        ind.resources += gathered;
        if let Some(slot) = env.resources.get_mut(pos) {
            *slot = (*slot - gathered).max(0.0);
        }
    }
}

/// Resources gathered when foraging at `pos` (explicit gather or cooldown rest-forage).
pub fn gather_amount_for_decision(
    decision: DecisionRow,
    env: &Environment,
    pos: usize,
    local_pop: i32,
) -> f64 {
    if decision.gathered > 0.0 {
        return decision.gathered;
    }
    if decision.move_one_step {
        let center = env.resources.get(pos).copied().unwrap_or(0.0);
        return center / (local_pop.max(1) as f64);
    }
    0.0
}

/// Whether an individual may execute a hex move this step.
pub fn move_allowed(ind: &Individual, wants_move: bool, cfg: &Config) -> bool {
    wants_move && ind.move_cooldown == 0 && cfg.movement_interval.max(1) >= 1
}

/// Set cooldown after a successful hex move.
pub fn start_move_cooldown(ind: &mut Individual, cfg: &Config) {
    let interval = cfg.movement_interval.max(1);
    if interval > 1 {
        ind.move_cooldown = interval - 1;
    }
}

/// Tick cooldown while the brain signals move but cannot relocate yet.
pub fn tick_move_wait(ind: &mut Individual) {
    if ind.move_cooldown > 0 {
        ind.move_cooldown -= 1;
    }
}

pub fn compute_new_position(
    pos: usize,
    direction_degrees: f64,
    cfg: &Config,
    cell_count: usize,
    dish: Option<&HexDishLayout>,
) -> usize {
    if let Some(dish) = dish {
        let sector = direction_to_sector(direction_degrees);
        return neighbor_flat_dish(pos, sector, dish);
    }
    let width = cfg.grid_width.max(1);
    let height = if cfg.grid_height > 0 {
        cfg.grid_height
    } else {
        width
    };
    match cfg.grid_topology {
        GridTopology::Hex2d => {
            let sector = direction_to_sector(direction_degrees);
            neighbor_flat(pos, sector, width, height)
        }
        GridTopology::Ring1d => {
            let dir = direction_degrees.rem_euclid(360.0);
            let delta = if dir < 180.0 { 1i32 } else { -1i32 };
            ((pos as i32 + delta).rem_euclid(cell_count as i32)) as usize
        }
    }
}

pub struct StepOutcome {
    pub last_spend: Vec<f64>,
    pub last_delta: Vec<i32>,
    pub last_movers: usize,
}

/// Full CPU step using shared semantics (canonical path).
pub fn step_individuals_cpu(
    individuals: &mut [Individual],
    env: &mut Environment,
    cfg: &Config,
    hidden_buf: &mut [f64],
    out2_buf: &mut [f64],
    _rng: &mut impl Rng,
    dish: Option<&HexDishLayout>,
) -> StepOutcome {
    let cell_count = grid_cell_count(cfg);
    let mut occupancy = vec![0i32; cell_count];
    for ind in individuals.iter() {
        let pos = ind.position % cell_count;
        occupancy[pos] = occupancy[pos].saturating_add(1);
    }

    let n = individuals.len();
    let mut decisions = Vec::with_capacity(n);
    for i in 0..n {
        let pos = individuals[i].position % cell_count;
        let local_pop = occupancy[pos];
        let edges = edge_resources_for_position(cfg, env, pos, cell_count, dish);
        let center = env.resources.get(pos).copied().unwrap_or(0.0);
        decisions.push(forward_decision(
            &mut individuals[i],
            edges,
            center,
            local_pop,
            cfg.movement_cost_factor,
            hidden_buf,
            out2_buf,
        ));
    }

    apply_decisions_sequential(individuals, &decisions, cfg, env, dish)
}

/// Apply precomputed decisions in index order (GPU forward + CPU apply path).
pub fn apply_decisions_sequential(
    individuals: &mut [Individual],
    decisions: &[DecisionRow],
    cfg: &Config,
    env: &mut Environment,
    dish: Option<&HexDishLayout>,
) -> StepOutcome {
    let cell_count = grid_cell_count(cfg);
    let mut occupancy = vec![0i32; cell_count];
    for ind in individuals.iter() {
        let pos = ind.position % cell_count;
        occupancy[pos] = occupancy[pos].saturating_add(1);
    }

    let n = individuals.len().min(decisions.len());
    let mut last_spend = vec![0.0; n];
    let mut last_delta = vec![0; n];
    let mut last_movers = 0usize;

    for i in 0..n {
        if decisions[i].move_one_step {
            last_movers += 1;
            last_spend[i] = decisions[i].movement_spend;
        }
        last_delta[i] = apply_decision(
            &mut individuals[i],
            decisions[i],
            cfg,
            env,
            cell_count,
            &mut occupancy,
            dish,
        );
    }

    StepOutcome {
        last_spend,
        last_delta,
        last_movers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{BrainGenome, BrainInner, PerceptronBrain};

    #[test]
    fn gather_decision_does_not_move() {
        let brain = BrainGenome {
            kind: crate::brain::BrainKind::Perceptron,
            inner: BrainInner::Perceptron(PerceptronBrain::new(3, 1)),
        };
        let mut ind = Individual::new(brain, 0, 1);
        let edges = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0];
        let mut hb = [0.0; 4];
        let mut ob = [0.0; 4];
        let d = forward_decision(&mut ind, edges, 2.0, 1, 0.5, &mut hb, &mut ob);
        assert!(!d.move_one_step || d.move_signal > 0.0);
    }

    #[test]
    fn movement_interval_blocks_rapid_moves() {
        use crate::config::Config;
        use crate::environment::Environment;

        let mut cfg = Config::default();
        cfg.movement_interval = 3;
        cfg.grid_width = 10;
        cfg.movement_cost_factor = 0.1;

        let mut env = Environment::new(10);
        env.resources = vec![2.0; 10];

        let brain = BrainGenome {
            kind: crate::brain::BrainKind::Perceptron,
            inner: BrainInner::Perceptron(PerceptronBrain::new(3, 1)),
        };
        let mut ind = Individual::new(brain, 0, 1);
        ind.resources = 50.0;

        let decision = DecisionRow {
            move_one_step: true,
            move_signal: 1.0,
            direction_degrees: 0.0,
            movement_spend: 0.1,
            gathered: 0.0,
        };
        let mut occupancy = vec![1i32; 10];

        apply_decision(&mut ind, decision, &cfg, &mut env, 10, &mut occupancy, None);
        assert_eq!(ind.position, 1);
        assert_eq!(ind.move_cooldown, 2);

        apply_decision(&mut ind, decision, &cfg, &mut env, 10, &mut occupancy, None);
        assert_eq!(ind.position, 1);
        assert_eq!(ind.move_cooldown, 1);

        apply_decision(&mut ind, decision, &cfg, &mut env, 10, &mut occupancy, None);
        assert_eq!(ind.position, 1);
        assert_eq!(ind.move_cooldown, 0);

        apply_decision(&mut ind, decision, &cfg, &mut env, 10, &mut occupancy, None);
        assert_eq!(ind.position, 2);
        assert_eq!(ind.move_cooldown, 2);
    }
}
