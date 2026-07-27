//! Random brain generation and JSON helpers for the WASM playground.

use crate::brain::goo_graph::GooGraphBrain;

use crate::brain::mlp::MlpBrain;

use crate::brain::mutation::init_founder_weights;

use crate::brain::spatial_graph::SpatialGraphBrain;

use crate::brain::{BrainGenome, BrainInner, BrainKind, BrainOutput, ForwardContext, GooOp, SENSOR_DIM};

use crate::brain_diagram::{diagram_from_brain, BrainDiagram};

use crate::config::{ArchitecturePolicy, Config, MovementOutputMode};

use rand::rngs::StdRng;

use rand::{Rng, SeedableRng};



pub fn parse_brain_json(json: &str) -> Result<BrainGenome, String> {

    serde_json::from_str(json).map_err(|e| e.to_string())

}



pub fn brain_to_json(brain: &BrainGenome) -> Result<String, String> {

    serde_json::to_string_pretty(brain).map_err(|e| e.to_string())

}



pub fn brain_to_diagram_json(brain: &BrainGenome, width: f32, height: f32) -> Result<String, String> {

    let diagram = diagram_from_brain(brain, width, height);

    serde_json::to_string(&diagram).map_err(|e| e.to_string())

}



pub fn diagram_for_brain_json(

    brain_json: &str,

    width: f32,

    height: f32,

) -> Result<BrainDiagram, String> {

    let brain = parse_brain_json(brain_json)?;

    Ok(diagram_from_brain(&brain, width, height))

}



pub fn random_perceptron_brain(seed: u32, inputs: usize, nodes: usize) -> BrainGenome {

    let mut rng = StdRng::seed_from_u64(seed as u64);

    let inputs = inputs.clamp(1, 8);

    let nodes = nodes.max(1);

    let mut brain = BrainGenome {

        kind: BrainKind::Perceptron,

        inner: BrainInner::Perceptron(crate::brain::PerceptronBrain::new(inputs, nodes)),

    };

    init_founder_weights(&mut brain, &mut rng, &Config::default());

    brain

}



pub fn random_mlp_brain(seed: u32, hidden: usize, two_layer: bool) -> BrainGenome {

    let mut rng = StdRng::seed_from_u64(seed as u64);

    let hidden = hidden.clamp(1, 12);

    let layers = if two_layer { 2 } else { 1 };

    let nodes2 = if two_layer { rng.gen_range(2..6) } else { 0 };

    let mut brain = BrainGenome {

        kind: BrainKind::Mlp,

        inner: BrainInner::Mlp(MlpBrain::from_config(hidden, 8, layers, nodes2)),

    };

    init_founder_weights(&mut brain, &mut rng, &Config::default());

    brain

}



pub fn random_spatial_brain(seed: u32, landmarks: usize, edges: usize) -> BrainGenome {

    let mut rng = StdRng::seed_from_u64(seed as u64);

    let mut sg = SpatialGraphBrain::new(2);

    let landmarks = landmarks.clamp(2, 12);

    while sg.landmark_count < landmarks {

        sg.add_landmark(&mut rng);

    }

    for _ in 0..edges {

        sg.add_edge(&mut rng);

    }

    BrainGenome {

        kind: BrainKind::SpatialGraph,

        inner: BrainInner::SpatialGraph(sg),

    }

}



fn grow_goo_graph(rng: &mut StdRng, target_nodes: usize) -> GooGraphBrain {
    let mut goo = GooGraphBrain::new_initial();
    goo.grow_to_size(rng, target_nodes);
    goo
}



/// Grow a GoO graph using the same structural ops as `mutation::apply_goo_structural`.

pub fn random_goo_brain(seed: u32, target_nodes: usize) -> BrainGenome {

    let mut rng = StdRng::seed_from_u64(seed as u64);

    BrainGenome {

        kind: BrainKind::GooGraph,

        inner: BrainInner::GooGraph(grow_goo_graph(&mut rng, target_nodes)),

    }

}



/// Random founder brain for a given architecture policy (lab / playground seeding).
pub fn random_founder_for_policy(
    policy: ArchitecturePolicy,
    seed: u32,
    cfg: &Config,
) -> BrainGenome {
    match policy {
        ArchitecturePolicy::PerceptronOnly => {
            let inputs = cfg.nn_inputs.max(8) as usize;
            let nodes = cfg.nn_nodes.max(1) as usize;
            random_perceptron_brain(seed, inputs, nodes)
        }
        ArchitecturePolicy::EvolveMlp => {
            random_mlp_brain(seed, cfg.nn_nodes.max(1) as usize, cfg.nn_layers >= 2)
        }
        ArchitecturePolicy::EvolveSpatialGraph => {
            random_spatial_brain(seed, 4, 6)
        }
        ArchitecturePolicy::EvolveGoo => {
            random_goo_brain(seed, cfg.max_brain_nodes.max(4) as usize)
        }
    }
}



/// Hand-built branching/merging example (GoT-style DAG, no self-loops).

pub fn demo_goo_branching_brain() -> BrainGenome {

    let mut goo = GooGraphBrain::new_initial();

    let mut rng = StdRng::seed_from_u64(0);

    goo.add_generate_node(&mut rng, Some(0)); // branch A

    goo.add_generate_node(&mut rng, Some(0)); // branch B from root

    goo.add_aggregate_node(&mut rng, vec![1, 2]);

    goo.add_score_node(1);

    goo.add_score_node(2);

    goo.add_keep_best_node(vec![4, 5]);

    BrainGenome {

        kind: BrainKind::GooGraph,

        inner: BrainInner::GooGraph(goo),

    }

}



pub fn goo_cycle_node_op(brain: &mut BrainGenome, node_id: usize) -> bool {

    let BrainInner::GooGraph(goo) = &mut brain.inner else {

        return false;

    };

    let Some(node) = goo.nodes.iter_mut().find(|n| n.id == node_id) else {

        return false;

    };

    node.op = match node.op {

        GooOp::Generate => GooOp::Score,

        GooOp::Score => GooOp::Aggregate,

        GooOp::Aggregate => GooOp::KeepBest,

        GooOp::KeepBest => GooOp::Generate,

    };

    true

}



pub fn goo_add_structural_node(brain: &mut BrainGenome, op: &str) -> Result<(), String> {

    let BrainInner::GooGraph(goo) = &mut brain.inner else {

        return Err("brain is not GoO graph".to_string());

    };

    let mut rng = rand::thread_rng();

    let n = goo.nodes.len();

    match op {

        "generate" => {

            let parent_id = if n > 0 {

                Some(rng.gen_range(0..n))

            } else {

                None

            };

            goo.add_generate_node(&mut rng, parent_id);

        }

        "aggregate" => {

            if n < 2 {

                return Err("need at least 2 nodes for aggregate".to_string());

            }

            let parents = GooGraphBrain::sample_parent_ids(&mut rng, n, 2);

            goo.add_aggregate_node(&mut rng, parents);

        }

        "score" => {

            if n < 1 {

                return Err("need at least 1 node for score".to_string());

            }

            goo.add_score_node(rng.gen_range(0..n));

        }

        "keep_best" => {

            if n < 2 {

                return Err("need at least 2 nodes for keep_best".to_string());

            }

            let parents = GooGraphBrain::sample_parent_ids(&mut rng, n, 2);

            goo.add_keep_best_node(parents);

        }

        other => return Err(format!("unknown op {other}")),

    }

    Ok(())

}



#[derive(Debug, Clone, serde::Serialize)]
pub struct ProbeDecision {
    pub move_one_step: bool,
    pub move_signal: f64,
    pub direction_degrees: f64,
    pub movement_spend: f64,
    pub gathered: f64,
}

fn parse_movement_output_mode(s: &str) -> Result<MovementOutputMode, String> {
    match s {
        "separate_readout" | "separate" => Ok(MovementOutputMode::SeparateReadout),
        "linear_readout" | "linear" => Ok(MovementOutputMode::LinearReadout),
        "hidden_sum" | "hidden" | "sum" => Ok(MovementOutputMode::HiddenSum),
        "hex_compass" | "compass" => Ok(MovementOutputMode::HexCompass),
        "split_channels" | "split" => Ok(MovementOutputMode::SplitChannels),
        other => Err(format!("unknown movement_output_mode: {other}")),
    }
}

#[inline]
fn direction_signal_to_degrees(signal: f64) -> f64 {
    let normalized = ((signal.tanh() + 1.0) * 0.5).clamp(0.0, 1.0);
    normalized * 360.0
}

fn probe_decision_from_output(
    output: BrainOutput,
    center: f64,
    local_population: i32,
    individual_resources: f64,
    movement_cost_factor: f64,
) -> ProbeDecision {
    let direction_degrees = direction_signal_to_degrees(output.direction_signal);
    if output.move_signal <= 0.0 {
        let denom = (local_population.max(1)) as f64;
        return ProbeDecision {
            move_one_step: false,
            move_signal: output.move_signal,
            direction_degrees,
            movement_spend: 0.0,
            gathered: center / denom,
        };
    }
    let spend = output.move_signal.abs() * movement_cost_factor;
    let movement_spend = if spend.is_nan() {
        0.0
    } else {
        spend.min(individual_resources)
    };
    ProbeDecision {
        move_one_step: true,
        move_signal: output.move_signal,
        direction_degrees,
        movement_spend,
        gathered: 0.0,
    }
}

fn build_probe_sensors(
    kind: BrainKind,
    edge_resources: [f64; 6],
    center_resource: f64,
    local_population: i32,
    individual_resources: f64,
) -> [f64; SENSOR_DIM] {
    match kind {
        BrainKind::Perceptron => {
            let mut s = [0.0f64; SENSOR_DIM];
            s[0] = center_resource;
            s[1] = local_population as f64;
            s[2] = individual_resources;
            s
        }
        _ => {
            let mut s = [0.0f64; SENSOR_DIM];
            s[..6].copy_from_slice(&edge_resources);
            s[6] = local_population as f64;
            s[7] = individual_resources;
            s
        }
    }
}

/// Run one brain forward pass for the landscape hex probe (no env mutation).
pub fn forward_probe(
    brain_json: &str,
    edge_resources: [f64; 6],
    center_resource: f64,
    local_population: i32,
    individual_resources: f64,
    position: usize,
    movement_output_mode: &str,
    movement_cost_factor: f64,
) -> Result<ProbeDecision, String> {
    let brain = parse_brain_json(brain_json)?;
    let mode = parse_movement_output_mode(movement_output_mode)?;
    let sensors = build_probe_sensors(
        brain.kind(),
        edge_resources,
        center_resource,
        local_population,
        individual_resources,
    );
    let max_buf = brain.max_buffer_nodes().max(32);
    let mut hidden_buf = vec![0.0; max_buf];
    let mut out2_buf = vec![0.0; max_buf];
    let mut spatial_runtime = brain.init_spatial_runtime();
    let ctx = ForwardContext {
        sensors,
        position,
        hidden_buf: &mut hidden_buf,
        out2_buf: &mut out2_buf,
        spatial_runtime: spatial_runtime.as_mut(),
        movement_output_mode: mode,
    };
    let output = brain.forward(ctx);
    Ok(probe_decision_from_output(
        output,
        center_resource,
        local_population,
        individual_resources,
        movement_cost_factor,
    ))
}

pub fn forward_probe_json(
    brain_json: &str,
    edge_json: &str,
    center_resource: f64,
    local_population: i32,
    individual_resources: f64,
    position: u32,
    movement_output_mode: &str,
    movement_cost_factor: f64,
) -> String {
    let edge: [f64; 6] = match serde_json::from_str(edge_json) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({ "error": format!("edge_resources: {e}") }).to_string();
        }
    };
    match forward_probe(
        brain_json,
        edge,
        center_resource,
        local_population,
        individual_resources,
        position as usize,
        movement_output_mode,
        movement_cost_factor,
    ) {
        Ok(d) => serde_json::to_string(&d).unwrap_or_else(|e| {
            serde_json::json!({ "error": e.to_string() }).to_string()
        }),
        Err(err) => serde_json::json!({ "error": err }).to_string(),
    }
}



#[cfg(test)]

mod tests {

    use super::*;



    #[test]

    fn random_goo_reaches_target_size() {

        let brain = random_goo_brain(7, 5);

        if let BrainInner::GooGraph(g) = &brain.inner {

            assert_eq!(g.nodes.len(), 5);

        } else {

            panic!("expected goo");

        }

    }



    #[test]

    fn random_goo_can_branch() {

        let brain = random_goo_brain(99, 8);

        if let BrainInner::GooGraph(g) = &brain.inner {

            let has_branch = g.nodes.iter().any(|n| {

                n.op == GooOp::Generate

                    && n.parents.len() == 1

                    && g.nodes.iter().filter(|o| {

                        o.op == GooOp::Generate

                            && o.parents == n.parents

                            && o.id != n.id

                    }).count()

                        >= 1

            });

            let has_merge = g

                .nodes

                .iter()

                .any(|n| n.parents.len() >= 2);

            assert!(has_branch || has_merge, "expected branch or merge in random goo");

        } else {

            panic!("expected goo");

        }

    }



    #[test]

    fn diagram_roundtrip_json() {

        let brain = random_perceptron_brain(1, 8, 3);

        let json = brain_to_json(&brain).unwrap();

        let diagram = diagram_for_brain_json(&json, 520.0, 340.0).unwrap();

        assert!(!diagram.nodes.is_empty());

    }

}


