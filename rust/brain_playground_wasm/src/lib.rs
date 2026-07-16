use evo_movement_rust::brain_playground::{
    brain_to_diagram_json, brain_to_json, demo_goo_branching_brain, goo_add_structural_node,
    goo_cycle_node_op, parse_brain_json, random_goo_brain, random_mlp_brain,
    random_perceptron_brain, random_spatial_brain,
};
use wasm_bindgen::prelude::*;

fn wrap_json<T>(result: Result<T, String>) -> String
where
    T: serde::Serialize,
{
    match result {
        Ok(value) => serde_json::to_string(&value).unwrap_or_else(|e| {
            serde_json::json!({ "error": e.to_string() }).to_string()
        }),
        Err(err) => serde_json::json!({ "error": err }).to_string(),
    }
}

#[wasm_bindgen]
pub fn random_perceptron_json(seed: u32, inputs: u32, nodes: u32) -> String {
    let brain = random_perceptron_brain(seed, inputs as usize, nodes as usize);
    brain_to_json(&brain).unwrap_or_else(|e| serde_json::json!({ "error": e }).to_string())
}

#[wasm_bindgen]
pub fn random_goo_json(seed: u32, target_nodes: u32) -> String {
    let brain = random_goo_brain(seed, target_nodes as usize);
    brain_to_json(&brain).unwrap_or_else(|e| serde_json::json!({ "error": e }).to_string())
}

#[wasm_bindgen]
pub fn random_mlp_json(seed: u32, hidden: u32, two_layer: bool) -> String {
    let brain = random_mlp_brain(seed, hidden as usize, two_layer);
    brain_to_json(&brain).unwrap_or_else(|e| serde_json::json!({ "error": e }).to_string())
}

#[wasm_bindgen]
pub fn random_spatial_json(seed: u32, landmarks: u32, edges: u32) -> String {
    let brain = random_spatial_brain(seed, landmarks as usize, edges as usize);
    brain_to_json(&brain).unwrap_or_else(|e| serde_json::json!({ "error": e }).to_string())
}

#[wasm_bindgen]
pub fn demo_goo_branching_json() -> String {
    let brain = demo_goo_branching_brain();
    brain_to_json(&brain).unwrap_or_else(|e| serde_json::json!({ "error": e }).to_string())
}

#[wasm_bindgen]
pub fn diagram_json(brain_json: &str, width: f32, height: f32) -> String {
    match parse_brain_json(brain_json).and_then(|brain| brain_to_diagram_json(&brain, width, height)) {
        Ok(json) => json,
        Err(err) => serde_json::json!({ "error": err }).to_string(),
    }
}

#[wasm_bindgen]
pub fn validate_brain_json(brain_json: &str) -> String {
    wrap_json(
        parse_brain_json(brain_json)
            .map(|brain| serde_json::json!({ "ok": true, "kind": brain.kind.as_str() })),
    )
}

#[wasm_bindgen]
pub fn goo_cycle_op_json(brain_json: &str, node_id: u32) -> String {
    wrap_json(parse_brain_json(brain_json).and_then(|mut brain| {
        if goo_cycle_node_op(&mut brain, node_id as usize) {
            brain_to_json(&brain)
                .map(|json| serde_json::from_str::<serde_json::Value>(&json).unwrap())
        } else {
            Err("node not found or not a GoO brain".to_string())
        }
    }))
}

#[wasm_bindgen]
pub fn goo_add_node_json(brain_json: &str, op: &str) -> String {
    wrap_json(parse_brain_json(brain_json).and_then(|mut brain| {
        goo_add_structural_node(&mut brain, op)?;
        brain_to_json(&brain).map(|json| serde_json::from_str::<serde_json::Value>(&json).unwrap())
    }))
}
