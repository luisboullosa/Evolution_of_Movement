//! Serializable brain diagram layout shared by egui (`nn_viz`) and WASM playground.
use crate::brain::goo_graph::{GooGraphBrain, GooOp};
use crate::brain::perceptron::PerceptronBrain;
use crate::brain::spatial_blob::SpatialBlobBrain;
use crate::brain::spatial_graph::SpatialGraphBrain;
use crate::brain::{BrainGenome, BrainInner};
use crate::neural_network::NeuralNetwork;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorRgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorRgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagramText {
    pub x: f32,
    pub y: f32,
    pub text: String,
    pub size: f32,
    pub color: ColorRgb,
    #[serde(default = "default_align_left")]
    pub align: String,
    #[serde(default = "default_valign_top")]
    pub valign: String,
}

fn default_align_left() -> String {
    "left".to_string()
}

fn default_valign_top() -> String {
    "top".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagramNode {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub fill: ColorRgb,
    pub label: String,
    pub label_color: ColorRgb,
    pub label_size: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagramEdge {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub color: ColorRgba,
    pub width: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainDiagram {
    pub title: String,
    pub subtitle: String,
    pub legend: String,
    pub width: f32,
    pub height: f32,
    pub panel_bg: ColorRgb,
    pub panel_border: ColorRgb,
    pub nodes: Vec<DiagramNode>,
    pub edges: Vec<DiagramEdge>,
    pub texts: Vec<DiagramText>,
}

fn rgb(r: u8, g: u8, b: u8) -> ColorRgb {
    ColorRgb { r, g, b }
}

fn rgba(r: u8, g: u8, b: u8, a: u8) -> ColorRgba {
    ColorRgba { r, g, b, a }
}

fn input_label(i: usize) -> String {
    match i {
        0 => "Edge 0".to_string(),
        1 => "Edge 1".to_string(),
        2 => "Edge 2".to_string(),
        3 => "Edge 3".to_string(),
        4 => "Edge 4".to_string(),
        5 => "Edge 5".to_string(),
        6 => "Local Occupancy".to_string(),
        7 => "Internal Resources".to_string(),
        _ => format!("Input {i}"),
    }
}

fn weight_to_color(weight: f64) -> ColorRgba {
    let intensity = (80.0 + weight.abs() * 140.0).clamp(0.0, 255.0) as u8;
    if weight >= 0.0 {
        rgba(45, 130, intensity, 210)
    } else {
        rgba(intensity, 80, 70, 210)
    }
}

fn bias_to_color(bias: f64) -> ColorRgb {
    let normalized = ((bias.tanh() + 1.0) * 0.5).clamp(0.0, 1.0);
    let r = (70.0 + normalized * 140.0) as u8;
    let g = (80.0 + normalized * 120.0) as u8;
    let b = (90.0 + (1.0 - normalized) * 110.0) as u8;
    rgb(r, g, b)
}

fn goo_op_label(op: GooOp) -> &'static str {
    match op {
        GooOp::Generate => "Gen",
        GooOp::Aggregate => "Agg",
        GooOp::Score => "Score",
        GooOp::KeepBest => "Best",
    }
}

fn goo_op_color(op: GooOp) -> ColorRgb {
    match op {
        GooOp::Generate => rgb(100, 150, 255),
        GooOp::Aggregate => rgb(120, 200, 140),
        GooOp::Score => rgb(220, 180, 80),
        GooOp::KeepBest => rgb(220, 120, 160),
    }
}

pub fn diagram_from_brain(brain: &BrainGenome, width: f32, height: f32) -> BrainDiagram {
    match &brain.inner {
        BrainInner::Perceptron(p) => perceptron_diagram(p, width, height),
        BrainInner::Mlp(m) => mlp_diagram(&m.nn, width, height),
        BrainInner::SpatialGraph(s) => spatial_diagram(s, width, height),
        BrainInner::SpatialBlob(b) => spatial_blob_diagram(b, width, height),
        BrainInner::GooGraph(g) => goo_diagram(g, width, height),
    }
}

fn panel_frame(_width: f32, _height: f32) -> (ColorRgb, ColorRgb) {
    (rgb(22, 26, 34), rgb(70, 80, 100))
}

pub fn perceptron_diagram(p: &PerceptronBrain, width: f32, height: f32) -> BrainDiagram {
    let (panel_bg, panel_border) = panel_frame(width, height);
    let mut edges = Vec::new();
    let mut nodes = Vec::new();
    let texts = vec![
        DiagramText {
            x: 12.0,
            y: 8.0,
            text: "Perceptron brain".to_string(),
            size: 13.0,
            color: rgb(200, 205, 215),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 10.0,
            y: height - 18.0,
            text: "Perceptron — weights: red=negative, blue=positive".to_string(),
            size: 11.0,
            color: rgb(170, 176, 188),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
    ];

    let margin_x = 34.0;
    let margin_y = 18.0;
    let top_margin = 28.0;
    let plot_x0 = margin_x;
    let plot_y0 = margin_y + top_margin;
    let plot_x1 = width - margin_x;
    let plot_y1 = height - margin_y - 22.0;
    let plot_w = plot_x1 - plot_x0;
    let plot_h = plot_y1 - plot_y0;
    let layer_spacing = plot_w / 3.0;

    let input_count = p.inputs.min(8);
    let hidden_count = p.nodes.max(1);
    let max_layer_nodes = input_count.max(hidden_count);
    let node_radius = (11.0 - max_layer_nodes as f32 * 0.12).clamp(5.0, 10.0);

    let input_y_spacing = plot_h / (input_count + 1) as f32;
    let mut input_positions = Vec::new();
    for i in 0..input_count {
        let y = plot_y0 + input_y_spacing * (i + 1) as f32;
        input_positions.push((plot_x0 + layer_spacing * 0.5, y));
    }

    let hidden_y_spacing = plot_h / (hidden_count + 1) as f32;
    let mut hidden_positions = Vec::new();
    for i in 0..hidden_count {
        let y = plot_y0 + hidden_y_spacing * (i + 1) as f32;
        hidden_positions.push((plot_x0 + layer_spacing * 1.5, y));
    }

    let output_pos = (
        plot_x0 + layer_spacing * 2.5,
        plot_y0 + plot_h / 2.0,
    );

    for (i, &(ix, iy)) in input_positions.iter().enumerate() {
        for (j, &(hx, hy)) in hidden_positions.iter().enumerate() {
            let weight_idx = j * p.inputs + i;
            let weight = p.weights.get(weight_idx).copied().unwrap_or(0.0);
            let mut color = weight_to_color(weight);
            if weight.abs() < 1e-6 {
                color.a = 80;
            }
            let thickness = if weight.abs() < 1e-6 {
                0.35
            } else {
                (0.5 + weight.abs() as f32 * 1.6).clamp(0.5, 2.8)
            };
            edges.push(DiagramEdge {
                x1: ix,
                y1: iy,
                x2: hx,
                y2: hy,
                color,
                width: thickness,
            });
        }
    }

    for &(hx, hy) in &hidden_positions {
        edges.push(DiagramEdge {
            x1: hx,
            y1: hy,
            x2: output_pos.0,
            y2: output_pos.1,
            color: rgba(160, 165, 178, 255),
            width: 1.0,
        });
    }

    for (i, &(x, y)) in input_positions.iter().enumerate() {
        nodes.push(DiagramNode {
            id: format!("i{i}"),
            x,
            y,
            r: node_radius,
            fill: rgb(100, 150, 255),
            label: format!("I{i}"),
            label_color: rgb(255, 255, 255),
            label_size: 9.0,
            side_label: Some(input_label(i)),
        });
    }

    for (i, &(x, y)) in hidden_positions.iter().enumerate() {
        let bias = p.biases.get(i).copied().unwrap_or(0.0);
        nodes.push(DiagramNode {
            id: format!("h{i}"),
            x,
            y,
            r: node_radius,
            fill: bias_to_color(bias),
            label: format!("H{i}"),
            label_color: rgb(255, 255, 255),
            label_size: 10.0,
            side_label: None,
        });
    }

    nodes.push(DiagramNode {
        id: "out".to_string(),
        x: output_pos.0,
        y: output_pos.1,
        r: node_radius,
        fill: rgb(255, 150, 100),
        label: "OUT".to_string(),
        label_color: rgb(255, 255, 255),
        label_size: 9.0,
        side_label: None,
    });

    BrainDiagram {
        title: "Perceptron brain".to_string(),
        subtitle: format!(
            "{} inputs · {} hidden · {} weights",
            p.inputs,
            p.nodes,
            p.weights.len()
        ),
        legend: "Perceptron — weights: red=negative, blue=positive".to_string(),
        width,
        height,
        panel_bg,
        panel_border,
        nodes,
        edges,
        texts,
    }
}

pub fn goo_diagram(brain: &GooGraphBrain, width: f32, height: f32) -> BrainDiagram {
    let (panel_bg, panel_border) = panel_frame(width, height);
    let mut edges = Vec::new();
    let mut nodes = Vec::new();
    let mut texts = vec![
        DiagramText {
            x: 12.0,
            y: 16.0,
            text: "GoO graph brain".to_string(),
            size: 15.0,
            color: rgb(255, 255, 255),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 12.0,
            y: 36.0,
            text: format!("{} nodes · {} params", brain.nodes.len(), brain.params.len()),
            size: 11.0,
            color: rgb(200, 205, 215),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 10.0,
            y: height - 18.0,
            text: "DAG left→right · Gen=blue Agg=green Score=gold Best=pink".to_string(),
            size: 11.0,
            color: rgb(170, 176, 188),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
    ];

    let plot_x0 = 20.0;
    let plot_y0 = 56.0;
    let plot_x1 = width - 20.0;
    let plot_y1 = height - 28.0;
    let plot_w = plot_x1 - plot_x0;
    let plot_h = plot_y1 - plot_y0;

    if brain.nodes.is_empty() {
        texts.push(DiagramText {
            x: (plot_x0 + plot_x1) * 0.5,
            y: (plot_y0 + plot_y1) * 0.5,
            text: "Empty graph".to_string(),
            size: 12.0,
            color: rgb(200, 205, 215),
            align: "center".to_string(),
            valign: "middle".to_string(),
        });
        return BrainDiagram {
            title: "GoO graph brain".to_string(),
            subtitle: "0 nodes · 0 params".to_string(),
            legend: "DAG left→right · Gen=blue Agg=green Score=gold Best=pink".to_string(),
            width,
            height,
            panel_bg,
            panel_border,
            nodes,
            edges,
            texts,
        };
    }

    let mut depth: HashMap<usize, usize> = HashMap::new();
    for node in &brain.nodes {
        let d = node
            .parents
            .iter()
            .filter_map(|p| depth.get(p))
            .max()
            .map(|d| d + 1)
            .unwrap_or(0);
        depth.insert(node.id, d);
    }
    let max_depth = depth.values().copied().max().unwrap_or(0);
    let layers = max_depth + 1;

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); layers];
    for node in &brain.nodes {
        let d = *depth.get(&node.id).unwrap_or(&0);
        buckets[d].push(node.id);
    }

    let node_radius = 12.0f32;
    let mut positions: HashMap<usize, (f32, f32)> = HashMap::new();
    for (layer_idx, bucket) in buckets.iter().enumerate() {
        let x = plot_x0 + (layer_idx as f32 + 0.5) * plot_w / layers.max(1) as f32;
        let count = bucket.len().max(1);
        for (row, &id) in bucket.iter().enumerate() {
            let y = plot_y0 + (row as f32 + 1.0) * plot_h / (count as f32 + 1.0);
            positions.insert(id, (x, y));
        }
    }

    for node in &brain.nodes {
        if let Some(&(x, y)) = positions.get(&node.id) {
            for &parent in &node.parents {
                if let Some(&(px, py)) = positions.get(&parent) {
                    edges.push(DiagramEdge {
                        x1: px,
                        y1: py,
                        x2: x,
                        y2: y,
                        color: rgba(150, 160, 180, 255),
                        width: 1.5,
                    });
                }
            }
        }
    }

    for node in &brain.nodes {
        if let Some(&(x, y)) = positions.get(&node.id) {
            nodes.push(DiagramNode {
                id: node.id.to_string(),
                x,
                y,
                r: node_radius,
                fill: goo_op_color(node.op),
                label: goo_op_label(node.op).to_string(),
                label_color: rgb(255, 255, 255),
                label_size: 8.0,
                side_label: None,
            });
        }
    }

    BrainDiagram {
        title: "GoO graph brain".to_string(),
        subtitle: format!("{} nodes · {} params", brain.nodes.len(), brain.params.len()),
        legend: "DAG left→right · branch+merge DAG (no self-loops/backtrack yet)".to_string(),
        width,
        height,
        panel_bg,
        panel_border,
        nodes,
        edges,
        texts,
    }
}

pub fn mlp_diagram(nn: &NeuralNetwork, width: f32, height: f32) -> BrainDiagram {
    let (panel_bg, panel_border) = panel_frame(width, height);
    let mut edges = Vec::new();
    let mut nodes = Vec::new();
    let texts = vec![
        DiagramText {
            x: 12.0,
            y: 8.0,
            text: "MLP brain".to_string(),
            size: 13.0,
            color: rgb(200, 205, 215),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 10.0,
            y: height - 18.0,
            text: "MLP — weights: red=negative, blue=positive (thicker = stronger)".to_string(),
            size: 11.0,
            color: rgb(170, 176, 188),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
    ];

    let margin_x = 34.0;
    let margin_y = 18.0;
    let top_margin = 28.0;
    let plot_x0 = margin_x;
    let plot_y0 = margin_y + top_margin;
    let plot_x1 = width - margin_x;
    let plot_y1 = height - margin_y - 22.0;
    let plot_w = plot_x1 - plot_x0;
    let plot_h = plot_y1 - plot_y0;

    let layer_count: f32 = if nn.layers == 2 && nn.nodes2 > 0 { 4.0 } else { 3.0 };
    let layer_spacing = plot_w / layer_count.max(1.0);
    let max_layer_nodes = nn.inputs.max(nn.nodes).max(nn.nodes2.max(1));
    let node_radius = (11.0 - max_layer_nodes as f32 * 0.12).clamp(5.0, 10.0);

    let input_y_spacing = plot_h / (nn.inputs + 1) as f32;
    let input_positions: Vec<(f32, f32)> = (0..nn.inputs)
        .map(|i| {
            let y = plot_y0 + input_y_spacing * (i + 1) as f32;
            (plot_x0 + layer_spacing * 0.5, y)
        })
        .collect();

    let hidden_y_spacing = plot_h / (nn.nodes + 1) as f32;
    let hidden_positions: Vec<(f32, f32)> = (0..nn.nodes)
        .map(|i| {
            let y = plot_y0 + hidden_y_spacing * (i + 1) as f32;
            (plot_x0 + layer_spacing * 1.5, y)
        })
        .collect();

    let hidden2_positions: Vec<(f32, f32)> = if nn.layers == 2 && nn.nodes2 > 0 {
        let spacing = plot_h / (nn.nodes2 + 1) as f32;
        (0..nn.nodes2)
            .map(|i| {
                let y = plot_y0 + spacing * (i + 1) as f32;
                (plot_x0 + layer_spacing * 2.5, y)
            })
            .collect()
    } else {
        Vec::new()
    };

    let output_pos = (
        plot_x0 + layer_spacing * if nn.layers == 2 && nn.nodes2 > 0 { 3.5 } else { 2.5 },
        plot_y0 + plot_h / 2.0,
    );

    for (i, &(ix, iy)) in input_positions.iter().enumerate() {
        for (j, &(hx, hy)) in hidden_positions.iter().enumerate() {
            if i < nn.inputs && j < nn.nodes {
                let idx = j * nn.inputs + i;
                if let Some(&weight) = nn.weights_multi.get(idx) {
                    edges.push(DiagramEdge {
                        x1: ix,
                        y1: iy,
                        x2: hx,
                        y2: hy,
                        color: weight_to_color(weight),
                        width: (0.5 + weight.abs() as f32 * 1.6).clamp(0.5, 2.8),
                    });
                }
            }
        }
    }

    if nn.layers == 2 && !hidden2_positions.is_empty() {
        for (i, &(hx, hy)) in hidden_positions.iter().enumerate() {
            for (j, &(h2x, h2y)) in hidden2_positions.iter().enumerate() {
                if i < nn.nodes && j < nn.nodes2 {
                    let idx = i * nn.nodes2 + j;
                    if let Some(&weight) = nn.weights2.get(idx) {
                        edges.push(DiagramEdge {
                            x1: hx,
                            y1: hy,
                            x2: h2x,
                            y2: h2y,
                            color: weight_to_color(weight),
                            width: (0.5 + weight.abs() as f32 * 1.6).clamp(0.5, 2.8),
                        });
                    }
                }
            }
        }
    }

    let last_layer = if nn.layers == 2 && !hidden2_positions.is_empty() {
        &hidden2_positions
    } else {
        &hidden_positions
    };
    for &(lx, ly) in last_layer {
        edges.push(DiagramEdge {
            x1: lx,
            y1: ly,
            x2: output_pos.0,
            y2: output_pos.1,
            color: rgba(160, 165, 178, 255),
            width: 1.0,
        });
    }

    for (i, &(x, y)) in input_positions.iter().enumerate() {
        nodes.push(DiagramNode {
            id: format!("i{i}"),
            x,
            y,
            r: node_radius,
            fill: rgb(100, 150, 255),
            label: format!("I{i}"),
            label_color: rgb(255, 255, 255),
            label_size: 9.0,
            side_label: Some(input_label(i)),
        });
    }
    for (i, &(x, y)) in hidden_positions.iter().enumerate() {
        let bias = nn.biases.get(i).copied().unwrap_or(0.0);
        nodes.push(DiagramNode {
            id: format!("h{i}"),
            x,
            y,
            r: node_radius,
            fill: bias_to_color(bias),
            label: format!("H{i}"),
            label_color: rgb(255, 255, 255),
            label_size: 10.0,
            side_label: None,
        });
    }
    if nn.layers == 2 {
        for (i, &(x, y)) in hidden2_positions.iter().enumerate() {
            let bias = nn.biases2.get(i).copied().unwrap_or(0.0);
            nodes.push(DiagramNode {
                id: format!("h2{i}"),
                x,
                y,
                r: node_radius,
                fill: bias_to_color(bias),
                label: format!("H2{i}"),
                label_color: rgb(255, 255, 255),
                label_size: 8.0,
                side_label: None,
            });
        }
    }
    nodes.push(DiagramNode {
        id: "out".to_string(),
        x: output_pos.0,
        y: output_pos.1,
        r: node_radius,
        fill: rgb(255, 150, 100),
        label: "OUT".to_string(),
        label_color: rgb(255, 255, 255),
        label_size: 9.0,
        side_label: Some("Move / Direction".to_string()),
    });

    BrainDiagram {
        title: "MLP brain".to_string(),
        subtitle: format!(
            "{} inputs · {} hidden · layers={}",
            nn.inputs, nn.nodes, nn.layers
        ),
        legend: "MLP — weights: red=negative, blue=positive".to_string(),
        width,
        height,
        panel_bg,
        panel_border,
        nodes,
        edges,
        texts,
    }
}

pub fn spatial_diagram(brain: &SpatialGraphBrain, width: f32, height: f32) -> BrainDiagram {
    let (panel_bg, panel_border) = panel_frame(width, height);
    let mut edges = Vec::new();
    let mut nodes = Vec::new();
    let texts = vec![
        DiagramText {
            x: 12.0,
            y: 16.0,
            text: "Spatial graph brain".to_string(),
            size: 15.0,
            color: rgb(255, 255, 255),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 12.0,
            y: 36.0,
            text: format!(
                "{} landmarks · {} edges · {} message passes",
                brain.landmark_count,
                brain.edges.len(),
                brain.message_passes
            ),
            size: 11.0,
            color: rgb(200, 205, 215),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 10.0,
            y: height - 18.0,
            text: "Landmarks on a ring; edge color = message weight".to_string(),
            size: 11.0,
            color: rgb(170, 176, 188),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
    ];

    let plot_x0 = 24.0;
    let plot_y0 = 58.0;
    let plot_x1 = width - 24.0;
    let plot_y1 = height - 28.0;
    let center_x = (plot_x0 + plot_x1) * 0.5;
    let center_y = (plot_y0 + plot_y1) * 0.5;
    let radius = (plot_x1 - plot_x0).min(plot_y1 - plot_y0) * 0.35;
    let landmark_count = brain.landmark_count.max(1);
    let node_radius = 10.0f32;

    let positions: Vec<(f32, f32)> = (0..landmark_count)
        .map(|i| {
            let angle =
                std::f32::consts::TAU * i as f32 / landmark_count as f32 - std::f32::consts::FRAC_PI_2;
            (
                center_x + radius * angle.cos(),
                center_y + radius * angle.sin(),
            )
        })
        .collect();

    for edge in &brain.edges {
        if edge.from < positions.len() && edge.to < positions.len() {
            let weight = brain.edge_weights.get(edge.weight_idx).copied().unwrap_or(0.0);
            let (x1, y1) = positions[edge.from];
            let (x2, y2) = positions[edge.to];
            edges.push(DiagramEdge {
                x1,
                y1,
                x2,
                y2,
                color: weight_to_color(weight),
                width: (0.8 + weight.abs() as f32 * 1.4).clamp(0.8, 3.0),
            });
        }
    }

    let sensor_pos = (plot_x0 + 8.0, center_y);
    if let Some(&(fx, fy)) = positions.first() {
        edges.push(DiagramEdge {
            x1: sensor_pos.0,
            y1: sensor_pos.1,
            x2: fx,
            y2: fy,
            color: rgba(140, 150, 170, 255),
            width: 1.2,
        });
    }

    nodes.push(DiagramNode {
        id: "sensors".to_string(),
        x: sensor_pos.0,
        y: sensor_pos.1,
        r: 7.0,
        fill: rgb(100, 150, 255),
        label: String::new(),
        label_color: rgb(255, 255, 255),
        label_size: 9.0,
        side_label: Some("Sensors".to_string()),
    });

    for (i, &(x, y)) in positions.iter().enumerate() {
        nodes.push(DiagramNode {
            id: format!("l{i}"),
            x,
            y,
            r: node_radius,
            fill: rgb(120, 200, 140),
            label: format!("L{i}"),
            label_color: rgb(255, 255, 255),
            label_size: 9.0,
            side_label: None,
        });
    }

    BrainDiagram {
        title: "Spatial graph brain".to_string(),
        subtitle: format!(
            "{} landmarks · {} edges · {} message passes",
            brain.landmark_count,
            brain.edges.len(),
            brain.message_passes
        ),
        legend: "Landmarks on a ring; edge color = message weight".to_string(),
        width,
        height,
        panel_bg,
        panel_border,
        nodes,
        edges,
        texts,
    }
}

pub fn spatial_blob_diagram(brain: &SpatialBlobBrain, width: f32, height: f32) -> BrainDiagram {
    let (panel_bg, panel_border) = panel_frame(width, height);
    let layout = format!("{:?}", brain.layout);
    let nodes_per = brain.nodes_per_layer();
    let texts = vec![
        DiagramText {
            x: 12.0,
            y: 16.0,
            text: "Spatial macromolecule blob".to_string(),
            size: 15.0,
            color: rgb(255, 255, 255),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 12.0,
            y: 36.0,
            text: format!(
                "{layout} · {} layers · {nodes_per} nodes/layer",
                brain.layer_count
            ),
            size: 11.0,
            color: rgb(200, 205, 215),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
        DiagramText {
            x: 10.0,
            y: height - 18.0,
            text: "Hex edges feed layer 0; deeper layers mix spatial nodes".to_string(),
            size: 11.0,
            color: rgb(170, 176, 188),
            align: "left".to_string(),
            valign: "top".to_string(),
        },
    ];
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let cx = width * 0.5;
    let cy = height * 0.55;
    let plot_r = (width.min(height) * 0.42).max(48.0);
    let node_r = 9.0f32;
    let base_r = node_r + 6.0;
    let layer_gap = if brain.layer_count <= 1 {
        0.0
    } else {
        ((plot_r - base_r) / (brain.layer_count - 1) as f32).clamp(8.0, 28.0)
    };

    for layer in 0..brain.layer_count {
        let r = base_r + layer as f32 * layer_gap;
        for i in 0..nodes_per {
            let angle = std::f32::consts::TAU * i as f32 / nodes_per as f32 - std::f32::consts::FRAC_PI_2;
            let x = cx + r * angle.cos();
            let y = cy + r * angle.sin();
            let label = if layer == 0 && nodes_per == 6 {
                ["E", "NE", "NW", "W", "SW", "SE"]
                    .get(i)
                    .unwrap_or(&"?")
                    .to_string()
            } else if layer == 0 && nodes_per == 4 {
                ["N", "S", "W", "E"].get(i).unwrap_or(&"?").to_string()
            } else {
                format!("{i}")
            };
            nodes.push(DiagramNode {
                id: format!("L{layer}_{i}"),
                x,
                y,
                r: node_r,
                fill: if layer == 0 {
                    rgb(100, 170, 255)
                } else {
                    rgb(140, 200, 160)
                },
                label,
                label_color: rgb(255, 255, 255),
                label_size: 9.0,
                side_label: None,
            });
            if layer > 0 {
                let prev_r = base_r + (layer - 1) as f32 * layer_gap;
                let prev_x = cx + prev_r * angle.cos();
                let prev_y = cy + prev_r * angle.sin();
                edges.push(DiagramEdge {
                    x1: prev_x,
                    y1: prev_y,
                    x2: x,
                    y2: y,
                    color: rgba(120, 150, 170, 140),
                    width: 0.8,
                });
            }
        }
    }

    nodes.push(DiagramNode {
        id: "sensors".to_string(),
        x: cx,
        y: cy,
        r: 12.0,
        fill: rgb(220, 180, 80),
        label: "hex".to_string(),
        label_color: rgb(30, 30, 30),
        label_size: 9.0,
        side_label: Some("edges".to_string()),
    });

    BrainDiagram {
        title: "Spatial blob".to_string(),
        subtitle: format!("{layout} · {} layers", brain.layer_count),
        legend: "Intracellular spatial macromolecule layers".to_string(),
        width,
        height,
        panel_bg,
        panel_border,
        nodes,
        edges,
        texts,
    }
}
