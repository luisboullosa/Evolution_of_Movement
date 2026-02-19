/// Neural network visualization utilities
use crate::neural_network::NeuralNetwork;
use eframe::egui;

fn input_label(i: usize) -> String {
    // Interactive mode uses 8 canonical inputs:
    // 6 edge resource sensors + local occupancy + internal resources.
    // Additional inputs (if any) are still rendered with generic labels.
    match i {
        0 => "Edge 0".to_string(),
        1 => "Edge 1".to_string(),
        2 => "Edge 2".to_string(),
        3 => "Edge 3".to_string(),
        4 => "Edge 4".to_string(),
        5 => "Edge 5".to_string(),
        6 => "Local Occupancy".to_string(),
        7 => "Internal Resources".to_string(),
        _ => format!("Input {}", i),
    }
}

pub fn draw_neural_network(
    ui: &mut egui::Ui,
    nn: &NeuralNetwork,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    let layer_spacing = rect.width() / 4.0;
    let node_radius = 8.0;

    // Calculate positions for input layer
    let input_count = nn.inputs;
    let input_y_spacing = rect.height() / (input_count + 1) as f32;
    let mut input_positions = Vec::new();
    for i in 0..input_count {
        let y = rect.min.y + input_y_spacing * (i + 1) as f32;
        input_positions.push(egui::pos2(rect.min.x + layer_spacing, y));
    }

    // Calculate positions for hidden layer
    let hidden_count = nn.nodes;
    let hidden_y_spacing = rect.height() / (hidden_count + 1) as f32;
    let mut hidden_positions = Vec::new();
    for i in 0..hidden_count {
        let y = rect.min.y + hidden_y_spacing * (i + 1) as f32;
        hidden_positions.push(egui::pos2(rect.min.x + layer_spacing * 2.0, y));
    }

    // Calculate positions for second hidden layer (if exists)
    let mut hidden2_positions = Vec::new();
    if nn.layers == 2 && nn.nodes2 > 0 {
        let nodes2_count = nn.nodes2;
        let nodes2_y_spacing = rect.height() / (nodes2_count + 1) as f32;
        for i in 0..nodes2_count {
            let y = rect.min.y + nodes2_y_spacing * (i + 1) as f32;
            hidden2_positions.push(egui::pos2(rect.min.x + layer_spacing * 3.0, y));
        }
    }

    // Output position (single output)
    let output_pos = egui::pos2(
        rect.min.x + layer_spacing * if nn.layers == 2 { 4.0 } else { 3.0 },
        rect.center().y,
    );

    // Draw connections from input to hidden
    for (i, &input_pos) in input_positions.iter().enumerate() {
        for (j, &hidden_pos) in hidden_positions.iter().enumerate() {
            if i < nn.inputs && j < hidden_count {
                let weight_idx = j * nn.inputs + i;
                if weight_idx < nn.weights_multi.len() {
                    let weight = nn.weights_multi[weight_idx];
                    let color = weight_to_color(weight);
                    let thickness = weight.abs().min(2.0) as f32;
                    painter.line_segment(
                        [input_pos, hidden_pos],
                        egui::Stroke::new(thickness, color),
                    );
                }
            }
        }
    }

    // Draw connections from hidden to second hidden (if exists)
    if nn.layers == 2 && !hidden2_positions.is_empty() {
        for (i, &hidden_pos) in hidden_positions.iter().enumerate() {
            for (j, &hidden2_pos) in hidden2_positions.iter().enumerate() {
                if i < hidden_count && j < nn.nodes2 {
                    // weights2 are stored in row-major order: idx = i * nodes2 + j
                    let weight_idx = i * nn.nodes2 + j;
                    if weight_idx < nn.weights2.len() {
                        let weight = nn.weights2[weight_idx];
                        let color = weight_to_color(weight);
                        let thickness = weight.abs().min(2.0) as f32;
                        painter.line_segment(
                            [hidden_pos, hidden2_pos],
                            egui::Stroke::new(thickness, color),
                        );
                    }
                }
            }
        }
    }

    // Draw connections to output
    let last_layer_positions = if nn.layers == 2 && !hidden2_positions.is_empty() {
        &hidden2_positions
    } else {
        &hidden_positions
    };

    for &last_pos in last_layer_positions.iter() {
        // Simplified: draw uniform connections to output
        painter.line_segment(
            [last_pos, output_pos],
            egui::Stroke::new(1.0, egui::Color32::GRAY),
        );
    }

    // Draw input nodes
    for (i, &pos) in input_positions.iter().enumerate() {
        painter.circle_filled(pos, node_radius, egui::Color32::from_rgb(100, 150, 255));
        painter.circle_stroke(pos, node_radius, egui::Stroke::new(2.0, egui::Color32::BLACK));
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            format!("I{}", i),
            egui::FontId::proportional(10.0),
            egui::Color32::WHITE,
        );
        painter.text(
            egui::pos2(pos.x - 14.0, pos.y),
            egui::Align2::RIGHT_CENTER,
            input_label(i),
            egui::FontId::proportional(10.0),
            egui::Color32::LIGHT_GRAY,
        );
    }

    // Draw hidden nodes with biases
    for (i, &pos) in hidden_positions.iter().enumerate() {
        let bias = if i < nn.biases.len() {
            nn.biases[i]
        } else {
            0.0
        };
        let color = bias_to_color(bias);
        painter.circle_filled(pos, node_radius, color);
        painter.circle_stroke(pos, node_radius, egui::Stroke::new(2.0, egui::Color32::BLACK));
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            format!("H{}", i),
            egui::FontId::proportional(10.0),
            egui::Color32::WHITE,
        );
    }

    // Draw second hidden layer nodes (if exists)
    if nn.layers == 2 {
        for (i, &pos) in hidden2_positions.iter().enumerate() {
            let bias = if i < nn.biases2.len() {
                nn.biases2[i]
            } else {
                0.0
            };
            let color = bias_to_color(bias);
            painter.circle_filled(pos, node_radius, color);
            painter.circle_stroke(pos, node_radius, egui::Stroke::new(2.0, egui::Color32::BLACK));
            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                format!("H2_{}", i),
                egui::FontId::proportional(9.0),
                egui::Color32::WHITE,
            );
        }
    }

    // Draw output node
    painter.circle_filled(output_pos, node_radius, egui::Color32::from_rgb(255, 150, 100));
    painter.circle_stroke(
        output_pos,
        node_radius,
        egui::Stroke::new(2.0, egui::Color32::BLACK),
    );
    painter.text(
        output_pos,
        egui::Align2::CENTER_CENTER,
        "OUT",
        egui::FontId::proportional(10.0),
        egui::Color32::WHITE,
    );

    // Add legend
    let legend_y = rect.max.y - 40.0;
    ui.painter().text(
        egui::pos2(rect.min.x + 10.0, legend_y),
        egui::Align2::LEFT_TOP,
        "Weights: Red=negative, Blue=positive",
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );
}

fn weight_to_color(weight: f64) -> egui::Color32 {
    if weight > 0.0 {
        let intensity = (weight * 100.0).min(255.0) as u8;
        egui::Color32::from_rgb(0, 0, intensity)
    } else {
        let intensity = (-weight * 100.0).min(255.0) as u8;
        egui::Color32::from_rgb(intensity, 0, 0)
    }
}

fn bias_to_color(bias: f64) -> egui::Color32 {
    let normalized = (bias + 1.0) / 2.0; // Assume biases are roughly in [-1, 1]
    let intensity = (normalized * 255.0).min(255.0).max(0.0) as u8;
    egui::Color32::from_rgb(intensity, intensity, 100)
}
