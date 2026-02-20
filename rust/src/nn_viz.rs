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
    let panel_bg = egui::Color32::from_rgb(22, 26, 34);
    let panel_border = egui::Color32::from_rgb(70, 80, 100);
    painter.rect_filled(rect, 6.0, panel_bg);
    painter.rect_stroke(rect, 6.0, egui::Stroke::new(1.2, panel_border));

    let margin_x = 34.0;
    let margin_y = 18.0;
    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + margin_x, rect.min.y + margin_y),
        egui::pos2(rect.max.x - margin_x, rect.max.y - margin_y - 22.0),
    );

    let layer_count: f32 = if nn.layers == 2 && nn.nodes2 > 0 { 4.0 } else { 3.0 };
    let layer_spacing = plot.width() / layer_count.max(1.0);

    let max_layer_nodes = nn.inputs.max(nn.nodes).max(nn.nodes2.max(1));
    let node_radius = (11.0 - max_layer_nodes as f32 * 0.12).clamp(5.0, 10.0);

    // Calculate positions for input layer
    let input_count = nn.inputs;
    let input_y_spacing = plot.height() / (input_count + 1) as f32;
    let mut input_positions = Vec::new();
    for i in 0..input_count {
        let y = plot.min.y + input_y_spacing * (i + 1) as f32;
        input_positions.push(egui::pos2(plot.min.x + layer_spacing * 0.5, y));
    }

    // Calculate positions for hidden layer
    let hidden_count = nn.nodes;
    let hidden_y_spacing = plot.height() / (hidden_count + 1) as f32;
    let mut hidden_positions = Vec::new();
    for i in 0..hidden_count {
        let y = plot.min.y + hidden_y_spacing * (i + 1) as f32;
        hidden_positions.push(egui::pos2(plot.min.x + layer_spacing * 1.5, y));
    }

    // Calculate positions for second hidden layer (if exists)
    let mut hidden2_positions = Vec::new();
    if nn.layers == 2 && nn.nodes2 > 0 {
        let nodes2_count = nn.nodes2;
        let nodes2_y_spacing = plot.height() / (nodes2_count + 1) as f32;
        for i in 0..nodes2_count {
            let y = plot.min.y + nodes2_y_spacing * (i + 1) as f32;
            hidden2_positions.push(egui::pos2(plot.min.x + layer_spacing * 2.5, y));
        }
    }

    // Output position (single output)
    let output_pos = egui::pos2(
        plot.min.x + layer_spacing * if nn.layers == 2 && nn.nodes2 > 0 { 3.5 } else { 2.5 },
        plot.center().y,
    );

    // Draw connections from input to hidden
    for (i, &input_pos) in input_positions.iter().enumerate() {
        for (j, &hidden_pos) in hidden_positions.iter().enumerate() {
            if i < nn.inputs && j < hidden_count {
                let weight_idx = j * nn.inputs + i;
                if weight_idx < nn.weights_multi.len() {
                    let weight = nn.weights_multi[weight_idx];
                    let color = weight_to_color(weight);
                    let thickness = (0.5 + weight.abs() as f32 * 1.6).clamp(0.5, 2.8);
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
                        let thickness = (0.5 + weight.abs() as f32 * 1.6).clamp(0.5, 2.8);
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
            egui::Stroke::new(1.0, egui::Color32::from_rgb(160, 165, 178)),
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
            egui::FontId::proportional(9.0),
            egui::Color32::WHITE,
        );
        painter.text(
            egui::pos2(pos.x - 14.0, pos.y),
            egui::Align2::RIGHT_CENTER,
            input_label(i),
            egui::FontId::proportional(9.0),
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
                format!("H2{}", i),
                egui::FontId::proportional(8.0),
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
        egui::FontId::proportional(9.0),
        egui::Color32::WHITE,
    );
    painter.text(
        egui::pos2(output_pos.x + node_radius + 6.0, output_pos.y),
        egui::Align2::LEFT_CENTER,
        "Move / Direction",
        egui::FontId::proportional(9.0),
        egui::Color32::LIGHT_GRAY,
    );

    // Add legend
    let legend_y = rect.max.y - 18.0;
    ui.painter().text(
        egui::pos2(rect.min.x + 10.0, legend_y),
        egui::Align2::LEFT_TOP,
        "Weights: red=negative, blue=positive (thicker = stronger)",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(170, 176, 188),
    );
}

fn weight_to_color(weight: f64) -> egui::Color32 {
    let intensity = (80.0 + weight.abs() * 140.0).clamp(0.0, 255.0) as u8;
    if weight >= 0.0 {
        egui::Color32::from_rgba_unmultiplied(45, 130, intensity, 210)
    } else {
        egui::Color32::from_rgba_unmultiplied(intensity, 80, 70, 210)
    }
}

fn bias_to_color(bias: f64) -> egui::Color32 {
    let normalized = ((bias.tanh() + 1.0) * 0.5).clamp(0.0, 1.0);
    let r = (70.0 + normalized * 140.0) as u8;
    let g = (80.0 + normalized * 120.0) as u8;
    let b = (90.0 + (1.0 - normalized) * 110.0) as u8;
    egui::Color32::from_rgb(r, g, b)
}
