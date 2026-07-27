/// Neural network visualization utilities
use crate::brain::spatial_blob::SpatialBlobBrain;
use crate::brain::goo_graph::GooGraphBrain;
use crate::brain::perceptron::PerceptronBrain;
use crate::brain::spatial_graph::SpatialGraphBrain;
use crate::brain::{BrainGenome, BrainInner};
use crate::brain_diagram::{BrainDiagram, ColorRgb, ColorRgba};
use crate::neural_network::NeuralNetwork;
use eframe::egui;

fn color_rgb(c: ColorRgb) -> egui::Color32 {
    egui::Color32::from_rgb(c.r, c.g, c.b)
}

fn color_rgba(c: ColorRgba) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c.r, c.g, c.b, c.a)
}

fn text_align(align: &str, valign: &str) -> egui::Align2 {
    match (align, valign) {
        ("center", "middle") => egui::Align2::CENTER_CENTER,
        ("center", "bottom") => egui::Align2::CENTER_BOTTOM,
        ("center", _) => egui::Align2::CENTER_TOP,
        ("right", "middle") => egui::Align2::RIGHT_CENTER,
        ("right", "bottom") => egui::Align2::RIGHT_BOTTOM,
        ("right", _) => egui::Align2::RIGHT_TOP,
        (_, "middle") => egui::Align2::LEFT_CENTER,
        (_, "bottom") => egui::Align2::LEFT_BOTTOM,
        _ => egui::Align2::LEFT_TOP,
    }
}

fn render_diagram(painter: &egui::Painter, rect: egui::Rect, diagram: &BrainDiagram) {
    draw_panel_frame(painter, rect);

    let ox = rect.min.x;
    let oy = rect.min.y;

    for edge in &diagram.edges {
        painter.line_segment(
            [
                egui::pos2(ox + edge.x1, oy + edge.y1),
                egui::pos2(ox + edge.x2, oy + edge.y2),
            ],
            egui::Stroke::new(edge.width, color_rgba(edge.color)),
        );
    }

    for node in &diagram.nodes {
        let pos = egui::pos2(ox + node.x, oy + node.y);
        painter.circle_filled(pos, node.r, color_rgb(node.fill));
        painter.circle_stroke(pos, node.r, egui::Stroke::new(2.0, egui::Color32::BLACK));
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            &node.label,
            egui::FontId::proportional(node.label_size),
            color_rgb(node.label_color),
        );
        if let Some(side) = &node.side_label {
            painter.text(
                egui::pos2(pos.x - 14.0, pos.y),
                egui::Align2::RIGHT_CENTER,
                side,
                egui::FontId::proportional(9.0),
                egui::Color32::LIGHT_GRAY,
            );
        }
    }

    for text in &diagram.texts {
        painter.text(
            egui::pos2(ox + text.x, oy + text.y),
            text_align(&text.align, &text.valign),
            &text.text,
            egui::FontId::proportional(text.size),
            color_rgb(text.color),
        );
    }
}

fn draw_panel_frame(painter: &egui::Painter, rect: egui::Rect) {
    let panel_bg = egui::Color32::from_rgb(22, 26, 34);
    let panel_border = egui::Color32::from_rgb(70, 80, 100);
    painter.rect_filled(rect, 6.0, panel_bg);
    painter.rect_stroke(rect, 6.0, egui::Stroke::new(1.2, panel_border));
}

const NN_DIAGRAM_HEIGHT: f32 = 340.0;

/// Full NN panel: layered diagram plus numeric weights/biases where applicable.
pub fn draw_brain_panel(ui: &mut egui::Ui, brain: &BrainGenome) {
    // Use this ui's available width only — nesting inside Frame::canvas with a wider
    // outer width caused horizontal overflow that drifted the diagram left each frame.
    let width = ui.available_width().max(280.0);
    let (rect, _response) = ui.allocate_exact_size(
        egui::vec2(width, NN_DIAGRAM_HEIGHT),
        egui::Sense::hover(),
    );
    if rect.width() < 2.0 {
        return;
    }
    let painter = ui.painter_at(rect);
    draw_brain(ui, brain, &painter, rect);
    ui.add_space(6.0);
    draw_weight_bias_stats(ui, brain);
}

fn draw_weight_bias_stats(ui: &mut egui::Ui, brain: &BrainGenome) {
    match &brain.inner {
        BrainInner::Perceptron(p) => {
            ui.collapsing("Weight & bias values", |ui| {
                draw_perceptron_weights(ui, p);
            });
        }
        BrainInner::Mlp(m) => {
            ui.collapsing("Weight & bias values", |ui| {
                draw_mlp_weights(ui, &m.nn);
            });
        }
        BrainInner::SpatialGraph(s) => {
            ui.collapsing("Edge & readout weights", |ui| {
                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .show(ui, |ui| {
                        egui::Grid::new("spatial_weights")
                            .num_columns(2)
                            .spacing([14.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("Edge weights").strong());
                                ui.label("");
                                for (i, w) in s.edge_weights.iter().enumerate() {
                                    ui.label(format!("edge[{i}]"));
                                    ui.label(weight_rich_text(*w));
                                }
                                ui.separator();
                                ui.separator();
                                ui.label(egui::RichText::new("Readout").strong());
                                ui.label("");
                                for (i, w) in s.readout_weights.iter().enumerate().take(16) {
                                    ui.label(format!("readout_w[{i}]"));
                                    ui.label(weight_rich_text(*w));
                                }
                            });
                    });
            });
        }
        BrainInner::SpatialBlob(b) => {
            ui.collapsing("Layer & readout weights", |ui| {
                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .show(ui, |ui| {
                        ui.label(format!(
                            "layout {:?} · {} layers · {} nodes/layer",
                            b.layout,
                            b.layer_count,
                            b.nodes_per_layer()
                        ));
                        egui::Grid::new("blob_weights")
                            .num_columns(2)
                            .spacing([14.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                for (i, w) in b.readout_weights.iter().enumerate() {
                                    ui.label(format!("readout_w[{i}]"));
                                    ui.label(weight_rich_text(*w));
                                }
                            });
                    });
            });
        }
        BrainInner::GooGraph(g) => {
            ui.collapsing("Graph structure & parameters", |ui| {
                draw_goo_graph_stats(ui, g);
            });
        }
    }
}

fn weight_rich_text(w: f64) -> egui::RichText {
    let color = if w > 0.0 {
        egui::Color32::from_rgb(100, 160, 255)
    } else if w < 0.0 {
        egui::Color32::from_rgb(220, 100, 90)
    } else {
        egui::Color32::from_rgb(150, 155, 165)
    };
    egui::RichText::new(format!("{w:+.4}"))
        .monospace()
        .color(color)
}

fn synapse_label(hidden: usize, input: usize) -> String {
    format!("H{hidden} <- I{input}")
}

fn draw_perceptron_weights(ui: &mut egui::Ui, p: &PerceptronBrain) {
    let inputs = p.inputs.max(1);
    let hidden = p.nodes.max(1);
    ui.label(format!(
        "Architecture: {inputs} inputs -> {hidden} hidden ({} weights, {} biases)",
        p.weights.len(),
        p.biases.len()
    ));
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Weight matrix").strong());
    ui.label(
        egui::RichText::new("Rows = hidden units, columns = inputs")
            .small()
            .weak(),
    );
    egui::ScrollArea::both()
        .max_height(200.0)
        .show(ui, |ui| {
            egui::Grid::new("perceptron_weight_matrix")
                .num_columns(inputs + 1)
                .spacing([10.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("").weak());
                    for i in 0..inputs {
                        ui.label(egui::RichText::new(format!("I{i}")).monospace().strong());
                    }
                    for h in 0..hidden {
                        ui.label(egui::RichText::new(format!("H{h}")).monospace().strong());
                        for i in 0..inputs {
                            let idx = h * inputs + i;
                            let w = p.weights.get(idx).copied().unwrap_or(0.0);
                            ui.label(weight_rich_text(w)).on_hover_text(synapse_label(h, i));
                        }
                    }
                });
        });
    ui.add_space(6.0);
    ui.label(egui::RichText::new("Biases").strong());
    egui::Grid::new("perceptron_biases")
        .num_columns(2)
        .spacing([14.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            for (i, b) in p.biases.iter().enumerate() {
                ui.label(format!("bias H{i}"));
                ui.label(weight_rich_text(*b));
            }
        });
}

fn draw_mlp_weights(ui: &mut egui::Ui, nn: &NeuralNetwork) {
    ui.label(format!(
        "Architecture: {} inputs -> {} hidden (layers={})",
        nn.inputs, nn.nodes, nn.layers
    ));
    if nn.layers == 2 && nn.nodes2 > 0 {
        ui.label(format!("Second hidden layer: {} nodes", nn.nodes2));
    }
    ui.add_space(4.0);
    egui::ScrollArea::vertical()
        .max_height(260.0)
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Input -> hidden weights").strong());
            egui::Grid::new("mlp_weights")
                .num_columns(2)
                .spacing([14.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    for (idx, w) in nn.weights_multi.iter().enumerate() {
                        let input = idx % nn.inputs.max(1);
                        let hidden = idx / nn.inputs.max(1);
                        ui.label(
                            egui::RichText::new(synapse_label(hidden, input))
                                .monospace(),
                        );
                        ui.label(weight_rich_text(*w));
                    }
                });
            if nn.layers == 2 && !nn.weights2.is_empty() {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Hidden -> hidden2 weights").strong());
                egui::Grid::new("mlp_weights2")
                    .num_columns(2)
                    .spacing([14.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for (idx, w) in nn.weights2.iter().enumerate() {
                            let h2 = idx % nn.nodes2.max(1);
                            let h1 = idx / nn.nodes2.max(1);
                            ui.label(
                                egui::RichText::new(format!("H2{h2} <- H1{h1}"))
                                    .monospace(),
                            );
                            ui.label(weight_rich_text(*w));
                        }
                    });
            }
            ui.add_space(6.0);
            ui.label(egui::RichText::new("Biases").strong());
            egui::Grid::new("mlp_biases")
                .num_columns(2)
                .spacing([14.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    for (i, b) in nn.biases.iter().enumerate() {
                        ui.label(format!("bias H1[{i}]"));
                        ui.label(weight_rich_text(*b));
                    }
                    for (i, b) in nn.biases2.iter().enumerate() {
                        ui.label(format!("bias H2[{i}]"));
                        ui.label(weight_rich_text(*b));
                    }
                });
        });
}

fn draw_goo_graph_stats(ui: &mut egui::Ui, g: &GooGraphBrain) {
    ui.label(format!(
        "{} nodes · {} params · thought_dim={}",
        g.nodes.len(),
        g.params.len(),
        g.thought_dim
    ));
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Nodes").strong());
    egui::Grid::new("goo_nodes")
        .num_columns(3)
        .spacing([12.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label(egui::RichText::new("id").strong());
            ui.label(egui::RichText::new("op").strong());
            ui.label(egui::RichText::new("parents").strong());
            for node in &g.nodes {
                ui.label(format!("{}", node.id));
                ui.label(format!("{:?}", node.op));
                ui.label(format!("{:?}", node.parents));
            }
        });
    ui.add_space(6.0);
    ui.label(egui::RichText::new("Parameters (sample)").strong());
    egui::ScrollArea::vertical()
        .max_height(160.0)
        .show(ui, |ui| {
            egui::Grid::new("goo_params")
                .num_columns(2)
                .spacing([14.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    let cap = g.params.len().min(32);
                    for (i, w) in g.params.iter().enumerate().take(cap) {
                        ui.label(format!("param[{i}]"));
                        ui.label(weight_rich_text(*w));
                    }
                    if g.params.len() > cap {
                        ui.label("…");
                        ui.label(format!("(+{} more)", g.params.len() - cap));
                    }
                });
        });
}

/// Route to the correct visualization for each evolved brain architecture.
pub fn draw_brain(
    ui: &mut egui::Ui,
    brain: &BrainGenome,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    match &brain.inner {
        BrainInner::Perceptron(p) => draw_perceptron(ui, p, painter, rect),
        BrainInner::Mlp(m) => draw_neural_network(ui, &m.nn, painter, rect),
        BrainInner::SpatialGraph(s) => draw_spatial_graph(ui, s, painter, rect),
        BrainInner::SpatialBlob(b) => draw_spatial_blob(ui, b, painter, rect),
        BrainInner::GooGraph(g) => draw_goo_graph(ui, g, painter, rect),
    }
}

fn draw_perceptron(
    _ui: &mut egui::Ui,
    p: &PerceptronBrain,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    let diagram = crate::brain_diagram::perceptron_diagram(p, rect.width(), rect.height());
    render_diagram(painter, rect, &diagram);
}

fn draw_spatial_blob(
    _ui: &mut egui::Ui,
    brain: &SpatialBlobBrain,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    let diagram = crate::brain_diagram::spatial_blob_diagram(brain, rect.width(), rect.height());
    render_diagram(painter, rect, &diagram);
}

fn draw_spatial_graph(
    _ui: &mut egui::Ui,
    brain: &SpatialGraphBrain,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    let diagram = crate::brain_diagram::spatial_diagram(brain, rect.width(), rect.height());
    render_diagram(painter, rect, &diagram);
}

fn draw_goo_graph(
    _ui: &mut egui::Ui,
    brain: &GooGraphBrain,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    let diagram = crate::brain_diagram::goo_diagram(brain, rect.width(), rect.height());
    render_diagram(painter, rect, &diagram);
}

pub fn draw_neural_network(
    _ui: &mut egui::Ui,
    nn: &NeuralNetwork,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    let diagram = crate::brain_diagram::mlp_diagram(nn, rect.width(), rect.height());
    render_diagram(painter, rect, &diagram);
}

/// Text-only fallback kept for callers that do not need a diagram.
pub fn draw_brain_summary(
    _ui: &mut egui::Ui,
    brain: &BrainGenome,
    painter: &egui::Painter,
    rect: egui::Rect,
) {
    draw_panel_frame(painter, rect);
    painter.text(
        egui::pos2(rect.min.x + 12.0, rect.min.y + 20.0),
        egui::Align2::LEFT_TOP,
        format!("Brain: {}", brain.kind().as_str()),
        egui::FontId::proportional(16.0),
        egui::Color32::WHITE,
    );
    painter.text(
        egui::pos2(rect.min.x + 12.0, rect.min.y + 44.0),
        egui::Align2::LEFT_TOP,
        format!("Structure: {}", brain.structure_signature()),
        egui::FontId::proportional(12.0),
        egui::Color32::LIGHT_GRAY,
    );
}
