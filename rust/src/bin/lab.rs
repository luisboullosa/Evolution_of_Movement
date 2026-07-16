//! Lab bench: controlled hex-dish simulator for testing architectures, mutations, and environment.
use eframe::egui;
use evo_movement_rust::{
    brain::{apply_lab_structural, LabStructuralOp},
    build_info,
    config::{ArchitecturePolicy, Config},
    individual::Individual,
    lab_dish::{genotype_signature, max_cell_occupancy, resource_background_color, LabDish},
    nn_viz,
};

const DEFAULT_DISH_RADIUS: i32 = 10;
const DEFAULT_HEX_SIZE: f32 = 14.0;

fn lab_default_config() -> Config {
    let mut cfg = Config::default();
    cfg.initial_population = 5;
    cfg.max_population = 200;
    cfg.hex_dish_radius = 0;
    cfg.reproduction_threshold = 20.0;
    cfg.neutral_founder_weights = false;
    cfg
}

struct LabApp {
    dish: LabDish,
    config: Config,
    dish_radius: i32,
    hex_size: f32,
    /// Initial seed population
    seed_count: i32,
    seed_architecture: ArchitecturePolicy,
    /// Insert-more controls
    insert_count: i32,
    insert_architecture: ArchitecturePolicy,
    insert_radius: i32,
    insert_resources: f64,
    starting_resources: f64,
    move_threshold: f64,
    env_speed_multiplier: f32,
    auto_reproduce_on_step: bool,
    selected_individual_id: Option<i32>,
    zoom: f32,
    show_nn: bool,
    status_message: String,
}

impl LabApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let config = lab_default_config();
        let dish = LabDish::new(DEFAULT_DISH_RADIUS, DEFAULT_HEX_SIZE, config.clone());
        let mut app = Self {
            dish,
            config,
            dish_radius: DEFAULT_DISH_RADIUS,
            hex_size: DEFAULT_HEX_SIZE,
            seed_count: 5,
            seed_architecture: ArchitecturePolicy::PerceptronOnly,
            insert_count: 1,
            insert_architecture: ArchitecturePolicy::PerceptronOnly,
            insert_radius: DEFAULT_DISH_RADIUS,
            insert_resources: 10.0,
            starting_resources: 10.0,
            move_threshold: 0.0,
            env_speed_multiplier: 1.0,
            auto_reproduce_on_step: false,
            selected_individual_id: None,
            zoom: 1.0,
            show_nn: true,
            status_message: String::new(),
        };
        app.seed_population();
        app
    }

    fn selected_index(&self) -> Option<usize> {
        let id = self.selected_individual_id?;
        self.dish
            .simulation
            .individuals
            .iter()
            .position(|ind| ind.id == id)
    }

    fn selected_individual(&self) -> Option<&Individual> {
        let idx = self.selected_index()?;
        self.dish.simulation.individuals.get(idx)
    }

    fn reset_dish(&mut self) {
        self.dish = LabDish::new(self.dish_radius, self.hex_size, self.config.clone());
        self.selected_individual_id = None;
        self.status_message = format!("Reset dish r={}", self.dish_radius);
    }

    fn seed_population(&mut self) {
        self.dish.clear_population();
        self.dish.insert_individuals(
            self.seed_count.max(0) as usize,
            self.seed_architecture,
            self.dish_radius,
            self.starting_resources,
        );
        self.status_message = format!(
            "Seeded {} {} individuals",
            self.seed_count,
            self.seed_architecture.label()
        );
    }

    fn insert_more(&mut self) {
        let n = self.insert_count.max(0) as usize;
        self.dish.insert_individuals(
            n,
            self.insert_architecture,
            self.insert_radius,
            self.insert_resources,
        );
        self.status_message = format!("Inserted {n} individuals");
    }

    fn apply_mutation_to_targets(&mut self, op: LabStructuralOp) {
        self.dish.sync_config();
        let cfg = self.dish.simulation.config.clone();
        let targets: Vec<usize> = if let Some(idx) = self.selected_index() {
            vec![idx]
        } else {
            (0..self.dish.simulation.individuals.len()).collect()
        };
        let mut applied = 0usize;
        for idx in targets {
            if let Some(ind) = self.dish.simulation.individuals.get_mut(idx) {
                apply_lab_structural(&mut ind.brain, op, &cfg, &mut self.dish.simulation.rng);
                ind.spatial_runtime = ind.brain.init_spatial_runtime();
                applied += 1;
            }
        }
        self.status_message = format!("{} on {applied} individual(s)", op.label());
    }

    fn step_n(&mut self, n: usize) {
        self.dish.simulation.config = self.config.clone();
        self.dish.sync_config();
        for _ in 0..n {
            self.dish
                .step(self.move_threshold, self.env_speed_multiplier, self.auto_reproduce_on_step);
        }
        self.status_message = format!("Advanced {n} step(s) → t={}", self.dish.timestep);
    }

    fn manual_reproduce(&mut self) {
        self.dish.simulation.config = self.config.clone();
        let born = self.dish.reproduce_eligible();
        self.status_message = format!("Reproduction: {born} offspring");
    }

    fn architecture_combo(ui: &mut egui::Ui, id: &str, value: &mut ArchitecturePolicy) {
        egui::ComboBox::from_id_source(id)
            .selected_text(value.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(value, ArchitecturePolicy::PerceptronOnly, "Perceptron");
                ui.selectable_value(value, ArchitecturePolicy::EvolveMlp, "MLP");
                ui.selectable_value(
                    value,
                    ArchitecturePolicy::EvolveSpatialGraph,
                    "Spatial Graph",
                );
                ui.selectable_value(value, ArchitecturePolicy::EvolveGoo, "GoO Graph");
            });
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Lab Simulator");
        ui.label(
            egui::RichText::new(build_info::VERSION_LABEL)
                .weak()
                .monospace(),
        );
        if !self.status_message.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_GREEN, &self.status_message);
        }

        ui.separator();

        // 1 — Dish & population
        egui::CollapsingHeader::new("1 · Dish & population")
            .id_source("lab_sec_1")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut self.dish_radius, 3..=40).text("Dish radius (cells)"));
                ui.add(egui::Slider::new(&mut self.hex_size, 6.0..=22.0).text("Hex size"));
                ui.add(egui::Slider::new(&mut self.seed_count, 0..=100).text("Initial count (n)"));
                Self::architecture_combo(ui, "seed_arch", &mut self.seed_architecture);
                ui.add(
                    egui::Slider::new(&mut self.starting_resources, 0.0..=100.0)
                        .text("Starting resources"),
                );
                ui.horizontal(|ui| {
                    if ui.button("Reset dish").clicked() {
                        self.reset_dish();
                    }
                    if ui.button("Seed population").clicked() {
                        self.reset_dish();
                        self.seed_population();
                    }
                });
            });

        egui::CollapsingHeader::new("2 · Insert individuals")
            .id_source("lab_sec_2")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut self.insert_count, 1..=50).text("Count (n)"));
                Self::architecture_combo(ui, "insert_arch", &mut self.insert_architecture);
                ui.add(
                    egui::Slider::new(&mut self.insert_radius, 0..=40)
                        .text("Placement radius (0 = whole dish)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.insert_resources, 0.0..=100.0)
                        .text("Starting resources"),
                );
                if ui.button("Insert").clicked() {
                    self.insert_more();
                }
            });

        egui::CollapsingHeader::new("3 · Time")
            .id_source("lab_sec_3")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.move_threshold, -2.0..=2.0).text("Move threshold"),
                );
                ui.add(
                    egui::Slider::new(&mut self.env_speed_multiplier, 0.0..=5.0)
                        .text("Env speed multiplier"),
                );
                ui.checkbox(
                    &mut self.auto_reproduce_on_step,
                    "Auto-reproduce during steps",
                );
                ui.horizontal(|ui| {
                    if ui.button("Step ×1").clicked() {
                        self.step_n(1);
                    }
                    if ui.button("Step ×100").clicked() {
                        self.step_n(100);
                    }
                });
                ui.label(format!("Timestep: {}", self.dish.timestep));
                ui.label(format!(
                    "Population: {}",
                    self.dish.simulation.individuals.len()
                ));
                ui.label(format!(
                    "Avg move signal: {:.3}",
                    self.dish.last_avg_move_signal
                ));
            });

        egui::CollapsingHeader::new("4 · Mutations")
            .id_source("lab_sec_4")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("Target: selected individual, or all if none selected.");
                ui.add(
                    egui::Slider::new(&mut self.config.mutation_rate, 0.0..=1.0)
                        .text("Mutation rate"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.mutation_sigma, 0.0..=2.0)
                        .text("Mutation sigma"),
                );
                egui::Grid::new("mutation_grid")
                    .num_columns(2)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        for op in LabStructuralOp::ALL {
                            if ui.button(op.label()).clicked() {
                                self.apply_mutation_to_targets(*op);
                            }
                            ui.end_row();
                        }
                    });
            });

        egui::CollapsingHeader::new("5 · Reproduction")
            .id_source("lab_sec_5")
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.config.reproduction_threshold, 1.0..=100.0)
                        .text("Resource minimum"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.reproduction_cost, 0.0..=50.0)
                        .text("Reproduction cost"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.max_population, 1..=500)
                        .text("Max population"),
                );
                if ui.button("Reproduce (eligible now)").clicked() {
                    self.manual_reproduce();
                }
            });

        egui::CollapsingHeader::new("6 · Environment (Perlin)")
            .id_source("lab_sec_6")
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.config.resource_scale, 0.0..=5.0)
                        .text("Resource scale"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.env_time_speed, 0.0..=0.2)
                        .text("Env time speed"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.env_time_amplitude, 0.0..=3.0)
                        .text("Time amplitude"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.env_perlin_spatial_scale, 0.001..=1.0)
                        .text("Spatial scale"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.env_perlin_time_scale, 0.001..=1.0)
                        .text("Time scale"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.perlin_octaves, 1..=8).text("Octaves"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.perlin_persistence, 0.1..=1.0)
                        .text("Persistence"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.perlin_lacunarity, 1.0..=4.0)
                        .text("Lacunarity"),
                );
                ui.add(egui::Slider::new(&mut self.config.seed, 0..=u32::MAX).text("Seed"));
                if ui.button("Refresh environment").clicked() {
                    self.dish.simulation.config = self.config.clone();
                    self.dish.update_resource_grid();
                    self.status_message = "Environment refreshed".to_string();
                }
            });

        egui::CollapsingHeader::new("7 · Metabolism & movement")
            .id_source("lab_sec_7")
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.config.basal_metabolic_cost, 0.0..=1.0)
                        .text("Basal metabolic cost / step"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.maintenance_cost_per_node, 0.0..=0.5)
                        .text("Maintenance / node / step"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.movement_cost_factor, 0.0..=10.0)
                        .text("Movement cost factor"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.death_risk, 0.0..=0.1).text("Death risk"),
                );
            });

        ui.separator();
        ui.checkbox(&mut self.show_nn, "Show brain diagram");
        ui.add(egui::Slider::new(&mut self.zoom, 0.4..=2.5).text("Zoom"));

        if let Some(ind) = self.selected_individual() {
            ui.separator();
            ui.heading(format!("Individual #{}", ind.id));
            ui.label(format!("Resources: {:.2}", ind.resources));
            ui.label(format!("Age: {}", ind.age));
            ui.label(format!("Brain: {:?}", ind.brain.kind()));
            ui.label(format!("Complexity: {}", ind.complexity()));
            ui.label(format!("Genotype: {}", genotype_signature(ind)));
        }
    }

    fn draw_hex_view(&mut self, ui: &mut egui::Ui) {
        let dish = &self.dish;
        let hex_size = dish.hex_grid.hex_size * self.zoom;

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for coord in dish.hex_grid.cells.keys() {
            let (x, y) = coord.to_pixel(hex_size);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let (min_x, max_x, min_y, max_y) = if min_x.is_finite() {
            (min_x, max_x, min_y, max_y)
        } else {
            (0.0, 1.0, 0.0, 1.0)
        };

        let width_px = (max_x - min_x) + 2.0 * hex_size;
        let height_px = (max_y - min_y) + 2.0 * hex_size;
        let padding = hex_size;
        let canvas_size = egui::vec2(width_px + 20.0, height_px + 20.0);

        let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click());
        let offset = response.rect.min + egui::vec2(10.0, 10.0);
        let max_occ = max_cell_occupancy(&dish.hex_grid.cells);

        let mut click_hex: Option<usize> = None;

        for (coord, cell) in &dish.hex_grid.cells {
            let (x, y) = coord.to_pixel(hex_size);
            let screen_pos =
                offset + egui::vec2(x - min_x + padding, y - min_y + padding);
            let points = hexagon_points(screen_pos, hex_size);
            let rgb = resource_background_color(cell.resource_level);
            painter.add(egui::Shape::convex_polygon(
                points,
                egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
                egui::Stroke::new(0.5, egui::Color32::DARK_GRAY),
            ));

            if !cell.individual_indices.is_empty() {
                let t = if max_occ > 0 {
                    cell.individual_indices.len() as f32 / max_occ as f32
                } else {
                    0.5
                };
                let radius = (hex_size * 0.35 * (0.6 + 0.4 * t)).max(3.0);
                if let Some(&first_idx) = cell.individual_indices.first() {
                    if let Some(ind) = dish.simulation.individuals.get(first_idx) {
                        let c = ind.lineage_color;
                        painter.circle_filled(
                            screen_pos,
                            radius,
                            egui::Color32::from_rgb(c[0], c[1], c[2]),
                        );
                        if cell.individual_indices.len() > 1 {
                            painter.text(
                                screen_pos + egui::vec2(0.0, -radius - 4.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!("{}", cell.individual_indices.len()),
                                egui::FontId::proportional(10.0),
                                egui::Color32::WHITE,
                            );
                        }
                    }
                }
            }

            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let dx = pos.x - screen_pos.x;
                    let dy = pos.y - screen_pos.y;
                    if dx * dx + dy * dy < hex_size * hex_size {
                        if let Some(&idx) = dish
                            .hex_grid
                            .cells
                            .get(coord)
                            .and_then(|c| c.individual_indices.first())
                        {
                            click_hex = Some(idx);
                        } else {
                            self.selected_individual_id = None;
                        }
                    }
                }
            }
        }

        if let Some(idx) = click_hex {
            if let Some(ind) = dish.simulation.individuals.get(idx) {
                self.selected_individual_id = Some(ind.id);
            }
        }
    }
}

impl eframe::App for LabApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("lab_controls")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_controls(ui);
                });
            });

        if self.show_nn {
            egui::SidePanel::right("brain_panel")
                .resizable(true)
                .default_width(280.0)
                .show(ctx, |ui| {
                    ui.heading("Brain");
                    if let Some(ind) = self.selected_individual() {
                        nn_viz::draw_brain_panel(ui, &ind.brain);
                    } else {
                        ui.label("Select an individual on the dish.");
                    }
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hex dish lab");
            ui.label(format!(
                "r={} · t={} · n={}",
                self.dish.radius_cells,
                self.dish.timestep,
                self.dish.simulation.individuals.len()
            ));
            egui::ScrollArea::both().show(ui, |ui| {
                self.draw_hex_view(ui);
            });
        });
    }
}

fn hexagon_points(center: egui::Pos2, size: f32) -> Vec<egui::Pos2> {
    let mut points = Vec::new();
    for i in 0..6 {
        let angle = std::f32::consts::PI / 3.0 * i as f32 + std::f32::consts::PI / 6.0;
        points.push(egui::pos2(
            center.x + size * angle.cos(),
            center.y + size * angle.sin(),
        ));
    }
    points
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title(format!(
                "evo-movement lab · {}",
                build_info::VERSION_LABEL
            )),
        ..Default::default()
    };
    eframe::run_native(
        &format!("evo-movement lab · {}", build_info::VERSION_LABEL),
        options,
        Box::new(|cc| Box::new(LabApp::new(cc))),
    )
}
