/// Interactive evolution simulation viewer with GUI
use eframe::egui;
use evo_movement_rust::{
    cladogram::Cladogram,
    config::Config,
    hex_grid::{HexCoord, HexGrid},
    individual::Individual,
    neural_network::NeuralNetwork,
    nn_viz,
    save_state::SimulationState,
    simulation::Simulation,
};
use rand::Rng;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title("Evolution of Movement - Interactive Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "Evolution of Movement",
        options,
        Box::new(|cc| Box::new(EvolutionApp::new(cc))),
    )
}

struct PetriDish {
    id: usize,
    hex_grid: HexGrid,
    simulation: Simulation,
    cladogram: Cladogram,
    name: String,
    timestep: i32,
}

impl PetriDish {
    fn new(id: usize, width: usize, height: usize, hex_size: f32, config: Config) -> Self {
        let hex_grid = HexGrid::new(width, height, hex_size);
        let simulation = Simulation::new(&config);
        let cladogram = Cladogram::new();

        Self {
            id,
            hex_grid,
            simulation,
            cladogram,
            name: format!("Dish {}", id),
            timestep: 0,
        }
    }

    fn init_population(&mut self, count: usize) {
        use rand::thread_rng;
        let mut rng = thread_rng();

        // Generate unique colors for each founder
        for i in 0..count {
            let color = Self::generate_color(i);
            let mut nn = NeuralNetwork::new(
                self.simulation.config.nn_nodes.max(0) as usize,
                self.simulation.config.nn_inputs.max(0) as usize,
                self.simulation.config.nn_layers,
                self.simulation.config.nn_nodes2.max(0) as usize,
            );

            // Apply initial mutations if configured
            if self.simulation.config.radiate > 0 {
                use rand_distr::{Cauchy, Distribution};
                let cauchy = Cauchy::new(0.0, self.simulation.config.mutation_sigma).unwrap();
                let mutation_rate = self.simulation.config.mutation_rate;
                let events = self.simulation.config.radiate as usize;

                for w in nn.weights.iter_mut() {
                    for _ in 0..events {
                        if rng.gen::<f64>() < mutation_rate {
                            *w += cauchy.sample(&mut rng);
                        }
                    }
                }
                for w in nn.weights_multi.iter_mut() {
                    for _ in 0..events {
                        if rng.gen::<f64>() < mutation_rate {
                            *w += cauchy.sample(&mut rng);
                        }
                    }
                }
            }

            let pos = rng.gen_range(0..self.hex_grid.width * self.hex_grid.height);
            let id = self.simulation.next_id;
            self.simulation.next_id += 1;

            let ind = Individual::new_with_color(nn, pos, id, color);
            self.simulation.individuals.push(ind);
            self.cladogram.add_founder(id, color);

            // Add to hex grid
            let coord = HexCoord::from_flat(pos, self.hex_grid.width);
            if let Some(cell) = self.hex_grid.get_cell_mut(coord) {
                cell.individuals.push(self.simulation.individuals.len() - 1);
            }
        }

        // Initialize environment resources
        self.update_resource_grid();
    }

    fn update_resource_grid(&mut self) {
        // Update hex grid cells with environment resources
        for r in 0..self.hex_grid.height {
            for q in 0..self.hex_grid.width {
                let flat_idx = r * self.hex_grid.width + q;
                if flat_idx < self.simulation.env.resources.len() {
                    let resource = self.simulation.env.resources[flat_idx];
                    let coord = HexCoord::new(q as i32, r as i32);
                    if let Some(cell) = self.hex_grid.get_cell_mut(coord) {
                        cell.resource_level = resource;
                    }
                }
            }
        }
    }

    fn step(&mut self) {
        // Step the simulation
        // Note: The actual simulation stepping logic would need to be
        // integrated here. For now, we'll implement a simplified version.
        
        // Clear individual positions in hex grid
        for cell in self.hex_grid.cells.values_mut() {
            cell.individuals.clear();
        }

        // Advance time
        self.timestep += 1;
        self.cladogram.advance_time();

        // Update environment (time-varying)
        self.simulation.env.step(
            self.simulation.config.seed as u64,
            self.simulation.config.env_time_speed,
            self.simulation.config.env_time_amplitude,
            self.simulation.config.env_perlin_spatial_scale,
            self.simulation.config.env_perlin_time_scale,
            self.simulation.config.resource_scale,
        );

        // Update individuals and rebuild hex grid positions
        for (idx, ind) in self.simulation.individuals.iter_mut().enumerate() {
            ind.age += 1;
            
            // Add back to hex grid
            let coord = HexCoord::from_flat(ind.position, self.hex_grid.width);
            if let Some(cell) = self.hex_grid.get_cell_mut(coord) {
                cell.individuals.push(idx);
            }
        }

        // Update resource display
        self.update_resource_grid();
    }

    fn generate_color(index: usize) -> [u8; 3] {
        // Generate visually distinct colors using golden angle
        let hue = (index as f32 * 137.508) % 360.0;
        Self::hsv_to_rgb(hue, 0.7, 0.95)
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
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
}

struct EvolutionApp {
    petri_dishes: Vec<PetriDish>,
    selected_dish: usize,
    selected_individual: Option<usize>,
    paused: bool,
    simulation_speed: f32,
    timesteps_per_frame: usize,
    show_nn_diagram: bool,
    show_cladogram: bool,
    config: Config,
}

impl EvolutionApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut config = Config::default();
        config.grid_width = 50;
        config.initial_population = 10;
        config.steps = 10000;

        let mut dishes = Vec::new();
        let dish = PetriDish::new(0, 50, 30, 10.0, config.clone());
        dishes.push(dish);

        Self {
            petri_dishes: dishes,
            selected_dish: 0,
            selected_individual: None,
            paused: true,
            simulation_speed: 1.0,
            timesteps_per_frame: 1,
            show_nn_diagram: false,
            show_cladogram: false,
            config,
        }
    }

    fn add_petri_dish(&mut self) {
        let id = self.petri_dishes.len();
        let dish = PetriDish::new(id, 50, 30, 10.0, self.config.clone());
        self.petri_dishes.push(dish);
        self.selected_dish = id;
    }

    fn initialize_population(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            dish.init_population(self.config.initial_population as usize);
        }
    }

    fn step_simulation(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            for _ in 0..self.timesteps_per_frame {
                dish.step();
            }
        }
    }

    fn restart_simulation(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            // Clear all individuals
            dish.simulation.individuals.clear();
            dish.cladogram = Cladogram::new();
            dish.timestep = 0;
            
            // Clear hex grid
            for cell in dish.hex_grid.cells.values_mut() {
                cell.individuals.clear();
            }
            
            // Reinitialize
            dish.init_population(self.config.initial_population as usize);
        }
    }

    fn save_state(&self, filename: &str) {
        if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
            let state = SimulationState::from_individuals_and_grid(
                &dish.simulation.individuals,
                &dish.hex_grid,
                &dish.cladogram,
                &dish.simulation.config,
                dish.timestep,
                dish.simulation.next_id,
            );

            match state.save_to_file(filename) {
                Ok(_) => println!("State saved to {}", filename),
                Err(e) => eprintln!("Failed to save state: {}", e),
            }
        }
    }

    fn load_state(&mut self, filename: &str) {
        match SimulationState::load_from_file(filename) {
            Ok(state) => {
                // Reconstruct the petri dish from the saved state
                let dish_id = self.selected_dish;
                let mut dish = PetriDish::new(
                    dish_id,
                    state.grid_width,
                    state.grid_height,
                    state.hex_size,
                    state.config.clone(),
                );

                dish.timestep = state.timestep;
                dish.simulation.next_id = state.next_id;

                // Reconstruct individuals
                for ind_state in state.individuals {
                    let mut nn = NeuralNetwork::new(
                        ind_state.nn_nodes,
                        ind_state.nn_inputs,
                        ind_state.nn_layers,
                        ind_state.nn_nodes2,
                    );
                    nn.weights = ind_state.nn_weights;
                    nn.weights_multi = ind_state.nn_weights_multi;
                    nn.biases = ind_state.nn_biases;
                    nn.weights2 = ind_state.nn_weights2;
                    nn.biases2 = ind_state.nn_biases2;

                    let mut ind = Individual::new_with_color(
                        nn,
                        ind_state.position,
                        ind_state.id,
                        ind_state.lineage_color,
                    );
                    ind.resources = ind_state.resources;
                    ind.age = ind_state.age;
                    ind.parent_id = ind_state.parent_id;

                    dish.simulation.individuals.push(ind);
                }

                // Reconstruct cladogram
                for node_state in state.cladogram_nodes {
                    use evo_movement_rust::cladogram::LineageNode;
                    dish.cladogram.nodes.insert(
                        node_state.id,
                        LineageNode {
                            id: node_state.id,
                            parent_id: node_state.parent_id,
                            children: node_state.children,
                            birth_time: node_state.birth_time,
                            death_time: node_state.death_time,
                            color: node_state.color,
                        },
                    );
                }
                dish.cladogram.current_time = state.timestep;

                // Reconstruct hex cells
                for ((q, r), cell_state) in state.hex_cells {
                    let coord = HexCoord::new(q, r);
                    if let Some(cell) = dish.hex_grid.get_cell_mut(coord) {
                        cell.resource_level = cell_state.resource_level;
                        cell.individuals = cell_state.individual_indices;
                    }
                }

                self.petri_dishes[self.selected_dish] = dish;
                println!("State loaded from {}", filename);
            }
            Err(e) => eprintln!("Failed to load state: {}", e),
        }
    }
}

impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top menu bar
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New Simulation").clicked() {
                        self.restart_simulation();
                    }
                    if ui.button("Save State").clicked() {
                        self.save_state("simulation_state.json");
                    }
                    if ui.button("Load State").clicked() {
                        self.load_state("simulation_state.json");
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_nn_diagram, "Show Neural Network");
                    ui.checkbox(&mut self.show_cladogram, "Show Cladogram");
                });

                ui.separator();

                // Simulation controls
                if self.paused {
                    if ui.button("▶ Play").clicked() {
                        self.paused = false;
                    }
                } else {
                    if ui.button("⏸ Pause").clicked() {
                        self.paused = true;
                    }
                }

                if ui.button("⏭ Step").clicked() {
                    self.step_simulation();
                }

                if ui.button("🔄 Restart").clicked() {
                    self.restart_simulation();
                }

                ui.separator();

                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.simulation_speed, 0.1..=10.0).text("x"));

                ui.separator();

                if ui.button("➕ Add Petri Dish").clicked() {
                    self.add_petri_dish();
                }

                if ui.button("🧬 Initialize Population").clicked() {
                    self.initialize_population();
                }
            });
        });

        // Left panel - Controls and info
        egui::SidePanel::left("control_panel").min_width(250.0).show(ctx, |ui| {
            ui.heading("Controls");

            ui.separator();

            egui::ComboBox::from_label("Active Dish")
                .selected_text(format!("Dish {}", self.selected_dish))
                .show_ui(ui, |ui| {
                    for (i, dish) in self.petri_dishes.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_dish, i, &dish.name);
                    }
                });

            ui.separator();

            ui.label("Simulation Settings");
            ui.add(egui::Slider::new(&mut self.config.initial_population, 1..=100)
                .text("Initial Population"));
            ui.add(egui::Slider::new(&mut self.timesteps_per_frame, 1..=10)
                .text("Steps per Frame"));

            ui.separator();

            if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                ui.heading("Statistics");
                ui.label(format!("Individuals: {}", dish.simulation.individuals.len()));
                ui.label(format!("Timestep: {}", dish.timestep));

                if let Some(idx) = self.selected_individual {
                    if let Some(ind) = dish.simulation.individuals.get(idx) {
                        ui.separator();
                        ui.heading("Selected Individual");
                        ui.label(format!("ID: {}", ind.id));
                        ui.label(format!("Resources: {:.2}", ind.resources));
                        ui.label(format!("Age: {}", ind.age));
                        ui.label(format!("Position: {}", ind.position));
                    }
                }
            }
        });

        // Right panel - Neural Network / Cladogram
        if self.show_nn_diagram || self.show_cladogram {
            egui::SidePanel::right("viz_panel").min_width(400.0).show(ctx, |ui| {
                if self.show_nn_diagram {
                    ui.heading("Neural Network");
                    
                    if let Some(idx) = self.selected_individual {
                        if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                            if let Some(ind) = dish.simulation.individuals.get(idx) {
                                let available = ui.available_size();
                                let rect = egui::Rect::from_min_size(
                                    ui.cursor().min,
                                    egui::vec2(available.x - 20.0, 300.0),
                                );
                                
                                let (response, painter) = ui.allocate_painter(
                                    rect.size(),
                                    egui::Sense::hover(),
                                );
                                
                                nn_viz::draw_neural_network(ui, &ind.nn, &painter, response.rect);
                                
                                ui.add_space(310.0);
                                
                                // Show some weight statistics
                                ui.separator();
                                ui.label("Weight Statistics:");
                                let (nodes, inputs, nodes2, layers) = ind.nn.shape();
                                ui.label(format!("Architecture: {}x{} (layers: {})", nodes, inputs, layers));
                                if nodes2 > 0 {
                                    ui.label(format!("Hidden2: {}", nodes2));
                                }
                            } else {
                                ui.label("Select an individual to see its neural network");
                            }
                        }
                    } else {
                        ui.label("Select an individual to see its neural network");
                    }
                }

                if self.show_cladogram {
                    ui.separator();
                    ui.heading("Cladogram");
                    ui.label("Phylogenetic tree visualization");
                    
                    if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                        ui.label(format!("Total nodes: {}", dish.cladogram.nodes.len()));
                        
                        if let Some(idx) = self.selected_individual {
                            if let Some(ind) = dish.simulation.individuals.get(idx) {
                                let lineage = dish.cladogram.get_lineage_path(ind.id);
                                ui.label(format!("Lineage depth: {}", lineage.len()));
                                ui.label(format!("Ancestor IDs: {:?}", &lineage[..lineage.len().min(10)]));
                            }
                        }
                    }
                }
            });
        }

        // Central panel - Main visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            let dish_name = if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                dish.name.clone()
            } else {
                return;
            };

            ui.heading(&dish_name);

            // Get dish data before the closure
            let dish_opt = self.petri_dishes.get(self.selected_dish);
            if dish_opt.is_none() {
                return;
            }
            let dish = dish_opt.unwrap();

            let (width_px, height_px) = dish.hex_grid.pixel_dimensions();
            let available_size = ui.available_size();
            let scale = (available_size.x / width_px).min(available_size.y / height_px);
            let hex_size = dish.hex_grid.hex_size;
            let cells = &dish.hex_grid.cells;
            let individuals = &dish.simulation.individuals;
            let selected_ind = self.selected_individual;

            let mut clicked_hex: Option<HexCoord> = None;

            egui::ScrollArea::both().show(ui, |ui| {
                let (response, painter) = ui.allocate_painter(
                    egui::vec2(width_px * scale, height_px * scale),
                    egui::Sense::click_and_drag(),
                );

                // Draw hexagonal grid
                for (coord, cell) in cells {
                    let (x, y) = coord.to_pixel(hex_size);
                    let screen_pos = response.rect.min + egui::vec2(x * scale, y * scale);

                    // Draw hexagon
                    let hex_points = hexagon_points(screen_pos, hex_size * scale);
                    
                    // Color based on resource level (light = resource)
                    let brightness = (cell.resource_level * 255.0).min(255.0).max(0.0) as u8;
                    let color = egui::Color32::from_rgb(brightness, brightness, 100);
                    
                    painter.add(egui::Shape::convex_polygon(
                        hex_points,
                        color,
                        egui::Stroke::new(0.5, egui::Color32::GRAY),
                    ));

                    // Draw individuals
                    for &ind_idx in &cell.individuals {
                        if let Some(ind) = individuals.get(ind_idx) {
                            let color = egui::Color32::from_rgb(
                                ind.lineage_color[0],
                                ind.lineage_color[1],
                                ind.lineage_color[2],
                            );
                            painter.circle_filled(screen_pos, 3.0 * scale, color);

                            if Some(ind_idx) == selected_ind {
                                painter.circle_stroke(
                                    screen_pos,
                                    5.0 * scale,
                                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                                );
                            }
                        }
                    }
                }

                // Handle clicks for selection
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let rel_pos = pos - response.rect.min;
                        let hex_coord = HexCoord::from_pixel(
                            rel_pos.x / scale,
                            rel_pos.y / scale,
                            hex_size,
                        );
                        clicked_hex = Some(hex_coord);
                    }
                }
            });

            // Handle selection outside of the closure
            if let Some(coord) = clicked_hex {
                if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                    if let Some(cell) = dish.hex_grid.get_cell(coord) {
                        if !cell.individuals.is_empty() {
                            self.selected_individual = Some(cell.individuals[0]);
                        }
                    }
                }
            }
        });

        // Run simulation if not paused
        if !self.paused {
            self.step_simulation();
            ctx.request_repaint();
        }
    }
}

fn hexagon_points(center: egui::Pos2, size: f32) -> Vec<egui::Pos2> {
    let mut points = Vec::new();
    for i in 0..6 {
        let angle = std::f32::consts::PI / 3.0 * i as f32;
        let x = center.x + size * angle.cos();
        let y = center.y + size * angle.sin();
        points.push(egui::pos2(x, y));
    }
    points
}
