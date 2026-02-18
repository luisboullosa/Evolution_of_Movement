/// Interactive evolution simulation viewer with GUI
use eframe::egui;
use evo_movement_rust::{
    cladogram::Cladogram,
    config::Config,
    gpu_resource::GpuResourceUpdater,
    hex_grid::{HexCell, HexCoord, HexGrid},
    individual::Individual,
    neural_network::NeuralNetwork,
    nn_viz,
    perlin::Perlin,
    save_state::SimulationState,
    simulation::Simulation,
};
use std::collections::{HashMap, HashSet};

const DEFAULT_DISH_RADIUS_CELLS: i32 = 100;

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
    hex_grid: HexGrid,
    simulation: Simulation,
    cladogram: Cladogram,
    name: String,
    timestep: i32,
    radius_cells: i32,
    active_coords: Vec<HexCoord>,
    coord_to_index: HashMap<HexCoord, usize>,
    resource_base: Vec<f64>,
    gpu_resource_updater: Option<GpuResourceUpdater>,
    last_avg_move_signal: f64,
}

impl PetriDish {
    fn new_hex_dish(id: usize, radius_cells: i32, hex_size: f32, config: Config) -> Self {
        let hex_grid = HexGrid::new_hex_dish(radius_cells, hex_size);
        Self::from_grid(id, hex_grid, config, radius_cells)
    }

    fn from_grid(id: usize, hex_grid: HexGrid, mut config: Config, radius_cells: i32) -> Self {
        let mut active_coords: Vec<HexCoord> = hex_grid.cells.keys().copied().collect();
        active_coords.sort_by_key(|c| (c.r, c.q));
        let mut coord_to_index = HashMap::with_capacity(active_coords.len());
        for (idx, coord) in active_coords.iter().enumerate() {
            coord_to_index.insert(*coord, idx);
        }

        // Interactive dishes require one resource sensor per hex edge (6),
        // plus occupancy and internal resources.
        config.nn_inputs = config.nn_inputs.max(8);
        config.grid_width = active_coords.len().max(1);
        let simulation = Simulation::new(&config);
        let cladogram = Cladogram::new();

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

        let gpu_resource_updater = GpuResourceUpdater::new(&active_coords, &resource_base).ok();

        let mut dish = Self {
            hex_grid,
            simulation,
            cladogram,
            name: format!("Dish {}", id),
            timestep: 0,
            radius_cells,
            active_coords,
            coord_to_index,
            resource_base,
            gpu_resource_updater,
            last_avg_move_signal: 0.0,
        };
        dish.update_resource_grid();
        dish
    }

    fn init_population(&mut self, count: usize) {
        let start = self.simulation.individuals.len();
        self.simulation.init_population(count);

        for i in start..self.simulation.individuals.len() {
            if let Some(ind) = self.simulation.individuals.get_mut(i) {
                let color = Self::generate_color(i - start);
                ind.lineage_color = color;
                ind.parent_id = None;
                self.cladogram.add_founder(ind.id, color);
            }
        }

        self.rebuild_hex_occupancy();

        // Initialize environment resources
        self.update_resource_grid();
    }

    fn update_resource_grid(&mut self) {
        let seed = self.simulation.config.seed as u64;
        let spatial_scale = self.simulation.config.env_perlin_spatial_scale;
        let time_scale = self.simulation.config.env_perlin_time_scale;
        let time_amplitude = self.simulation.config.env_time_amplitude;
        let resource_scale = self.simulation.config.resource_scale;

        if let Some(gpu) = self.gpu_resource_updater.as_mut() {
            if let Some(resources) = gpu.compute(
                seed as u32,
                self.simulation.env.perlin_time as f32,
                spatial_scale as f32,
                time_scale as f32,
                time_amplitude as f32,
                resource_scale as f32,
            ) {
                for (idx, coord) in self.active_coords.iter().enumerate() {
                    let resource = resources.get(idx).copied().unwrap_or(0.0).max(0.0);
                    if let Some(slot) = self.simulation.env.resources.get_mut(idx) {
                        *slot = resource;
                    }
                    if let Some(cell) = self.hex_grid.get_cell_mut(*coord) {
                        cell.resource_level = resource;
                    }
                }
                return;
            }
        }

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

    fn rebuild_hex_occupancy(&mut self) {
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

    fn step(&mut self, move_threshold: f64) {
        if self.active_coords.is_empty() {
            return;
        }

        let before_ids: HashSet<i32> = self
            .simulation
            .individuals
            .iter()
            .map(|ind| ind.id)
            .collect();

        // Advance time
        self.timestep += 1;
        self.cladogram.advance_time();
        self.simulation.env.perlin_time += self.simulation.config.env_time_speed;

        // Refresh 2D spatial + temporal resources for this dish
        self.update_resource_grid();

        // Run full simulation dynamics with one-hex directional movement.
        let width = self.active_coords.len();
        let mut occupancy = vec![0i32; width];
        for ind in &self.simulation.individuals {
            let pos = ind.position % width;
            occupancy[pos] = occupancy[pos].saturating_add(1);
        }

        self.simulation.last_spend.resize(self.simulation.individuals.len(), 0.0);
        self.simulation.last_delta.resize(self.simulation.individuals.len(), 0);
        self.simulation.last_movers = 0;
        let mut move_signal_sum = 0.0f64;
        let mut move_signal_count = 0usize;

        for (i, ind) in self.simulation.individuals.iter_mut().enumerate() {
            let pos = ind.position % width;
            let coord = self.active_coords[pos];
            let neighbors = coord.neighbors();

            let center = self.simulation.env.resources.get(pos).copied().unwrap_or(0.0);
            let mut edge_resources = [center; 6];
            for (edge, ncoord) in neighbors.iter().enumerate() {
                edge_resources[edge] = self
                    .coord_to_index
                    .get(ncoord)
                    .and_then(|idx| self.simulation.env.resources.get(*idx))
                    .copied()
                    .unwrap_or(center);
            }
            let local_population = if self.simulation.config.nn_use_occupancy {
                occupancy[pos]
            } else {
                0
            };

            let decision = ind.decide_action(
                edge_resources,
                local_population,
                self.simulation.config.movement_cost_factor,
                &mut self.simulation.rng,
                &mut self.simulation.nn_hidden_buf,
                &mut self.simulation.nn_out2_buf,
            );
            move_signal_sum += decision.move_signal;
            move_signal_count = move_signal_count.saturating_add(1);

            if decision.gathered > 0.0 {
                if let Some(resource_slot) = self.simulation.env.resources.get_mut(pos) {
                    *resource_slot = (*resource_slot - decision.gathered).max(0.0);
                }
            }

            if decision.move_one_step && decision.move_signal > move_threshold {
                self.simulation.last_spend[i] = decision.movement_spend;
                self.simulation.last_movers = self.simulation.last_movers.saturating_add(1);

                let mut sector = ((decision.direction_degrees.rem_euclid(360.0) / 60.0).floor() as usize) % 6;
                if sector >= 6 {
                    sector = 0;
                }
                let target_coord = neighbors[sector];
                if let Some(target_idx) = self.coord_to_index.get(&target_coord).copied() {
                    ind.position = target_idx;
                    self.simulation.last_delta[i] = 1;
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

        let before = self.simulation.individuals.len();
        for idx in 0..before {
            if idx >= self.simulation.individuals.len() {
                break;
            }
            let res = self.simulation.individuals[idx].resources;
            if res > self.simulation.config.reproduction_threshold {
                if self.simulation.config.max_population > 0
                    && (self.simulation.individuals.len() as i32) >= self.simulation.config.max_population
                {
                    break;
                }
                let child_opt = {
                    let ind_mut = &mut self.simulation.individuals[idx];
                    ind_mut.reproduce(
                        &mut self.simulation.rng,
                        &mut self.simulation.next_id,
                        self.simulation.config.reproduction_threshold,
                        self.simulation.config.reproduction_cost,
                        self.simulation.config.mutation_rate,
                        self.simulation.config.mutation_sigma,
                    )
                };
                if let Some(child) = child_opt {
                    self.simulation.individuals.push(child);
                }
            }
        }

        for ind in &mut self.simulation.individuals {
            let total_nodes = ind.nn.nodes + ind.nn.nodes2;
            ind.resources -= (total_nodes as f64) * self.simulation.config.maintenance_cost_per_node;
        }

        use rand::distributions::{Distribution, Uniform};
        let ud = Uniform::new(0.0f64, 1.0f64);
        self.simulation.individuals.retain(|ind| {
            if ind.resources <= 0.0 {
                return false;
            }
            if self.simulation.config.death_risk > 0.0 && ind.age > 0 {
                let p = 1.0 - (-(self.simulation.config.death_risk * (ind.age as f64))).exp();
                if ud.sample(&mut self.simulation.rng) < p {
                    return false;
                }
            }
            true
        });

        let after_ids: HashSet<i32> = self
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
                            .map(|p| Cladogram::genetic_distance(&p.nn, &ind.nn))
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
        for dead_id in before_ids.difference(&after_ids) {
            self.cladogram.mark_death(*dead_id);
        }

        self.rebuild_hex_occupancy();
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
    selected_individual_id: Option<i32>,
    selected_hex: Option<HexCoord>,
    paused: bool,
    paused_before_drag: bool,
    is_dragging_individual: bool,
    simulation_speed: f32,
    zoom: f32,
    timesteps_per_frame: usize,
    move_threshold: f64,
    show_nn_diagram: bool,
    show_cladogram: bool,
    phylogeny_zoom: f32,
    phylogeny_height: f32,
    phylogeny_max_render_nodes: usize,
    show_config_editor: bool,
    config_radius_cells: i32,
    config_hex_size: f32,
    config: Config,
}

impl EvolutionApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut config = Config::default();
        config.initial_population = 10;
        config.steps = 10000;

        let mut dishes = Vec::new();
        let dish = PetriDish::new_hex_dish(0, DEFAULT_DISH_RADIUS_CELLS, 10.0, config.clone());
        dishes.push(dish);

        Self {
            petri_dishes: dishes,
            selected_dish: 0,
            selected_individual_id: None,
            selected_hex: None,
            paused: true,
            paused_before_drag: true,
            is_dragging_individual: false,
            simulation_speed: 1.0,
            zoom: 1.0,
            timesteps_per_frame: 1,
            move_threshold: 0.0,
            show_nn_diagram: false,
            show_cladogram: false,
            phylogeny_zoom: 1.0,
            phylogeny_height: 300.0,
            phylogeny_max_render_nodes: 1500,
            show_config_editor: false,
            config_radius_cells: DEFAULT_DISH_RADIUS_CELLS,
            config_hex_size: 10.0,
            config,
        }
    }

    fn add_petri_dish(&mut self) {
        let id = self.petri_dishes.len();
        let dish = PetriDish::new_hex_dish(id, DEFAULT_DISH_RADIUS_CELLS, 10.0, self.config.clone());
        self.petri_dishes.push(dish);
        self.selected_dish = id;
    }

    fn initialize_population(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            dish.init_population(self.config.initial_population as usize);
        }
    }

    fn start_new_simulation_from_config(&mut self) {
        let radius = self.config_radius_cells.max(1);
        let hex_size = self.config_hex_size.max(2.0);
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            *dish = PetriDish::new_hex_dish(self.selected_dish, radius, hex_size, self.config.clone());
            dish.init_population(self.config.initial_population as usize);
            self.selected_individual_id = None;
        }
    }

    fn add_dish_from_config(&mut self) {
        let radius = self.config_radius_cells.max(1);
        let hex_size = self.config_hex_size.max(2.0);
        let id = self.petri_dishes.len();
        let mut dish = PetriDish::new_hex_dish(id, radius, hex_size, self.config.clone());
        dish.init_population(self.config.initial_population as usize);
        self.petri_dishes.push(dish);
        self.selected_dish = id;
        self.selected_individual_id = None;
    }

    fn selected_individual_index(&self, dish: &PetriDish) -> Option<usize> {
        let selected_id = self.selected_individual_id?;
        dish.simulation
            .individuals
            .iter()
            .position(|ind| ind.id == selected_id)
    }

    fn step_simulation(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            for _ in 0..self.timesteps_per_frame {
                dish.step(self.move_threshold);
            }
        }
    }

    fn restart_simulation(&mut self) {
        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
            let radius = dish.radius_cells;
            let hex_size = dish.hex_grid.hex_size;
            *dish = PetriDish::new_hex_dish(self.selected_dish, radius, hex_size, self.config.clone());
            dish.init_population(self.config.initial_population as usize);
        }
        self.selected_individual_id = None;
        self.selected_hex = None;
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
                let mut grid = HexGrid {
                    width: state.grid_width,
                    height: state.grid_height,
                    hex_size: state.hex_size,
                    cells: HashMap::new(),
                };
                for &(q, r) in state.hex_cells.keys() {
                    grid.cells.insert(HexCoord::new(q, r), HexCell::new());
                }
                let mut dish = PetriDish::from_grid(dish_id, grid, state.config.clone(), DEFAULT_DISH_RADIUS_CELLS);

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
                            genetic_distance_to_parent: node_state.genetic_distance_to_parent,
                        },
                    );
                }
                dish.cladogram.current_time = state.timestep;

                // Reconstruct hex cells
                for ((q, r), cell_state) in state.hex_cells {
                    let coord = HexCoord::new(q, r);
                    if let Some(cell) = dish.hex_grid.get_cell_mut(coord) {
                        cell.resource_level = cell_state.resource_level;
                        cell.individual_indices = cell_state.individual_indices;
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
                    if ui.button("Config Editor").clicked() {
                        self.show_config_editor = true;
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

        if self.show_config_editor {
            let mut editor_open = self.show_config_editor;
            let mut start_new = false;
            let mut add_new_dish = false;
            egui::Window::new("Simulation Config Editor")
                .open(&mut editor_open)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.label("Core");
                    ui.add(egui::Slider::new(&mut self.config.initial_population, 1..=1000).text("Initial Population"));
                    ui.add(egui::Slider::new(&mut self.config.seed, 0..=u32::MAX).text("Seed"));
                    ui.add(egui::Slider::new(&mut self.config_radius_cells, 10..=220).text("Dish Radius (cells)"));
                    ui.add(egui::Slider::new(&mut self.config_hex_size, 4.0..=24.0).text("Hex Size"));

                    ui.separator();
                    ui.label("Neural Network");
                    ui.add(egui::Slider::new(&mut self.config.nn_nodes, 1..=128).text("Hidden Nodes"));
                    ui.add(egui::Slider::new(&mut self.config.nn_layers, 1..=2).text("Hidden Layers"));
                    ui.add(egui::Slider::new(&mut self.config.nn_nodes2, 0..=128).text("Second Hidden Nodes"));
                    ui.label("NN Inputs are auto-enforced to 8 in interactive mode (6 edges + occupancy + internal resources).");

                    ui.separator();
                    ui.label("Dynamics");
                    ui.add(egui::Slider::new(&mut self.config.reproduction_threshold, 1.0..=100.0).text("Reproduction Threshold"));
                    ui.add(egui::Slider::new(&mut self.config.reproduction_cost, 0.0..=50.0).text("Reproduction Cost"));
                    ui.add(egui::Slider::new(&mut self.config.mutation_rate, 0.0..=1.0).text("Mutation Rate"));
                    ui.add(egui::Slider::new(&mut self.config.mutation_sigma, 0.0..=2.0).text("Mutation Sigma"));
                    ui.add(egui::Slider::new(&mut self.config.movement_cost_factor, 0.0..=10.0).text("Movement Cost"));

                    ui.separator();
                    ui.label("Environment");
                    ui.add(egui::Slider::new(&mut self.config.resource_scale, 0.0..=5.0).text("Resource Scale"));
                    ui.add(egui::Slider::new(&mut self.config.env_time_speed, 0.0..=0.2).text("Env Time Speed"));
                    ui.add(egui::Slider::new(&mut self.config.env_time_amplitude, 0.0..=3.0).text("Env Time Amplitude"));
                    ui.add(egui::Slider::new(&mut self.config.env_perlin_spatial_scale, 0.001..=1.0).text("Spatial Scale"));
                    ui.add(egui::Slider::new(&mut self.config.env_perlin_time_scale, 0.001..=1.0).text("Time Scale"));

                    ui.separator();
                    if ui.button("Start New Simulation (Selected Dish)").clicked() {
                        start_new = true;
                    }
                    if ui.button("Add New Dish From Config").clicked() {
                        add_new_dish = true;
                    }
                });
            self.show_config_editor = editor_open;
            if start_new {
                self.start_new_simulation_from_config();
            }
            if add_new_dish {
                self.add_dish_from_config();
            }
        }

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
            ui.add(egui::Slider::new(&mut self.move_threshold, -2.0..=2.0)
                .text("Move Threshold"));
            ui.add(egui::Slider::new(&mut self.zoom, 0.25..=3.0)
                .text("Zoom"));

            ui.separator();

            if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                ui.heading("Statistics");
                ui.label(format!("Individuals: {}", dish.simulation.individuals.len()));
                ui.label(format!("Timestep: {}", dish.timestep));
                ui.label(format!("Move Threshold: {:.2}", self.move_threshold));
                ui.label(format!("Avg Move Signal (last step): {:.3}", dish.last_avg_move_signal));

                // Display selected hex information
                if let Some(hex_coord) = self.selected_hex {
                    if let Some(cell) = dish.hex_grid.get_cell(hex_coord) {
                        ui.separator();
                        ui.heading("Selected Hex");
                        ui.label(format!("Coordinates: ({}, {})", hex_coord.q, hex_coord.r));
                        
                        let pop_density = cell.individual_indices.len();
                        ui.label(format!("Population: {}", pop_density));
                        
                        // Calculate approximate resources per individual per timestep
                        let resource_level = cell.resource_level;
                        let resources_per_ind = if pop_density > 0 {
                            resource_level / pop_density as f64
                        } else {
                            resource_level
                        };
                        ui.label(format!("Resource Level: {:.2}", resource_level));
                        ui.label(format!("Resources/Individual: {:.2}", resources_per_ind));
                        
                        // Display unique lineage colors present in this hex
                        if !cell.individual_indices.is_empty() {
                            ui.separator();
                            ui.label("Lineages in this hex:");
                            
                            // Collect unique lineage colors
                            let mut lineage_colors: std::collections::HashMap<[u8; 3], Vec<usize>> = 
                                std::collections::HashMap::new();
                            
                            for &ind_idx in &cell.individual_indices {
                                if let Some(ind) = dish.simulation.individuals.get(ind_idx) {
                                    lineage_colors.entry(ind.lineage_color)
                                        .or_insert_with(Vec::new)
                                        .push(ind_idx);
                                }
                            }
                            
                            // Display each lineage with a button to select a random individual
                            for (color, individual_indices) in lineage_colors.iter() {
                                ui.horizontal(|ui| {
                                    // Color swatch
                                    let color_rect = egui::Rect::from_min_size(
                                        ui.cursor().min,
                                        egui::vec2(16.0, 16.0),
                                    );
                                    let (color_resp, color_painter) = ui.allocate_painter(
                                        color_rect.size(), 
                                        egui::Sense::hover()
                                    );
                                    color_painter.rect_filled(
                                        color_resp.rect,
                                        2.0,
                                        egui::Color32::from_rgb(color[0], color[1], color[2]),
                                    );
                                    
                                    ui.label(format!("Count: {}", individual_indices.len()));
                                    
                                    // Button to select random individual from this lineage
                                    if ui.small_button("Select Random").clicked() {
                                        use rand::seq::SliceRandom;
                                        if let Some(&random_idx) = individual_indices.choose(&mut rand::thread_rng()) {
                                            if let Some(ind) = dish.simulation.individuals.get(random_idx) {
                                                self.selected_individual_id = Some(ind.id);
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    }
                }

                if let Some(idx) = self.selected_individual_index(dish) {
                    if let Some(ind) = dish.simulation.individuals.get(idx) {
                        ui.separator();
                        ui.heading("Selected Individual");
                        ui.label(format!("ID: {}", ind.id));
                        ui.label(format!("Resources: {:.2}", ind.resources));
                        ui.label(format!("Age: {}", ind.age));
                        ui.label(format!("Position: {}", ind.position));
                        let lineage_hex = format!(
                            "#{:02X}{:02X}{:02X}",
                            ind.lineage_color[0], ind.lineage_color[1], ind.lineage_color[2]
                        );
                        ui.label(format!("Lineage Color: {}", lineage_hex));
                        let color_rect = egui::Rect::from_min_size(
                            ui.cursor().min,
                            egui::vec2(24.0, 12.0),
                        );
                        let (color_resp, color_painter) = ui.allocate_painter(color_rect.size(), egui::Sense::hover());
                        color_painter.rect_filled(
                            color_resp.rect,
                            2.0,
                            egui::Color32::from_rgb(
                                ind.lineage_color[0],
                                ind.lineage_color[1],
                                ind.lineage_color[2],
                            ),
                        );
                        ui.add_space(4.0);
                    }
                }
            }
        });

        // Right panel - Neural Network / Cladogram
        if self.show_nn_diagram || self.show_cladogram {
            egui::SidePanel::right("viz_panel")
                .min_width(400.0)
                .resizable(true)
                .show(ctx, |ui| {
                if self.show_nn_diagram {
                    ui.heading("Neural Network");
                    
                    if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                        if let Some(idx) = self.selected_individual_index(dish) {
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
                        } else {
                            ui.label("Select an individual to see its neural network");
                        }
                    } else {
                        ui.label("Select an individual to see its neural network");
                    }
                }

                if self.show_cladogram {
                    ui.separator();
                    ui.heading("Cladogram");
                    ui.label("Phylogenetic tree visualization");
                    ui.add(egui::Slider::new(&mut self.phylogeny_zoom, 0.2..=4.0).text("Phylogeny Zoom"));
                    ui.add(egui::Slider::new(&mut self.phylogeny_height, 140.0..=1200.0).text("Phylogeny Height"));
                    ui.add(egui::Slider::new(&mut self.phylogeny_max_render_nodes, 200..=10000).text("Max Render Nodes"));
                    
                    if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                        ui.label(format!("Total nodes: {}", dish.cladogram.nodes.len()));
                        
                        if let Some(idx) = self.selected_individual_index(dish) {
                            if let Some(ind) = dish.simulation.individuals.get(idx) {
                                let lineage = dish.cladogram.get_lineage_path(ind.id);
                                ui.label(format!("Lineage depth: {}", lineage.len()));
                                ui.label(format!("Ancestor IDs: {:?}", &lineage[..lineage.len().min(10)]));
                            }
                        }

                        let area_h = self.phylogeny_height;

                        let mut ids: Vec<i32> = dish.cladogram.nodes.keys().copied().collect();
                        ids.sort_by_key(|id| {
                            let node = &dish.cladogram.nodes[id];
                            (node.birth_time, *id)
                        });

                        if !ids.is_empty() {
                            // Performance guardrail: for large trees, render a focused subset.
                            // Priority order is extant leaves + their ancestor paths, then recent nodes.
                            let original_total = ids.len();
                            if ids.len() > self.phylogeny_max_render_nodes {
                                let mut extant: Vec<i32> = ids
                                    .iter()
                                    .copied()
                                    .filter(|id| dish.cladogram.nodes[id].death_time.is_none())
                                    .collect();
                                extant.sort_unstable_by_key(|id| {
                                    let n = &dish.cladogram.nodes[id];
                                    (-n.birth_time, -*id)
                                });

                                let mut keep: HashSet<i32> = HashSet::new();
                                for id in &extant {
                                    let mut cur = *id;
                                    loop {
                                        keep.insert(cur);
                                        if keep.len() >= self.phylogeny_max_render_nodes {
                                            break;
                                        }
                                        if let Some(parent_id) = dish.cladogram.nodes[&cur].parent_id {
                                            cur = parent_id;
                                        } else {
                                            break;
                                        }
                                    }
                                    if keep.len() >= self.phylogeny_max_render_nodes {
                                        break;
                                    }
                                }

                                if keep.len() < self.phylogeny_max_render_nodes {
                                    let mut recent = ids.clone();
                                    recent.sort_unstable_by_key(|id| {
                                        let n = &dish.cladogram.nodes[id];
                                        (-n.birth_time, -*id)
                                    });
                                    for id in recent {
                                        keep.insert(id);
                                        if keep.len() >= self.phylogeny_max_render_nodes {
                                            break;
                                        }
                                    }
                                }

                                ids.retain(|id| keep.contains(id));
                                ids.sort_by_key(|id| {
                                    let node = &dish.cladogram.nodes[id];
                                    (node.birth_time, *id)
                                });
                            }

                            ui.label(format!("Showing {} / {} nodes", ids.len(), original_total));

                            // Zoom applies to both axes for quick magnification; the surrounding
                            // ScrollArea lets users navigate large pedigrees without clipping.
                            let canvas_w = ((ui.available_width() - 8.0).max(200.0) * self.phylogeny_zoom).max(220.0);
                            let canvas_h = (area_h.max(140.0) * self.phylogeny_zoom.max(0.2)).max(160.0);

                            egui::ScrollArea::both().show(ui, |ui| {
                                let (response, painter) = ui.allocate_painter(
                                    egui::vec2(canvas_w, canvas_h),
                                    egui::Sense::hover(),
                                );

                                let left = response.rect.left() + 12.0;
                                let right = response.rect.right() - 12.0;
                                let top = response.rect.top() + 10.0;
                                let bottom = response.rect.bottom() - 20.0;

                                let mut roots: Vec<i32> = ids
                                .iter()
                                .copied()
                                .filter(|id| dish.cladogram.nodes[id].parent_id.is_none())
                                .collect();
                                roots.sort_unstable();

                                let draw_set: HashSet<i32> = ids.iter().copied().collect();

                                let mut depth: HashMap<i32, usize> = HashMap::new();
                                let mut stack: Vec<(i32, usize)> = roots.iter().map(|id| (*id, 0usize)).collect();
                                while let Some((id, d)) = stack.pop() {
                                let insert = match depth.get(&id) {
                                    Some(prev) => d < *prev,
                                    None => true,
                                };
                                if !insert {
                                    continue;
                                }
                                depth.insert(id, d);
                                let mut children = dish.cladogram.nodes[&id].children.clone();
                                children.sort_unstable_by(|a, b| b.cmp(a));
                                for child_id in children {
                                    if draw_set.contains(&child_id) {
                                        stack.push((child_id, d + 1));
                                    }
                                }
                            }
                            for id in &ids {
                                depth.entry(*id).or_insert(0);
                            }

                                // Assign y by post-order traversal so leaves spread out and
                                // internal nodes are centered above their descendant spans.
                                fn assign_y(
                                id: i32,
                                nodes: &HashMap<i32, evo_movement_rust::cladogram::LineageNode>,
                                draw_set: &HashSet<i32>,
                                y_map: &mut HashMap<i32, f32>,
                                next_y: &mut f32,
                            ) -> f32 {
                                if let Some(v) = y_map.get(&id) {
                                    return *v;
                                }
                                let node = &nodes[&id];
                                let mut children = node.children.clone();
                                children.sort_unstable();
                                let mut child_ys = Vec::new();
                                for child_id in children {
                                    if draw_set.contains(&child_id) {
                                        child_ys.push(assign_y(child_id, nodes, draw_set, y_map, next_y));
                                    }
                                }
                                let y_val = if child_ys.is_empty() {
                                    let y = *next_y;
                                    *next_y += 1.0;
                                    y
                                } else {
                                    child_ys.iter().sum::<f32>() / child_ys.len() as f32
                                };
                                y_map.insert(id, y_val);
                                y_val
                            }

                                let mut y_map: HashMap<i32, f32> = HashMap::new();
                                let mut next_y = 0.0f32;
                                for root in roots {
                                    let _ = assign_y(root, &dish.cladogram.nodes, &draw_set, &mut y_map, &mut next_y);
                                }
                                for id in &ids {
                                    if !y_map.contains_key(id) {
                                        let _ = assign_y(*id, &dish.cladogram.nodes, &draw_set, &mut y_map, &mut next_y);
                                    }
                                }

                                let mut effective_depth_max = depth.values().copied().max().unwrap_or(1);
                                let extant_ids: HashSet<i32> = dish
                                .cladogram
                                .nodes
                                .iter()
                                .filter_map(|(id, n)| {
                                    if n.death_time.is_none() && draw_set.contains(id) {
                                        Some(*id)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if !extant_ids.is_empty() {
                                effective_depth_max += 1;
                            }
                                let depth_span = effective_depth_max.max(1) as f32;
                                let y_span = (next_y.max(1.0) - 1.0).max(1.0);

                                let mut positions: HashMap<i32, egui::Pos2> = HashMap::new();
                                for id in &ids {
                                let d = *depth.get(id).unwrap_or(&0);
                                let draw_depth = if extant_ids.contains(id) {
                                    effective_depth_max
                                } else {
                                    d
                                };
                                let x = left + (draw_depth as f32 / depth_span) * (right - left);
                                let y_norm = y_map.get(id).copied().unwrap_or(0.0) / y_span;
                                let y = top + y_norm * (bottom - top);
                                positions.insert(*id, egui::pos2(x, y));
                            }

                                for id in &ids {
                                let node = &dish.cladogram.nodes[id];
                                if let Some(parent_id) = node.parent_id {
                                    if let (Some(parent_pos), Some(child_pos)) =
                                        (positions.get(&parent_id), positions.get(id))
                                    {
                                        let branch_color = egui::Color32::from_rgb(
                                            node.color[0],
                                            node.color[1],
                                            node.color[2],
                                        );
                                        let branch_width = (1.2 + node.genetic_distance_to_parent as f32 * 2.0)
                                            .clamp(1.0, 4.0);
                                        // Pedigree elbow segments keep branches readable in dense trees.
                                        let elbow = egui::pos2(child_pos.x, parent_pos.y);
                                        painter.line_segment(
                                            [*parent_pos, elbow],
                                            egui::Stroke::new(branch_width, branch_color),
                                        );
                                        painter.line_segment(
                                            [elbow, *child_pos],
                                            egui::Stroke::new(branch_width, branch_color),
                                        );
                                    }
                                }
                            }

                                for id in &ids {
                                let node = &dish.cladogram.nodes[id];
                                if let Some(pos) = positions.get(id) {
                                    let is_extant = node.death_time.is_none();
                                    let mut radius = if is_extant { 4.0 } else { 3.0 };
                                    let mut stroke = egui::Stroke::new(1.0, egui::Color32::BLACK);
                                    if self.selected_individual_id == Some(*id) {
                                        radius = 6.0;
                                        stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
                                    }
                                    painter.circle_filled(
                                        *pos,
                                        radius,
                                        egui::Color32::from_rgb(
                                            node.color[0],
                                            node.color[1],
                                            node.color[2],
                                        ),
                                    );
                                    painter.circle_stroke(*pos, radius, stroke);

                                    if is_extant {
                                        painter.text(
                                            egui::pos2(pos.x + 5.0, pos.y),
                                            egui::Align2::LEFT_CENTER,
                                            format!("{}", id),
                                            egui::FontId::proportional(9.0),
                                            egui::Color32::LIGHT_GRAY,
                                        );
                                    }
                                }
                            }

                                painter.text(
                                egui::pos2(left, bottom + 2.0),
                                egui::Align2::LEFT_TOP,
                                "Pedigree view (extant organisms shown as terminal tips)",
                                egui::FontId::proportional(10.0),
                                egui::Color32::GRAY,
                            );
                            });
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

            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for coord in dish.hex_grid.cells.keys() {
                let (x, y) = coord.to_pixel(dish.hex_grid.hex_size);
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

            let width_px = (max_x - min_x) + 2.0 * dish.hex_grid.hex_size;
            let height_px = (max_y - min_y) + 2.0 * dish.hex_grid.hex_size;
            let available_size = ui.available_size();
            let scale = (available_size.x / width_px).min(available_size.y / height_px) * self.zoom.max(0.01);
            let hex_size = dish.hex_grid.hex_size;
            let padding = hex_size;
            let cells = &dish.hex_grid.cells;
            let individuals = &dish.simulation.individuals;
            let selected_ind_id = self.selected_individual_id;

            let mut clicked_hex: Option<HexCoord> = None;
            let mut dragged_hex: Option<HexCoord> = None;
            let mut drag_active = false;
            let mut drag_stopped = false;

            egui::ScrollArea::both().show(ui, |ui| {
                let (response, painter) = ui.allocate_painter(
                    egui::vec2(width_px * scale, height_px * scale),
                    egui::Sense::click_and_drag(),
                );

                // Draw hexagonal grid
                for (coord, cell) in cells {
                    let (x, y) = coord.to_pixel(hex_size);
                    let local_x = (x - min_x + padding) * scale;
                    let local_y = (y - min_y + padding) * scale;
                    let screen_pos = response.rect.min + egui::vec2(local_x, local_y);

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
                    for &ind_idx in &cell.individual_indices {
                        if let Some(ind) = individuals.get(ind_idx) {
                            let color = egui::Color32::from_rgb(
                                ind.lineage_color[0],
                                ind.lineage_color[1],
                                ind.lineage_color[2],
                            );
                            painter.circle_filled(screen_pos, 3.0 * scale, color);

                            if Some(ind.id) == selected_ind_id {
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
                            (rel_pos.x / scale) - padding + min_x,
                            (rel_pos.y / scale) - padding + min_y,
                            hex_size,
                        );
                        clicked_hex = Some(hex_coord);
                    }
                }
                if response.secondary_clicked() {
                    self.selected_individual_id = None;
                }

                if response.dragged() {
                    drag_active = true;
                    if let Some(pos) = response.interact_pointer_pos() {
                        let rel_pos = pos - response.rect.min;
                        let hex_coord = HexCoord::from_pixel(
                            (rel_pos.x / scale) - padding + min_x,
                            (rel_pos.y / scale) - padding + min_y,
                            hex_size,
                        );
                        dragged_hex = Some(hex_coord);
                    }
                }
                if response.drag_stopped() {
                    drag_stopped = true;
                }
            });

            // Handle selection outside of the closure
            if let Some(coord) = clicked_hex {
                if let Some(dish) = self.petri_dishes.get(self.selected_dish) {
                    // Always set the selected hex
                    self.selected_hex = Some(coord);
                    
                    if let Some(cell) = dish.hex_grid.get_cell(coord) {
                        if !cell.individual_indices.is_empty() {
                            if let Some(ind) = dish.simulation.individuals.get(cell.individual_indices[0]) {
                                self.selected_individual_id = Some(ind.id);
                            }
                        } else {
                            // Clicked on empty hex, clear individual selection
                            self.selected_individual_id = None;
                        }
                    }
                }
            }

            if drag_active {
                if let Some(selected_id) = self.selected_individual_id {
                    if !self.is_dragging_individual {
                        self.paused_before_drag = self.paused;
                    }
                    self.is_dragging_individual = true;
                    self.paused = true;

                    if let Some(coord) = dragged_hex {
                        if let Some(dish) = self.petri_dishes.get_mut(self.selected_dish) {
                            if let Some(target_idx) = dish.coord_to_index.get(&coord).copied() {
                                if let Some(ind_idx) = dish
                                    .simulation
                                    .individuals
                                    .iter()
                                    .position(|ind| ind.id == selected_id)
                                {
                                    dish.simulation.individuals[ind_idx].position = target_idx;
                                    dish.rebuild_hex_occupancy();
                                }
                            }
                        }
                    }
                }
            }

            if drag_stopped && self.is_dragging_individual {
                self.is_dragging_individual = false;
                self.paused = self.paused_before_drag;
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
        let angle = std::f32::consts::PI / 3.0 * i as f32 + std::f32::consts::PI / 6.0;
        let x = center.x + size * angle.cos();
        let y = center.y + size * angle.sin();
        points.push(egui::pos2(x, y));
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_direction_test_dish(direction_bias: f64) -> (PetriDish, HexCoord) {
        let mut cfg = Config::default();
        cfg.nn_nodes = 3;
        cfg.nn_inputs = 1;
        cfg.nn_layers = 1;
        cfg.nn_nodes2 = 0;
        cfg.movement_cost_factor = 0.0;
        cfg.maintenance_cost_per_node = 0.0;
        cfg.death_risk = 0.0;
        cfg.reproduction_threshold = 1e9;

        let radius = 1;
        let mut dish = PetriDish::new_hex_dish(0, radius, 10.0, cfg);
        let center = HexCoord::new(radius, radius);
        let center_idx = dish
            .coord_to_index
            .get(&center)
            .copied()
            .expect("center coordinate should exist");

        let mut nn = NeuralNetwork::new(3, 1, 1, 0);
        for w in nn.weights.iter_mut() {
            *w = 0.0;
        }
        for b in nn.biases.iter_mut() {
            *b = 0.0;
        }
        // Keep move_signal positive regardless of direction node contribution.
        nn.biases[0] = 2.0;
        nn.biases[1] = direction_bias;
        nn.biases[2] = 2.0;

        let mut ind = Individual::new(nn, center_idx, 1);
        ind.resources = 10.0;
        dish.simulation.individuals.push(ind);
        dish.rebuild_hex_occupancy();
        (dish, center)
    }

    #[test]
    fn directional_movement_maps_to_expected_neighbor_sectors() {
        let cases = [
            (-5.0, 0usize), // ~0°
            (-0.35, 2usize), // ~120°
            (3.0, 5usize),  // ~300°
        ];

        for (direction_bias, expected_sector) in cases {
            let (mut dish, center) = make_direction_test_dish(direction_bias);
            dish.step(-10.0);

            let pos_idx = dish.simulation.individuals[0].position;
            let moved_coord = dish.active_coords[pos_idx];
            let expected_coord = center.neighbors()[expected_sector];

            assert_eq!(
                moved_coord, expected_coord,
                "direction bias {} should map to sector {}",
                direction_bias, expected_sector
            );
        }
    }

    #[test]
    fn movement_is_suppressed_when_below_threshold() {
        let (mut dish, center) = make_direction_test_dish(-5.0);
        let center_idx = dish
            .coord_to_index
            .get(&center)
            .copied()
            .expect("center coordinate should exist");

        // direction_bias=-5 yields a positive move signal via node 0+2,
        // but set threshold above that signal to suppress movement.
        dish.step(2.0);

        let pos_idx = dish.simulation.individuals[0].position;
        assert_eq!(pos_idx, center_idx, "individual should remain in place when move_signal <= threshold");
    }
}
