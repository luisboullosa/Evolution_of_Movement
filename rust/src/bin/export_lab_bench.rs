//! Export stream of Lab Bench frames (hex dish + brain diagram) to JSONL for rendering.
use evo_movement_rust::{
    brain_diagram::diagram_from_brain,
    config::{ArchitecturePolicy, Config},
    lab_dish::{resource_background_color, LabDish},
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Serialize)]
struct CellRecord {
    q: i32,
    r: i32,
    x: f32,
    y: f32,
    resource: f64,
    rgb: [u8; 3],
    count: usize,
}

#[derive(Serialize)]
struct IndividualRecord {
    id: i32,
    pos: usize,
    q: i32,
    r: i32,
    x: f32,
    y: f32,
    color: [u8; 3],
    resources: f64,
    is_selected: bool,
}

#[derive(Serialize)]
struct FrameRecord {
    t: i32,
    population: usize,
    selected_id: Option<i32>,
    selected_resources: f64,
    avg_move_signal: f64,
    cells: Vec<CellRecord>,
    individuals: Vec<IndividualRecord>,
    brain_diagram: Option<evo_movement_rust::brain_diagram::BrainDiagram>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let output_path = args
        .next()
        .unwrap_or_else(|| "temp_outputs/lab_bench_stream.jsonl".to_string());
    let steps: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    let radius = 10;
    let hex_size = 14.0f32;

    let mut cfg = Config::default();
    cfg.initial_population = 16;
    cfg.max_population = 250;
    cfg.reproduction_threshold = 18.0;
    cfg.reproduction_cost = 8.0;
    cfg.seed = 42;
    cfg.env_time_speed = 0.025;
    cfg.env_time_amplitude = 1.0;
    cfg.env_perlin_spatial_scale = 0.08;
    cfg.env_perlin_time_scale = 0.008;
    cfg.movement_cost_factor = 0.08;
    let mut dish = LabDish::new(radius, hex_size, cfg);
    dish.clear_population();
    dish.insert_individuals(18, ArchitecturePolicy::PerceptronOnly, radius, 12.0);

    const FOUNDER_PALETTE: [[u8; 3]; 10] = [
        [239, 68, 68],   // Coral red
        [59, 130, 246],  // Sky blue
        [234, 179, 8],   // Gold
        [168, 85, 247],  // Royal purple
        [34, 197, 94],   // Bright emerald
        [249, 115, 22],  // Amber orange
        [236, 72, 153],  // Rose pink
        [20, 184, 166],  // Teal
        [139, 92, 246],  // Indigo
        [244, 63, 94],   // Crimson
    ];
    for (i, ind) in dish.simulation.individuals.iter_mut().enumerate() {
        ind.lineage_color = FOUNDER_PALETTE[i % FOUNDER_PALETTE.len()];
    }

    if let Some(parent) = Path::new(&output_path).parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    let mut selected_id = dish.simulation.individuals.first().map(|ind| ind.id);

    println!(
        "Exporting {} Lab Bench frames to {} (radius={}, pop={})...",
        steps,
        output_path,
        radius,
        dish.simulation.individuals.len()
    );

    for t in 0..steps {
        // Step dish simulation
        let parent_colors: std::collections::HashMap<i32, [u8; 3]> = dish
            .simulation
            .individuals
            .iter()
            .map(|ind| (ind.id, ind.lineage_color))
            .collect();
        dish.step(0.0, 1.0, true);
        if t % 2 == 0 {
            dish.reproduce_eligible();
            for ind in &mut dish.simulation.individuals {
                if let Some(pid) = ind.parent_id {
                    if let Some(&pcol) = parent_colors.get(&pid) {
                        ind.lineage_color = pcol;
                    }
                }
            }
        }

        // Validate or update selected individual
        let selected_exists = selected_id.map_or(false, |id| {
            dish.simulation.individuals.iter().any(|ind| ind.id == id)
        });
        if !selected_exists {
            // Pick the individual with highest resources
            selected_id = dish
                .simulation
                .individuals
                .iter()
                .max_by(|a, b| a.resources.partial_cmp(&b.resources).unwrap())
                .map(|ind| ind.id);
        }

        // Collect cell data
        let mut cells = Vec::with_capacity(dish.hex_grid.cells.len());
        for (coord, cell) in &dish.hex_grid.cells {
            let (x, y) = coord.to_pixel(hex_size);
            cells.push(CellRecord {
                q: coord.q,
                r: coord.r,
                x,
                y,
                resource: cell.resource_level,
                rgb: resource_background_color(cell.resource_level),
                count: cell.individual_indices.len(),
            });
        }

        // Collect individuals
        let mut individuals = Vec::with_capacity(dish.simulation.individuals.len());
        let mut selected_resources = 0.0;
        let mut brain_diagram = None;

        for ind in &dish.simulation.individuals {
            let is_sel = Some(ind.id) == selected_id;
            let coord = dish.active_coords[ind.position % dish.active_coords.len()];
            let (x, y) = coord.to_pixel(hex_size);
            if is_sel {
                selected_resources = ind.resources;
                brain_diagram = Some(diagram_from_brain(&ind.brain, 360.0, 340.0));
            }
            individuals.push(IndividualRecord {
                id: ind.id,
                pos: ind.position,
                q: coord.q,
                r: coord.r,
                x,
                y,
                color: ind.lineage_color,
                resources: ind.resources,
                is_selected: is_sel,
            });
        }

        let frame = FrameRecord {
            t: dish.timestep,
            population: dish.simulation.individuals.len(),
            selected_id,
            selected_resources,
            avg_move_signal: dish.last_avg_move_signal,
            cells,
            individuals,
            brain_diagram,
        };

        serde_json::to_writer(&mut writer, &frame)?;
        writer.write_all(b"\n")?;
    }

    writer.flush()?;
    println!("Export complete: {} frames written.", steps);
    Ok(())
}
