/// Save/load state for the interactive simulation
use crate::{
    cladogram::Cladogram,
    config::Config,
    hex_grid::{HexCoord, HexGrid, HexCell},
    individual::Individual,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;

#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationState {
    pub config: Config,
    pub individuals: Vec<IndividualState>,
    pub grid_width: usize,
    pub grid_height: usize,
    pub hex_size: f32,
    pub timestep: i32,
    pub next_id: i32,
    pub cladogram_nodes: Vec<CladogramNodeState>,
    pub hex_cells: HashMap<(i32, i32), HexCellState>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct IndividualState {
    pub id: i32,
    pub position: usize,
    pub resources: f64,
    pub age: i32,
    pub lineage_color: [u8; 3],
    pub parent_id: Option<i32>,
    pub nn_weights: Vec<f64>,
    pub nn_weights_multi: Vec<f64>,
    pub nn_biases: Vec<f64>,
    pub nn_weights2: Vec<f64>,
    pub nn_biases2: Vec<f64>,
    pub nn_nodes: usize,
    pub nn_inputs: usize,
    pub nn_nodes2: usize,
    pub nn_layers: i32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CladogramNodeState {
    pub id: i32,
    pub parent_id: Option<i32>,
    pub children: Vec<i32>,
    pub birth_time: i32,
    pub death_time: Option<i32>,
    pub color: [u8; 3],
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HexCellState {
    pub resource_level: f64,
    pub individual_indices: Vec<usize>,
}

impl SimulationState {
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    pub fn load_from_file(path: &str) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        serde_json::from_str(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn from_individuals_and_grid(
        individuals: &[Individual],
        hex_grid: &HexGrid,
        cladogram: &Cladogram,
        config: &Config,
        timestep: i32,
        next_id: i32,
    ) -> Self {
        let ind_states: Vec<IndividualState> = individuals
            .iter()
            .map(|ind| IndividualState {
                id: ind.id,
                position: ind.position,
                resources: ind.resources,
                age: ind.age,
                lineage_color: ind.lineage_color,
                parent_id: ind.parent_id,
                nn_weights: ind.nn.weights.clone(),
                nn_weights_multi: ind.nn.weights_multi.clone(),
                nn_biases: ind.nn.biases.clone(),
                nn_weights2: ind.nn.weights2.clone(),
                nn_biases2: ind.nn.biases2.clone(),
                nn_nodes: ind.nn.nodes,
                nn_inputs: ind.nn.inputs,
                nn_nodes2: ind.nn.nodes2,
                nn_layers: ind.nn.layers,
            })
            .collect();

        let clado_nodes: Vec<CladogramNodeState> = cladogram
            .nodes
            .values()
            .map(|node| CladogramNodeState {
                id: node.id,
                parent_id: node.parent_id,
                children: node.children.clone(),
                birth_time: node.birth_time,
                death_time: node.death_time,
                color: node.color,
            })
            .collect();

        let mut hex_cells = HashMap::new();
        for (coord, cell) in &hex_grid.cells {
            hex_cells.insert(
                (coord.q, coord.r),
                HexCellState {
                    resource_level: cell.resource_level,
                    individual_indices: cell.individuals.clone(),
                },
            );
        }

        SimulationState {
            config: config.clone(),
            individuals: ind_states,
            grid_width: hex_grid.width,
            grid_height: hex_grid.height,
            hex_size: hex_grid.hex_size,
            timestep,
            next_id,
            cladogram_nodes: clado_nodes,
            hex_cells,
        }
    }
}
