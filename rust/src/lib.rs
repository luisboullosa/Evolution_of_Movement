pub mod brain;
pub mod brain_diagram;
pub mod brain_playground;
pub mod config;
pub mod neural_network;

#[cfg(not(feature = "playground"))]
pub mod architecture_lineage;
#[cfg(not(feature = "playground"))]
pub mod build_info;
#[cfg(not(feature = "playground"))]
pub mod cladogram;
#[cfg(not(feature = "playground"))]
pub mod hex_simulation;
pub mod ecology_presets;
#[cfg(not(feature = "playground"))]
pub mod environment;
#[cfg(all(not(feature = "playground"), feature = "cuda"))]
pub mod gpu_resource;
#[cfg(not(feature = "playground"))]
pub mod headless;
#[cfg(not(feature = "playground"))]
pub mod hex_dish;
#[cfg(not(feature = "playground"))]
pub mod hex_grid;
#[cfg(all(not(feature = "playground"), feature = "gui"))]
pub mod lab_dish;
#[cfg(not(feature = "playground"))]
pub mod individual;
#[cfg(all(not(feature = "playground"), feature = "gui"))]
pub mod nn_viz;
#[cfg(not(feature = "playground"))]
pub mod perlin;
#[cfg(not(feature = "playground"))]
pub mod population_dynamics;
#[cfg(not(feature = "playground"))]
pub mod run_record;
#[cfg(not(feature = "playground"))]
pub mod save_state;
#[cfg(not(feature = "playground"))]
pub mod simulation;
#[cfg(not(feature = "playground"))]
pub mod snapshot_registry;
#[cfg(not(feature = "playground"))]
pub mod snapshot;
#[cfg(not(feature = "playground"))]
pub mod step_engine;
#[cfg(all(not(feature = "playground"), feature = "cuda"))]
pub mod tensor;
#[cfg(not(feature = "playground"))]
pub mod webviewer_writer;
