//! Integration tests for evolution-hunt HTML skeleton export scope.

use evo_movement_rust::config::Config;
use evo_movement_rust::fitness_landscape::{
    build_cells_to_sweep, build_expected_heatmaps, build_heatmaps_for_live, cells_to_grid_specs,
    escape_json_for_script_embed, evolution_hunt_grand_total, expand_output_mode_cells,
    inject_live_meta, pending_cell_count, LiveMeta, SweepProfile,
};
use std::path::Path;

fn evolution_hunt_configs() -> (Config, Config, Config, Config, Config) {
    let root = Path::new("configs/ecology_plateau80/evolution_hunt");
    let load = |name: &str| Config::from_file(&root.join(format!("{name}.json"))).unwrap();
    (
        load("perceptron"),
        load("mlp"),
        load("spatial_graph"),
        load("spatial_cardinal"),
        load("goo"),
    )
}

#[test]
fn html_embed_has_full_pending_skeleton_for_evolution_hunt() {
    let (perceptron, mlp, spatial, spatial_cardinal, goo) = evolution_hunt_configs();
    let mut cells = build_cells_to_sweep(
        &perceptron,
        &mlp,
        &[spatial, spatial_cardinal],
        &goo,
        SweepProfile::EvolutionHunt,
    );
    cells = expand_output_mode_cells(cells);

    let founders = vec!["random_uniform".into(), "random_small".into()];
    let specs = cells_to_grid_specs(&cells);
    let heatmaps = build_expected_heatmaps(&specs, &founders);

    assert_eq!(cells.len(), 108);
    assert_eq!(heatmaps.len(), 24);
    assert_eq!(pending_cell_count(&heatmaps), 216);

    let payload = serde_json::json!({
        "version": "test",
        "build_id": "test",
        "steps": 200000,
        "replicates": 3,
        "founder_modes": founders,
        "runs": [],
        "cells": [],
        "heatmaps": heatmaps,
        "cell_diagrams": [],
    });
    let embedded = inject_live_meta(
        payload,
        Some(LiveMeta {
            status: "running",
            run_count: 0,
            total_runs: evolution_hunt_grand_total(cells.len(), 2, 3, 4),
        }),
        0,
    );
    let json = serde_json::to_string(&embedded).unwrap();
    let safe = escape_json_for_script_embed(&json);
    assert!(!safe.contains("</script>"));
    assert_eq!(embedded["status"], "running");
    assert_eq!(embedded["total_runs"], 2592);
    assert_eq!(embedded["heatmaps"].as_array().unwrap().len(), 24);
}

#[test]
fn overlay_partial_run_keeps_remaining_cells_pending() {
    let (perceptron, mlp, spatial, spatial_cardinal, goo) = evolution_hunt_configs();
    let mut cells = build_cells_to_sweep(
        &perceptron,
        &mlp,
        &[spatial, spatial_cardinal],
        &goo,
        SweepProfile::EvolutionHunt,
    );
    cells = expand_output_mode_cells(cells);
    let founders = vec!["random_uniform".into(), "random_small".into()];
    let specs = cells_to_grid_specs(&cells);

    let partial_cells = vec![evo_movement_rust::fitness_landscape::CellAggregate {
        architecture: "Perceptron".into(),
        x_label: "nodes=1".into(),
        y_label: "baseline".into(),
        founder_mode: "random_uniform".into(),
        replicates: 1,
        fitness_mean: 0.55,
        fitness_std: 0.0,
        pop_ratio_mean: 0.9,
        complexity_mean: 2.0,
        data_na: false,
        trajectory_mean: vec![],
    }];
    let heatmaps = build_heatmaps_for_live(&partial_cells, Some(&specs), &founders);
    assert_eq!(pending_cell_count(&heatmaps), 215);
    let perceptron_uniform = heatmaps
        .iter()
        .find(|h| h.architecture == "Perceptron" && h.founder_mode == "random_uniform")
        .unwrap();
    assert_eq!(perceptron_uniform.fitness_matrix[0][0], Some(0.55));
}
