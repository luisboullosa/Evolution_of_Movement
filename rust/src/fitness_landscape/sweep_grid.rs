use crate::config::{ArchitecturePolicy, Config, MovementOutputMode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SweepProfile {
    Full,
    EvolutionHunt,
}

#[derive(Clone)]
pub struct SweepCell {
    pub cfg: Config,
    pub x_label: String,
    pub y_label: String,
    pub goo_generate_nodes: usize,
    pub diagram_seed: u64,
}

pub fn architecture_label(cfg: &Config) -> String {
    let base = match cfg.architecture_policy {
        ArchitecturePolicy::EvolveSpatialGraph if cfg.uses_legacy_spatial_graph() => {
            "Spatial graph (SG)".to_string()
        }
        ArchitecturePolicy::EvolveSpatialGraph => {
            let layout = cfg.spatial_blob_layout.to_ascii_lowercase();
            if layout == "hex_edge" || layout.is_empty() {
                "Spatial macromolecule blob".to_string()
            } else {
                format!("Spatial macromolecule blob ({layout})")
            }
        }
        other => other.label().to_string(),
    };
    let mode = cfg.resolved_movement_output_mode();
    let is_default = match cfg.architecture_policy {
        ArchitecturePolicy::EvolveSpatialGraph => mode == MovementOutputMode::SeparateReadout,
        ArchitecturePolicy::EvolveGoo => mode == MovementOutputMode::HiddenSum,
        _ => true,
    };
    if is_default {
        base
    } else {
        format!("{base} ({})", mode.label())
    }
}

pub fn expand_output_mode_cells(cells: Vec<SweepCell>) -> Vec<SweepCell> {
    let mut out = Vec::new();
    for cell in cells {
        let modes: Option<&[MovementOutputMode]> = match cell.cfg.architecture_policy {
            ArchitecturePolicy::EvolveSpatialGraph => Some(MovementOutputMode::spatial_sweep_modes()),
            ArchitecturePolicy::EvolveGoo => Some(MovementOutputMode::goo_sweep_modes()),
            _ => None,
        };
        if let Some(modes) = modes {
            for &mode in modes {
                let mut cfg = cell.cfg.clone();
                cfg.movement_output_mode = Some(mode);
                out.push(SweepCell {
                    cfg,
                    x_label: cell.x_label.clone(),
                    y_label: cell.y_label.clone(),
                    goo_generate_nodes: cell.goo_generate_nodes,
                    diagram_seed: cell.diagram_seed,
                });
            }
        } else {
            out.push(cell);
        }
    }
    out
}

fn build_spatial_cells(base_spatial: &Config, profile: SweepProfile) -> Vec<SweepCell> {
    let (spatial_layers, spatial_caps): (&[usize], &[usize]) = match profile {
        SweepProfile::Full => (&[1, 2, 4, 8, 12, 16], &[16, 24, 32, 48]),
        SweepProfile::EvolutionHunt => (&[1, 2, 4, 8], &[32, 48]),
    };
    let mut cells = Vec::new();
    for (xi, layers) in spatial_layers.iter().enumerate() {
        for (yi, cap) in spatial_caps.iter().enumerate() {
            let mut cfg = base_spatial.clone();
            cfg.nn_nodes = *layers as i32;
            cfg.max_brain_nodes = *cap as i32;
            let x_label = if cfg.uses_legacy_spatial_graph() {
                format!("landmarks={layers}")
            } else {
                format!("layers={layers}")
            };
            cells.push(SweepCell {
                cfg,
                x_label,
                y_label: format!("max_brain_nodes={cap}"),
                goo_generate_nodes: 1,
                diagram_seed: (xi * 23 + yi) as u64,
            });
        }
    }
    cells
}

fn build_goo_cells(base_goo: &Config, profile: SweepProfile) -> Vec<SweepCell> {
    let (goo_gen_nodes, goo_caps): (&[usize], &[usize]) = match profile {
        SweepProfile::Full => (&[1, 2, 4, 8, 12], &[16, 24, 32, 48]),
        SweepProfile::EvolutionHunt => (&[1, 2, 4, 8], &[32, 48]),
    };
    let mut cells = Vec::new();
    for (xi, gn) in goo_gen_nodes.iter().enumerate() {
        for (yi, cap) in goo_caps.iter().enumerate() {
            let mut cfg = base_goo.clone();
            cfg.max_brain_nodes = *cap as i32;
            cells.push(SweepCell {
                cfg,
                x_label: format!("goo_nodes={gn}"),
                y_label: format!("max_brain_nodes={cap}"),
                goo_generate_nodes: *gn,
                diagram_seed: (xi * 29 + yi) as u64,
            });
        }
    }
    cells
}

pub fn build_cells_to_sweep(
    base_perceptron: &Config,
    base_mlp: &Config,
    spatial_bases: &[Config],
    base_goo: &Config,
    profile: SweepProfile,
) -> Vec<SweepCell> {
    let mut cells_to_sweep: Vec<SweepCell> = Vec::new();

    let perceptron_nodes = [1, 2, 4, 8, 12, 16, 24, 32];
    for (xi, n) in perceptron_nodes.iter().enumerate() {
        let mut cfg = base_perceptron.clone();
        cfg.nn_nodes = *n;
        cells_to_sweep.push(SweepCell {
            cfg,
            x_label: format!("nodes={n}"),
            y_label: "baseline".to_string(),
            goo_generate_nodes: 1,
            diagram_seed: xi as u64,
        });
    }

    let mlp_nodes = [1, 2, 4, 8, 12];
    let mlp_nodes2 = [0, 2, 4, 8];
    for (xi, n1) in mlp_nodes.iter().enumerate() {
        for (yi, n2) in mlp_nodes2.iter().enumerate() {
            let mut cfg = base_mlp.clone();
            cfg.nn_nodes = *n1;
            if *n2 > 0 {
                cfg.nn_layers = 2;
                cfg.nn_nodes2 = *n2;
            } else {
                cfg.nn_layers = 1;
                cfg.nn_nodes2 = 0;
            }
            cells_to_sweep.push(SweepCell {
                cfg,
                x_label: format!("L1={n1}"),
                y_label: format!("L2={n2}"),
                goo_generate_nodes: 1,
                diagram_seed: (xi * 17 + yi) as u64,
            });
        }
    }

    for base_spatial in spatial_bases {
        let mut spatial_cells = build_spatial_cells(base_spatial, profile);
        for c in &mut spatial_cells {
            c.diagram_seed ^= base_spatial.seed as u64;
        }
        cells_to_sweep.extend(spatial_cells);
    }

    cells_to_sweep.extend(build_goo_cells(base_goo, profile));
    cells_to_sweep
}

pub fn cells_to_grid_specs(cells: &[SweepCell]) -> Vec<crate::fitness_landscape::GridCellSpec> {
    cells
        .iter()
        .map(|c| crate::fitness_landscape::GridCellSpec {
            architecture: architecture_label(&c.cfg),
            x_label: c.x_label.clone(),
            y_label: c.y_label.clone(),
        })
        .collect()
}

pub fn evolution_hunt_grand_total(
    cells: usize,
    founder_modes: usize,
    replicates: usize,
    stages: usize,
) -> usize {
    cells * founder_modes * replicates * stages
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_named(base: &Path, name: &str) -> Config {
        let path = base.join(format!("{name}.json"));
        Config::from_file(&path).expect("config should load")
    }

    #[test]
    fn evolution_hunt_scope_is_108_cells_24_heatmaps() {
        let root = Path::new("configs/ecology_plateau80/evolution_hunt");
        let perceptron = load_named(root, "perceptron");
        let mlp = load_named(root, "mlp");
        let spatial = load_named(root, "spatial_graph");
        let spatial_cardinal = load_named(root, "spatial_cardinal");
        let goo = load_named(root, "goo");

        let mut cells = build_cells_to_sweep(
            &perceptron,
            &mlp,
            &[spatial, spatial_cardinal],
            &goo,
            SweepProfile::EvolutionHunt,
        );
        cells = expand_output_mode_cells(cells);
        assert_eq!(cells.len(), 108, "evolution_hunt profile cell count");

        let founders = vec!["random_uniform".into(), "random_small".into()];
        let specs = cells_to_grid_specs(&cells);
        let heatmaps = crate::fitness_landscape::build_expected_heatmaps(&specs, &founders);
        assert_eq!(heatmaps.len(), 24, "architecture tables × founder modes");

        let pending = crate::fitness_landscape::pending_cell_count(&heatmaps);
        assert_eq!(pending, 108 * founders.len());

        assert_eq!(
            evolution_hunt_grand_total(cells.len(), founders.len(), 3, 4),
            2592
        );
    }

    #[test]
    fn output_mode_expansion_counts() {
        let root = Path::new("configs/ecology_plateau80/evolution_hunt");
        let spatial = load_named(root, "spatial_graph");
        let spatial_only = build_spatial_cells(&spatial, SweepProfile::EvolutionHunt);
        assert_eq!(spatial_only.len(), 8);
        let expanded = expand_output_mode_cells(spatial_only);
        assert_eq!(
            expanded.len(),
            8 * MovementOutputMode::spatial_sweep_modes().len()
        );
    }
}
