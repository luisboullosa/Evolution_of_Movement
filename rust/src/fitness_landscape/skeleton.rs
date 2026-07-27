use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryPoint {
    pub step: i32,
    pub fitness: f64,
    pub pop_ratio: f64,
    pub avg_complexity: f64,
    pub population: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GridCellSpec {
    pub architecture: String,
    pub x_label: String,
    pub y_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CellAggregate {
    pub architecture: String,
    pub x_label: String,
    pub y_label: String,
    pub founder_mode: String,
    pub replicates: usize,
    pub fitness_mean: f64,
    pub fitness_std: f64,
    pub pop_ratio_mean: f64,
    pub complexity_mean: f64,
    #[serde(default)]
    pub data_na: bool,
    #[serde(default)]
    pub trajectory_mean: Vec<TrajectoryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeatmapData {
    pub architecture: String,
    pub founder_mode: String,
    pub x_labels: Vec<String>,
    pub y_labels: Vec<String>,
    pub fitness_matrix: Vec<Vec<Option<f64>>>,
    #[serde(default)]
    pub fitness_na_matrix: Vec<Vec<bool>>,
    pub pop_ratio_matrix: Vec<Vec<Option<f64>>>,
    pub complexity_matrix: Vec<Vec<Option<f64>>>,
}

pub fn cmp_axis_labels(a: &str, b: &str) -> std::cmp::Ordering {
    fn key(label: &str) -> (String, Option<u64>) {
        if let Some((prefix, rest)) = label.split_once('=') {
            if let Ok(n) = rest.parse::<u64>() {
                return (prefix.to_string(), Some(n));
            }
        }
        (label.to_string(), None)
    }
    let (pa, na) = key(a);
    let (pb, nb) = key(b);
    match pa.cmp(&pb) {
        std::cmp::Ordering::Equal => match (na, nb) {
            (Some(a), Some(b)) => a.cmp(&b),
            (None, None) => a.cmp(b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
        },
        other => other,
    }
}

fn sort_axis_labels(labels: &mut [String]) {
    labels.sort_by(|a, b| cmp_axis_labels(a, b));
}

pub fn build_expected_heatmaps(
    cells: &[GridCellSpec],
    founder_modes: &[String],
) -> Vec<HeatmapData> {
    let mut by_arch: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for cell in cells {
        by_arch
            .entry(cell.architecture.clone())
            .or_default()
            .push((cell.x_label.clone(), cell.y_label.clone()));
    }
    let mut out = Vec::new();
    for (arch, coords) in by_arch {
        let mut x_labels: Vec<String> = coords.iter().map(|(x, _)| x.clone()).collect();
        sort_axis_labels(&mut x_labels);
        x_labels.dedup();
        let mut y_labels: Vec<String> = coords.iter().map(|(_, y)| y.clone()).collect();
        sort_axis_labels(&mut y_labels);
        y_labels.dedup();
        let n_x = x_labels.len();
        let n_y = y_labels.len();
        for founder in founder_modes {
            out.push(HeatmapData {
                architecture: arch.clone(),
                founder_mode: founder.clone(),
                x_labels: x_labels.clone(),
                y_labels: y_labels.clone(),
                fitness_matrix: vec![vec![None; n_x]; n_y],
                fitness_na_matrix: vec![vec![false; n_x]; n_y],
                pop_ratio_matrix: vec![vec![None; n_x]; n_y],
                complexity_matrix: vec![vec![None; n_x]; n_y],
            });
        }
    }
    out
}

pub fn overlay_cell_aggregates(heatmaps: &mut [HeatmapData], cells: &[CellAggregate]) {
    for c in cells {
        let Some(hm) = heatmaps
            .iter_mut()
            .find(|h| h.architecture == c.architecture && h.founder_mode == c.founder_mode)
        else {
            continue;
        };
        let Some(xi) = hm.x_labels.iter().position(|x| x == &c.x_label) else {
            continue;
        };
        let Some(yi) = hm.y_labels.iter().position(|y| y == &c.y_label) else {
            continue;
        };
        hm.fitness_na_matrix[yi][xi] = c.data_na;
        hm.fitness_matrix[yi][xi] = if c.data_na {
            None
        } else {
            Some(c.fitness_mean)
        };
        hm.pop_ratio_matrix[yi][xi] = Some(c.pop_ratio_mean);
        hm.complexity_matrix[yi][xi] = Some(c.complexity_mean);
    }
}

pub fn build_heatmaps_for_live(
    cells: &[CellAggregate],
    expected_grid: Option<&[GridCellSpec]>,
    founder_modes: &[String],
) -> Vec<HeatmapData> {
    if let Some(grid) = expected_grid {
        let mut heatmaps = build_expected_heatmaps(grid, founder_modes);
        overlay_cell_aggregates(&mut heatmaps, cells);
        heatmaps
    } else {
        build_heatmaps_from_cells(cells)
    }
}

pub fn build_heatmaps_from_cells(cells: &[CellAggregate]) -> Vec<HeatmapData> {
    let specs: Vec<GridCellSpec> = cells
        .iter()
        .map(|c| GridCellSpec {
            architecture: c.architecture.clone(),
            x_label: c.x_label.clone(),
            y_label: c.y_label.clone(),
        })
        .collect();
    let founders: Vec<String> = cells
        .iter()
        .map(|c| c.founder_mode.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut heatmaps = build_expected_heatmaps(&specs, &founders);
    overlay_cell_aggregates(&mut heatmaps, cells);
    heatmaps
}

pub fn pending_cell_count(heatmaps: &[HeatmapData]) -> usize {
    heatmaps
        .iter()
        .flat_map(|hm| hm.fitness_matrix.iter().flatten())
        .filter(|v| v.is_none())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn founders() -> Vec<String> {
        vec!["random_uniform".into(), "random_small".into()]
    }

    #[test]
    fn expected_heatmap_all_cells_pending() {
        let cells = vec![
            GridCellSpec {
                architecture: "Perceptron".into(),
                x_label: "nodes=1".into(),
                y_label: "baseline".into(),
            },
            GridCellSpec {
                architecture: "Perceptron".into(),
                x_label: "nodes=2".into(),
                y_label: "baseline".into(),
            },
        ];
        let heatmaps = build_expected_heatmaps(&cells, &founders());
        assert_eq!(heatmaps.len(), 2);
        assert_eq!(heatmaps[0].fitness_matrix, vec![vec![None, None]]);
        assert_eq!(pending_cell_count(&heatmaps), 4);
    }

    #[test]
    fn overlay_fills_one_cell_leaves_rest_pending() {
        let cells = vec![GridCellSpec {
            architecture: "Perceptron".into(),
            x_label: "nodes=1".into(),
            y_label: "baseline".into(),
        }];
        let mut heatmaps = build_expected_heatmaps(&cells, &founders());
        overlay_cell_aggregates(
            &mut heatmaps,
            &[CellAggregate {
                architecture: "Perceptron".into(),
                x_label: "nodes=1".into(),
                y_label: "baseline".into(),
                founder_mode: "random_uniform".into(),
                replicates: 1,
                fitness_mean: 0.42,
                fitness_std: 0.0,
                pop_ratio_mean: 0.8,
                complexity_mean: 1.2,
                data_na: false,
                trajectory_mean: vec![],
            }],
        );
        let uniform = heatmaps
            .iter()
            .find(|h| h.founder_mode == "random_uniform")
            .unwrap();
        assert_eq!(uniform.fitness_matrix[0][0], Some(0.42));
        let small = heatmaps
            .iter()
            .find(|h| h.founder_mode == "random_small")
            .unwrap();
        assert_eq!(small.fitness_matrix[0][0], None);
        assert_eq!(pending_cell_count(&heatmaps), 1);
    }

    #[test]
    fn axis_labels_sort_numeric_suffix() {
        let mut labels = vec!["nodes=8".into(), "nodes=2".into(), "nodes=24".into()];
        sort_axis_labels(&mut labels);
        assert_eq!(labels, vec!["nodes=2", "nodes=8", "nodes=24"]);
    }
}
