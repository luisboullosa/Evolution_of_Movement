//! Hexagonal dish layout (circular hex grid) shared by batch sim and interactive viewer.
use crate::hex_grid::HexCoord;
use std::collections::HashMap;

/// Flat-top hex dish: all axial coords within `radius` of center, embedded in a bounding box.
#[derive(Debug, Clone)]
pub struct HexDishLayout {
    pub radius: i32,
    /// Bounding box side length `2 * radius + 1`.
    pub bounding_size: usize,
    pub torus: bool,
    pub active_coords: Vec<HexCoord>,
    pub coord_to_index: HashMap<HexCoord, usize>,
    /// Six neighbor flat indices per cell (`cell * 6 + sector`). Missing neighbors point to self
    /// unless `torus` is enabled, in which case they wrap through the bounding box.
    pub neighbors_flat: Vec<usize>,
}

impl HexDishLayout {
    pub fn cell_count(&self) -> usize {
        self.active_coords.len()
    }

    /// Number of cells in a dish of axial radius `r`: `1 + 3*r*(r+1)`.
    pub fn cell_count_for_radius(radius: i32) -> usize {
        let r = radius.max(0) as u64;
        (1 + 3 * r * (r + 1)) as usize
    }

    pub fn from_radius(radius: i32) -> Self {
        Self::from_radius_options(radius, false)
    }

    pub fn from_radius_options(radius: i32, torus: bool) -> Self {
        let radius = radius.max(0);
        let active_coords = Self::active_coords_for_radius(radius);
        let box_size = radius * 2 + 1;
        let diameter = box_size as usize;
        let cell_count = active_coords.len();
        let mut coord_to_index = HashMap::with_capacity(cell_count);
        for (idx, coord) in active_coords.iter().enumerate() {
            coord_to_index.insert(*coord, idx);
        }

        let mut neighbors_flat = vec![0usize; cell_count * 6];
        for (idx, coord) in active_coords.iter().enumerate() {
            for (sector, ncoord) in coord.neighbors().iter().enumerate() {
                let target = resolve_neighbor_index(*ncoord, idx, &coord_to_index, box_size, torus);
                neighbors_flat[idx * 6 + sector] = target;
            }
        }

        HexDishLayout {
            radius,
            bounding_size: diameter,
            torus,
            active_coords,
            coord_to_index,
            neighbors_flat,
        }
    }

    /// Sorted active coordinates for a dish (same order as flat env indices in snapshots).
    pub fn active_coords_for_radius(radius: i32) -> Vec<HexCoord> {
        let radius = radius.max(0);
        let center = HexCoord::new(radius, radius);
        let mut active_coords = Vec::new();

        for q in -radius..=radius {
            for r in -radius..=radius {
                let axial = HexCoord::new(q, r);
                if axial.distance_to(HexCoord::new(0, 0)) <= radius {
                    active_coords.push(HexCoord::new(center.q + q, center.r + r));
                }
            }
        }
        active_coords.sort_by_key(|c| (c.r, c.q));
        active_coords
    }

    pub fn neighbor_flat(&self, flat: usize, sector: usize) -> usize {
        let pos = flat % self.cell_count().max(1);
        let sector = sector % 6;
        self.neighbors_flat[pos * 6 + sector]
    }

    pub fn edge_resources(&self, env_resources: &[f64], flat: usize) -> [f64; 6] {
        let pos = flat % self.cell_count().max(1);
        let center = env_resources.get(pos).copied().unwrap_or(0.0);
        let mut out = [center; 6];
        for sector in 0..6 {
            let n = self.neighbor_flat(pos, sector);
            out[sector] = env_resources.get(n).copied().unwrap_or(center);
        }
        out
    }
}

fn wrap_coord_in_box(coord: HexCoord, box_size: i32) -> HexCoord {
    HexCoord::new(
        coord.q.rem_euclid(box_size),
        coord.r.rem_euclid(box_size),
    )
}

fn resolve_neighbor_index(
    ncoord: HexCoord,
    self_idx: usize,
    coord_to_index: &HashMap<HexCoord, usize>,
    box_size: i32,
    torus: bool,
) -> usize {
    if let Some(&idx) = coord_to_index.get(&ncoord) {
        return idx;
    }
    if !torus {
        return self_idx;
    }
    let wrapped = wrap_coord_in_box(ncoord, box_size);
    coord_to_index.get(&wrapped).copied().unwrap_or(self_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex_grid::HexGrid;

    #[test]
    fn dish_cell_count_formula() {
        assert_eq!(HexDishLayout::cell_count_for_radius(2), 19);
        let layout = HexDishLayout::from_radius(2);
        assert_eq!(layout.cell_count(), 19);
        let grid = HexGrid::new_hex_dish(2, 10.0);
        assert_eq!(layout.cell_count(), grid.cells.len());
    }

    #[test]
    fn open_dish_edge_neighbor_is_self() {
        let layout = HexDishLayout::from_radius(2);
        let top = HexCoord::new(2, 0);
        let top_idx = *layout.coord_to_index.get(&top).unwrap();
        let north = top.neighbors()[2];
        assert!(!layout.coord_to_index.contains_key(&north));
        assert_eq!(layout.neighbor_flat(top_idx, 2), top_idx);
    }

    #[test]
    fn torus_dish_wraps_across_bounding_box() {
        let layout = HexDishLayout::from_radius_options(2, true);
        let top = HexCoord::new(2, 0);
        let bottom = HexCoord::new(2, 4);
        let top_idx = *layout.coord_to_index.get(&top).unwrap();
        let bottom_idx = *layout.coord_to_index.get(&bottom).unwrap();
        assert_ne!(top_idx, bottom_idx);
        // Sector 2 from top points north off-dish; should wrap to bottom rim.
        assert_eq!(layout.neighbor_flat(top_idx, 2), bottom_idx);
    }
}
