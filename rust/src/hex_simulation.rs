use crate::config::{Config, GridTopology};
use crate::environment::Environment;
use crate::hex_dish::HexDishLayout;
use crate::hex_grid::HexCoord;

/// Hex-direction sector index 0..5 matching HexCoord::neighbors order.
pub fn direction_to_sector(degrees: f64) -> usize {
    let normalized = degrees.rem_euclid(360.0);
    ((normalized / 60.0).floor() as usize).min(5)
}

pub fn grid_cell_count(cfg: &Config) -> usize {
    if cfg.hex_dish_radius > 0 {
        return HexDishLayout::cell_count_for_radius(cfg.hex_dish_radius);
    }
    match cfg.grid_topology {
        GridTopology::Ring1d => cfg.grid_width.max(1),
        GridTopology::Hex2d => {
            let h = if cfg.grid_height > 0 {
                cfg.grid_height
            } else {
                cfg.grid_width
            };
            cfg.grid_width.max(1) * h.max(1)
        }
    }
}

pub fn flat_to_hex(flat: usize, width: usize) -> HexCoord {
    HexCoord::from_flat(flat, width.max(1))
}

pub fn hex_to_flat(coord: HexCoord, width: usize, height: usize) -> usize {
    let q = coord.q.rem_euclid(width as i32) as usize;
    let r = coord.r.rem_euclid(height as i32) as usize;
    r * width + q
}

pub fn wrap_hex(coord: HexCoord, width: usize, height: usize) -> HexCoord {
    HexCoord::new(
        coord.q.rem_euclid(width as i32),
        coord.r.rem_euclid(height as i32),
    )
}

pub fn neighbor_flat(flat: usize, sector: usize, width: usize, height: usize) -> usize {
    let coord = flat_to_hex(flat, width);
    let neighbors = coord.neighbors();
    let wrapped = wrap_hex(neighbors[sector % 6], width, height);
    hex_to_flat(wrapped, width, height)
}

pub fn edge_resources_hex_dish(env: &Environment, flat: usize, dish: &HexDishLayout) -> [f64; 6] {
    dish.edge_resources(&env.resources, flat)
}

pub fn neighbor_flat_dish(flat: usize, sector: usize, dish: &HexDishLayout) -> usize {
    dish.neighbor_flat(flat, sector)
}

pub fn edge_resources_hex(
    env: &Environment,
    flat: usize,
    width: usize,
    height: usize,
) -> [f64; 6] {
    let coord = flat_to_hex(flat, width);
    let neighbors = coord.neighbors();
    let mut out = [0.0; 6];
    for (i, n) in neighbors.iter().enumerate() {
        let w = wrap_hex(*n, width, height);
        let idx = hex_to_flat(w, width, height);
        out[i] = env.resources.get(idx).copied().unwrap_or(0.0);
    }
    out
}

pub fn edge_resources_ring(env: &Environment, pos: usize) -> [f64; 6] {
    let w = env.width.max(1);
    let left = env.resources[(pos + w - 1) % w];
    let center = env.resources[pos % w];
    let right = env.resources[(pos + 1) % w];
    [left, left, center, right, right, center]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_sectors_cover_hex() {
        assert_eq!(direction_to_sector(0.0), 0);
        assert_eq!(direction_to_sector(59.0), 0);
        assert_eq!(direction_to_sector(60.0), 1);
    }

    #[test]
    fn hex_neighbor_wraps() {
        let w = 10;
        let h = 10;
        let n = neighbor_flat(0, 0, w, h);
        assert!(n < w * h);
    }
}
