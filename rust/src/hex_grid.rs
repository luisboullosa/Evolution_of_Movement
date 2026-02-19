/// Hexagonal grid implementation for the interactive viewer
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub q: i32,  // column
    pub r: i32,  // row
}

impl HexCoord {
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Convert flat index to hex coordinates
    pub fn from_flat(index: usize, width: usize) -> Self {
        let row = index / width;
        let col = index % width;
        Self {
            q: col as i32,
            r: row as i32,
        }
    }

    /// Hex distance in axial coordinates.
    pub fn distance_to(&self, other: HexCoord) -> i32 {
        let dq = self.q - other.q;
        let dr = self.r - other.r;
        let ds = -dq - dr;
        (dq.abs() + dr.abs() + ds.abs()) / 2
    }

    /// Convert hex coordinates to flat index
    pub fn to_flat(&self, width: usize) -> usize {
        (self.r as usize) * width + (self.q as usize)
    }

    /// Get the 6 neighboring hexagons (axial coordinates)
    pub fn neighbors(&self) -> [HexCoord; 6] {
        [
            HexCoord::new(self.q + 1, self.r),     // right
            HexCoord::new(self.q + 1, self.r - 1), // top-right
            HexCoord::new(self.q, self.r - 1),     // top-left
            HexCoord::new(self.q - 1, self.r),     // left
            HexCoord::new(self.q - 1, self.r + 1), // bottom-left
            HexCoord::new(self.q, self.r + 1),     // bottom-right
        ]
    }

    /// Convert axial coordinates to pixel position (flat-top hexagons)
    pub fn to_pixel(&self, hex_size: f32) -> (f32, f32) {
        let x = hex_size * (3.0_f32.sqrt() * self.q as f32 + 3.0_f32.sqrt() / 2.0 * self.r as f32);
        let y = hex_size * (3.0 / 2.0 * self.r as f32);
        (x, y)
    }

    /// Convert pixel position to axial coordinates (flat-top hexagons)
    pub fn from_pixel(x: f32, y: f32, hex_size: f32) -> Self {
        let q = (3.0_f32.sqrt() / 3.0 * x - 1.0 / 3.0 * y) / hex_size;
        let r = (2.0 / 3.0 * y) / hex_size;
        Self::round_hex(q, r)
    }

    /// Round fractional hex coordinates to nearest hex
    fn round_hex(q: f32, r: f32) -> Self {
        let s = -q - r;
        let mut rq = q.round();
        let mut rr = r.round();
        let rs = s.round();

        let q_diff = (rq - q).abs();
        let r_diff = (rr - r).abs();
        let s_diff = (rs - s).abs();

        if q_diff > r_diff && q_diff > s_diff {
            rq = -rr - rs;
        } else if r_diff > s_diff {
            rr = -rq - rs;
        }

        Self {
            q: rq as i32,
            r: rr as i32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HexGrid {
    pub width: usize,
    pub height: usize,
    pub hex_size: f32,
    /// Map from hex coordinates to data
    pub cells: HashMap<HexCoord, HexCell>,
}

#[derive(Debug, Clone)]
pub struct HexCell {
    pub resource_level: f64,
    /// Indices of individuals at this location (into the simulation's individuals Vec)
    pub individual_indices: Vec<usize>,
}

impl HexCell {
    pub fn new() -> Self {
        Self {
            resource_level: 0.0,
            individual_indices: Vec::new(),
        }
    }
}

impl HexGrid {
    pub fn new(width: usize, height: usize, hex_size: f32) -> Self {
        let mut cells = HashMap::new();
        for r in 0..height as i32 {
            for q in 0..width as i32 {
                cells.insert(HexCoord::new(q, r), HexCell::new());
            }
        }

        Self {
            width,
            height,
            hex_size,
            cells,
        }
    }

    /// Create a hexagonal dish (axial hexagon) embedded in a bounding box.
    pub fn new_hex_dish(radius: i32, hex_size: f32) -> Self {
        let radius = radius.max(0);
        let diameter = (radius * 2 + 1) as usize;
        let center = HexCoord::new(radius, radius);
        let mut cells = HashMap::new();

        for q in -radius..=radius {
            for r in -radius..=radius {
                let axial = HexCoord::new(q, r);
                if axial.distance_to(HexCoord::new(0, 0)) <= radius {
                    let shifted = HexCoord::new(center.q + q, center.r + r);
                    cells.insert(shifted, HexCell::new());
                }
            }
        }

        Self {
            width: diameter,
            height: diameter,
            hex_size,
            cells,
        }
    }

    pub fn get_cell(&self, coord: HexCoord) -> Option<&HexCell> {
        self.cells.get(&coord)
    }

    pub fn get_cell_mut(&mut self, coord: HexCoord) -> Option<&mut HexCell> {
        self.cells.get_mut(&coord)
    }

    /// Get the dimensions of the grid in pixels
    pub fn pixel_dimensions(&self) -> (f32, f32) {
        let width_px = self.hex_size * 3.0_f32.sqrt() * self.width as f32;
        let height_px = self.hex_size * 1.5 * self.height as f32 + self.hex_size * 0.5;
        (width_px, height_px)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_coord_neighbors() {
        let hex = HexCoord::new(0, 0);
        let neighbors = hex.neighbors();
        assert_eq!(neighbors.len(), 6);
    }

    #[test]
    fn test_hex_coord_to_flat() {
        let hex = HexCoord::new(2, 3);
        let flat = hex.to_flat(10);
        assert_eq!(flat, 32); // 3 * 10 + 2
    }

    #[test]
    fn test_hex_coord_from_flat() {
        let hex = HexCoord::from_flat(32, 10);
        assert_eq!(hex.q, 2);
        assert_eq!(hex.r, 3);
    }

    #[test]
    fn test_pixel_conversion() {
        let hex = HexCoord::new(0, 0);
        let (x, y) = hex.to_pixel(10.0);
        let back = HexCoord::from_pixel(x, y, 10.0);
        assert_eq!(hex, back);
    }

    #[test]
    fn test_hex_dish_cell_count() {
        let grid = HexGrid::new_hex_dish(2, 10.0);
        // 1 + 3r(r+1) for hex radius r
        assert_eq!(grid.cells.len(), 19);
    }
}
