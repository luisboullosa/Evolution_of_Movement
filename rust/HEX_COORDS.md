# Hex Coordinates in the Interactive Rust Viewer

This document explains how hex coordinates are represented and used in the Rust interactive simulation.

## Coordinate system

The viewer uses axial hex coordinates:

- q: axial column
- r: axial row
- implicit third axis: s = -q - r

This model is implemented in src/hex_grid.rs with the HexCoord struct.

## Core types

### HexCoord

HexCoord stores one hex cell location:

- q: i32
- r: i32

Important methods:

- new(q, r): construct a coordinate.
- neighbors(): returns the 6 adjacent cells in a fixed direction order:
  1. q+1, r
  2. q+1, r-1
  3. q, r-1
  4. q-1, r
  5. q-1, r+1
  6. q, r+1
- distance_to(other): axial hex distance using cube-distance formula.
- to_pixel(hex_size): axial -> screen position conversion for flat-top hexes.
- from_pixel(x, y, hex_size): screen position -> nearest axial coordinate.
- to_flat(width) / from_flat(index, width): index mapping helper used for compact arrays.

### HexCell

HexCell stores per-cell state:

- resource_level: current resource value
- individual_indices: indices of organisms currently in this cell

### HexGrid

HexGrid stores the active dish geometry:

- width, height: bounding dimensions
- hex_size: rendered hex radius/scale factor
- cells: sparse HashMap<HexCoord, HexCell>

Important constructors:

- new(width, height, hex_size): rectangular populated axial area.
- new_hex_dish(radius, hex_size): true hexagonal dish shape inside a bounding box.

## Geometry and rendering

The renderer uses a flat-top axial projection:

- x = hex_size * (sqrt(3) * q + sqrt(3)/2 * r)
- y = hex_size * (3/2 * r)

Neighbor directions around a center hex C(q, r):

```text
       N2: (q,   r-1)
     N1: (q+1, r-1)     \
           \
N3: (q-1, r)  ----  C(q, r)  ----  N0: (q+1, r)
           /
     N4: (q-1, r+1)     /
       N5: (q,   r+1)
```

This matches the `neighbors()` order in `HexCoord`:
`[N0, N1, N2, N3, N4, N5]`.

- flat-top means each hex has a horizontal top/bottom edge (not a point-up orientation).
- hex_size is the scale/radius factor; increasing it scales both spacing and hex polygon size.
- moving +q primarily shifts right; moving +r shifts down-right in screen space, which is why both q and r contribute to x.
- the sqrt(3) terms encode the horizontal spacing of regular hex geometry, while 3/2 sets vertical row spacing for flat-top packing.

Why this projection is useful:

- it gives a linear map from axial coordinates to screen coordinates,
- neighbor relations are preserved visually (the 6 axial neighbors appear as adjacent hexes),
- converting between simulation coordinates and rendered coordinates is predictable and efficient.

Pointer/click conversion works in the reverse direction:

- from_pixel applies the inverse axial transform to get fractional (q, r),
- round_hex then snaps that fractional coordinate to the nearest valid hex center.

Rounding detail:

- axial coordinates are equivalent to cube coordinates (q, r, s) with s = -q - r,
- round_hex computes rounded q, r, s and then corrects the component with the largest rounding error,
- this avoids drift and ensures the final integer coordinate is a valid hex cell, especially near shared borders/corners.

User interaction (click/drag) converts pointer coordinates back with from_pixel and rounds to the nearest valid hex.

The interactive view computes dish bounds from actual active coordinates (not just width/height), then offsets/scales so the full dish is visible.

## Dish shape and active cells

For hex dishes:

- Logical radius is set in cells (for example, 100).
- Active cells are those whose axial distance to dish center is <= radius.
- Non-active coordinates are absent from the cells map.

The simulation keeps an active_coords vector and coord_to_index map so it can:

- map between compact simulation arrays and sparse hex coordinates,
- sample neighbor resources by direction,
- rebuild occupancy efficiently each frame.

## Simulation interactions with hex coords

In the interactive simulation step:

1. For each organism, its current position index maps to a HexCoord.
2. The 6 neighbors are computed with neighbors().
3. One resource sensory value is read per edge direction (6 inputs).
4. Movement direction (degrees) is quantized to one of 6 sectors.
5. If movement occurs and target neighbor exists in coord_to_index, the organism moves one hex.

Multiple organisms can occupy the same hex because each cell stores a list of individual indices.

## Cladogram and spatial independence

Phylogeny tracking (cladogram) is independent of spatial coordinates:

- lineage and genetic distance metadata live in src/cladogram.rs,
- spatial position remains in each organism position field,
- the viewer combines both for display (dish + pedigree panel).

## Related files

- src/hex_grid.rs: coordinate math, grid/cell data structures
- src/bin/interactive.rs: rendering, click/drag selection, one-step hex movement
- src/individual.rs: decision inputs now include 6 edge resource sensors
- src/cladogram.rs: lineage tree and genetic-distance branch metadata

## Tests

Hex coordinate behavior has unit tests in src/hex_grid.rs:

- neighbor count
- flat index mapping round-trips
- pixel conversion round-trip
- hex dish cell count formula
