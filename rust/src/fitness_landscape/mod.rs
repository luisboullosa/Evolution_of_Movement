//! Shared fitness-landscape sweep grid + viewer skeleton helpers (testable from lib).

mod probe_timeline;
mod skeleton;
mod sweep_grid;
mod viewer_embed;

pub use probe_timeline::{
    best_by_resources, build_timeline, cell_key_from_run, find_sweep_cell, founder_brain_json,
    probe_meta_from_cfg, sanitize_filename_part, snapshot_path, timeline_has_data,
    timeline_json_filename, write_manifest, write_timeline, CellKey, ProbeMeta, ProbeTimeline,
    RunRef, TimelineManifestEntry,
};
pub use skeleton::{
    build_expected_heatmaps, build_heatmaps_for_live, build_heatmaps_from_cells, cmp_axis_labels,
    overlay_cell_aggregates, pending_cell_count, CellAggregate, GridCellSpec, HeatmapData,
    TrajectoryPoint,
};
pub use sweep_grid::{
    architecture_label, build_cells_to_sweep, cells_to_grid_specs, evolution_hunt_grand_total,
    expand_output_mode_cells, SweepCell, SweepProfile,
};
pub use viewer_embed::{
    escape_json_for_script_embed, inject_live_meta, slim_for_html_embed, LiveMeta,
};
