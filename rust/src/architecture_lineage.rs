use crate::brain::StructuralMutation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureLineageEvent {
    pub timestep: i32,
    pub parent_id: i32,
    pub child_id: i32,
    pub parent_kind: String,
    pub child_kind: String,
    pub parent_complexity: usize,
    pub child_complexity: usize,
    pub parent_structure: String,
    pub child_structure: String,
    pub genetic_distance: f64,
    pub structural_mutation: String,
}

impl ArchitectureLineageEvent {
    pub fn from_reproduction(
        timestep: i32,
        parent: &crate::individual::Individual,
        child: &crate::individual::Individual,
        genetic_distance: f64,
        mutation: &StructuralMutation,
    ) -> Self {
        ArchitectureLineageEvent {
            timestep,
            parent_id: parent.id,
            child_id: child.id,
            parent_kind: parent.brain.kind().as_str().to_string(),
            child_kind: child.brain.kind().as_str().to_string(),
            parent_complexity: parent.complexity(),
            child_complexity: child.complexity(),
            parent_structure: parent.brain.structure_signature(),
            child_structure: child.brain.structure_signature(),
            genetic_distance,
            structural_mutation: format!("{:?}", mutation),
        }
    }
}

pub fn write_lineage_jsonl(events: &[ArchitectureLineageEvent], path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    for ev in events {
        let line = serde_json::to_string(ev).unwrap_or_else(|_| "{}".to_string());
        writeln!(file, "{}", line)?;
    }
    Ok(())
}
