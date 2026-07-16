use evo_movement_rust::config::ArchitecturePolicy;

fn main() {
    evo_movement_rust::headless::run(Some(ArchitecturePolicy::EvolveSpatialGraph));
}
