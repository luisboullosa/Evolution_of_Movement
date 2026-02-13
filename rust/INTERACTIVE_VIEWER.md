# Interactive Evolution Simulation Viewer

A real-time, GPU-accelerated interactive visualization of evolving digital organisms in a hexagonal grid environment.

## Features

### Core Simulation
- **Hexagonal Grid Environment**: Organisms live in a 2D hexagonal grid (petri dish)
- **Light-Based Resources**: Resource availability represented by light intensity
- **Neural Network Agents**: Each individual has a neural network controlling movement and gathering behavior
- **Evolution**: Reproduction with mutation creates diverse lineages over time

### Interactive Visualization
- **Real-time Display**: Watch the simulation unfold in real-time
- **Multiple Petri Dishes**: Create and manage multiple independent environments
- **Individual Selection**: Click on any organism to inspect its properties
- **Pause/Play/Step Controls**: Full control over simulation execution
- **Speed Control**: Adjust simulation speed from 0.1x to 10x

### Visualization Panels
1. **Main View**: Hexagonal grid showing individuals (colored by lineage) and resources (brightness = light level)
2. **Control Panel**: Simulation parameters and statistics
3. **Neural Network Diagram**: Visualize the selected individual's brain
   - Node colors represent biases
   - Connection colors represent weights (red=negative, blue=positive)
   - Connection thickness represents weight magnitude
4. **Cladogram Panel**: Phylogenetic tree information
   - Shows lineage depth and ancestry of selected individual
   - Genetic distances calculated from neural network weights

### Color System
- Each founding individual receives a unique color
- All offspring inherit their parent's color
- This allows visual tracking of different lineages

### Save/Load System
- Save complete simulation state to JSON
- Load previously saved states
- Preserves: individuals, positions, neural networks, colors, ancestry, resources

## Building

### Prerequisites
- Rust 1.70 or newer
- Cargo

### Build Instructions

```bash
cd rust
cargo build --release
```

## Running

### Interactive Viewer

```bash
cd rust
cargo run --release --bin evo_movement_interactive
```

### Controls

**Menu Bar:**
- File → New Simulation: Restart current petri dish
- File → Save State: Save to `simulation_state.json`
- File → Load State: Load from `simulation_state.json`
- View → Show Neural Network: Toggle NN visualization panel
- View → Show Cladogram: Toggle phylogenetic tree panel

**Simulation Controls:**
- ▶ Play / ⏸ Pause: Start/stop simulation
- ⏭ Step: Advance simulation by one timestep
- 🔄 Restart: Clear and reinitialize population
- Speed slider: Adjust simulation speed (0.1x - 10x)
- Steps per Frame: How many simulation steps per visual update

**Petri Dish Controls:**
- ➕ Add Petri Dish: Create a new independent environment
- Active Dish dropdown: Switch between petri dishes
- 🧬 Initialize Population: Create initial organisms

**Selection:**
- Click on any individual to select it
- Selected individuals show:
  - ID, age, resources, position
  - Neural network architecture and weights
  - Lineage ancestry

## Configuration

Initial parameters can be adjusted in the code:

```rust
let mut config = Config::default();
config.grid_width = 50;           // Grid cells wide
config.initial_population = 10;    // Starting organisms
config.nn_nodes = 10;              // Hidden layer size
config.nn_inputs = 5;              // Input count
config.nn_layers = 2;              // Network depth
```

## Architecture

### Technology Stack
- **Rust**: Core simulation and logic
- **egui/eframe**: Cross-platform GUI framework
- **wgpu**: GPU compute support (planned for acceleration)
- **serde**: Serialization for save/load

### Key Modules
- `hex_grid.rs`: Hexagonal coordinate system and grid management
- `individual.rs`: Organism behavior with color inheritance
- `neural_network.rs`: Feedforward neural networks
- `cladogram.rs`: Phylogenetic tree tracking and genetic distances
- `nn_viz.rs`: Neural network visualization renderer
- `save_state.rs`: Serialization for persistence
- `interactive.rs`: Main GUI application

### Performance
- Target: 60 timesteps per second
- Adjustable via speed multiplier (up to 10x = 600 steps/sec)
- Future: GPU acceleration for large populations

## Memory Safety
All code is written in Rust, providing memory safety guarantees through:
- Ownership system prevents use-after-free
- Borrow checker prevents data races
- No null pointer dereferencing
- Automatic memory management without garbage collection

## Future Enhancements
- [ ] Drag-and-drop individuals between petri dishes
- [ ] GPU-accelerated simulation for large populations
- [ ] Full phylogenetic tree visualization with branch rendering
- [ ] File picker dialogs for save/load
- [ ] Multiple selection and batch operations
- [ ] Recording and playback of simulations
- [ ] Export visualizations as images/videos

## Development

### Running Tests

```bash
cd rust
cargo test
```

### Building Documentation

```bash
cd rust
cargo doc --open
```

## License

See LICENSE file in the repository root.
