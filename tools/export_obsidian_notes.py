#!/usr/bin/env python3
"""
Export curated Obsidian notes from Graphify output.
"""
from pathlib import Path
import re

VAULT_DIR = Path(r"D:\Development\KnowledgeBase\Projects\Movement\Graphify")
VAULT_DIR.mkdir(parents=True, exist_ok=True)
WIKI_DIR = VAULT_DIR / "wiki"
WIKI_DIR.mkdir(parents=True, exist_ok=True)

# 1. Graphify Overview.md
overview_content = """---
project: Movement
type: graphify-note
source: graphify
updated: 2026-10-22
---

# Graphify Overview: Movement

> Central knowledge graph and architectural entry point for the **Movement** (evo-movement-sim) project.

---

## Project Purpose
- **Evolution of Adaptive Movement**: Simulates digital organisms navigating dynamic 2D environmental resource gradients driven by continuous Perlin noise fields.
- **Heterogeneous Neural Substrates**: Explores comparative evolvability across multiple brain architectures, ranging from baseline Perceptrons and fixed MLPs to flexible Spatial Graphs and dynamic topological [[GooGraphBrain|Goo Graphs]].
- **High-Performance Rust Core**: Multi-threaded, SIMD-friendly engine ([rust/src/](file:///d:/Development/Movement/rust/src/)) with historical C++ codebase safely archived in [archive/cpp/](file:///d:/Development/Movement/archive/cpp/) to eliminate symbol ambiguity.
- **Decoupled DAG Architecture**: Clean hierarchical dataflow without cyclic dependencies across brain modules or simulation step engines.
- **Hexagonal Spatial Geometry**: Utilizes 6-connected hexagonal grid topology ([[HexCoord]], [[HexGrid]]) across isolated or toroidal dish environments ([[PetriDish]]).
- **Deep Lineage & Visualization**: Tracks phylogenetic descent and genetic distance via [[Cladogram]] structures, visualized through a native cross-platform GUI ([rust/src/bin/interactive.rs](file:///d:/Development/Movement/rust/src/bin/interactive.rs)) and Web viewer.

---

## Curated Vault Navigation
- [[Architecture Map]]: Subsystem topology, god nodes, cycle elimination, and coupling analysis.
- [[Key Concepts]]: Deep dive into core domain abstractions ([[BrainGenome]], [[Config]], [[ArchitecturePolicy]], [[PetriDish]], etc.).
- [[Communities]]: Clustering decomposition across 106 functional sub-graphs.
- [[Open Questions]]: Architectural hotspots, knowledge gaps, and refactoring targets.
- [[wiki/index|Knowledge Graph Wiki]]: Alphabetical and community index of all 116 generated wiki articles.

---

## Graph Metadata & Freshness
- **Last Rebuild Date**: 2026-10-22
- **Source Git Commit**: `2b908e25`
- **Graph Statistics**: 1,826 nodes · 4,319 edges · 106 communities
- **Edge Quality**: 97% `EXTRACTED` facts · 3% `INFERRED` relationships
- **Graphify Version**: `0.9.93`
- **Graph Schema Version**: `1`

---

## How to Rebuild
To regenerate the graph and update notes locally:
```bash
# 1. Re-extract code AST (code-only, privacy-preserving, zero API calls)
graphify . --code-only --no-viz --wiki

# 2. Re-cluster and generate report without external LLM calls
graphify cluster-only . --no-label --no-viz

# 3. Export fresh wiki
graphify export wiki

# 4. Sync curated notes and wiki to Obsidian vault
python tools/export_obsidian_notes.py
```

---

## How to Query the Graph
Query the codebase graph before reading files or grepping:
- **Concept Explanation**: `graphify explain "BrainGenome"`
- **Relationship Path**: `graphify path "rust/src/simulation.rs::Simulation" "GooGraphBrain"`
- **Semantic Question**: `graphify query "How does structural mutation work in GooGraph?"`
"""

# 2. Architecture Map.md
arch_content = """---
project: Movement
type: graphify-note
source: graphify
updated: 2026-10-22
---

# Architecture Map: Movement

> High-level decomposition of the Movement repository based on AST extraction and static dependency graph analysis.

---

## 1. System Subsystems

```mermaid
graph TD
    CLI["CLI & Run Harnesses<br/>(evo_movement_rust, interactive)"] --> SIM["Simulation Engine<br/>(Simulation, StepEngine)"]
    SIM --> ENV["Environment & Geometry<br/>(HexGrid, HexCoord, PetriDish)"]
    SIM --> POP["Population & Individuals<br/>(Individual, BrainGenome)"]
    POP --> BRAIN["Brain Substrates<br/>(Perceptron, MLP, SpatialGraph, GooGraph)"]
    BRAIN --> TYPES["Brain Shared Types<br/>(BrainOutput, ForwardContext, SpatialRuntime)"]
    SIM --> PERLIN["Noise Generator<br/>(Perlin 2D+Time)"]
    SIM --> CLAD["Phylogeny & Lineage<br/>(Cladogram, GenotypeLineage)"]
    SIM -.-> GPU["CUDA Acceleration<br/>(CachedCudaRuntime, DecisionRowGpu)"]
```

### Core Subsystems:
1. **Simulation Engine** ([rust/src/simulation.rs](file:///d:/Development/Movement/rust/src/simulation.rs), [rust/src/step_engine.rs](file:///d:/Development/Movement/rust/src/step_engine.rs)):
   - Controls lifecycle ticks, carrying capacity constraints, and individual action dispatch.
   - Computes movement steps, metabolic costs, consumption of light/food, and culling.
2. **Hexagonal Spatial Substrate** ([rust/src/hex_grid.rs](file:///d:/Development/Movement/rust/src/hex_grid.rs), [rust/src/hex_coord.rs](file:///d:/Development/Movement/rust/src/hex_coord.rs)):
   - Axial coordinate geometry for 6-connected neighbor lookups, flat vs. axial conversions, and boundary wrapping.
   - [[PetriDish]] encapsulates isolated dish domains and toroidal edge wrapping.
3. **Neural Substrates & Evolutionary Genomes** ([rust/src/brain/](file:///d:/Development/Movement/rust/src/brain/)):
   - Shared data contracts isolated in [rust/src/brain/types.rs](file:///d:/Development/Movement/rust/src/brain/types.rs).
   - Polymorphic dispatch unified under [[BrainGenome]]:
     - `Perceptron`: Baseline single-layer sensory-motor weight mapping.
     - `MlpBrain`: Fixed feed-forward multi-layer networks.
     - `SpatialGraph`: Spatially embedded graphs with cardinal sector activations.
     - `GooGraphBrain`: Dynamic topological graphs supporting structural mutations (`GooAddAggregate`, `GooAddGenerate`, `GooAddKeepBest`, `GooAddScore`).
4. **Phylogeny, Lineage & Diagnostics** ([rust/src/cladogram.rs](file:///d:/Development/Movement/rust/src/cladogram.rs), [rust/src/individual.rs](file:///d:/Development/Movement/rust/src/individual.rs)):
   - Lineage node tracking with parent IDs, death timestamps, and genetic distance algorithms.
   - Golden-angle visual color inheritance for visual identification of ancestral lineages.
5. **GPU Acceleration** ([rust/cuda/](file:///d:/Development/Movement/rust/cuda/), [rust/src/cuda_runtime.rs](file:///d:/Development/Movement/rust/src/cuda_runtime.rs)):
   - High-throughput forward-pass evaluation for massive populations via CUDA kernel modules.
6. **GUI & Visualization** ([rust/src/bin/interactive.rs](file:///d:/Development/Movement/rust/src/bin/interactive.rs), [webviewer/](file:///d:/Development/Movement/webviewer/)):
   - Real-time `egui` interactive viewer with interactive dish inspection and neural connection visualizers.
   - Standalone web bundle playback viewer.

---

## 2. God Nodes (Central Abstractions)
These 10 nodes possess the highest degree centrality and constitute the structural backbone:

| Node | Type | Degree | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| [[BrainGenome]] | Enum / Struct | 68 | Polymorphic neural genotype and structural mutation dispatcher |
| [[ArchitecturePolicy]] | Enum | 46 | Governance over permitted neural substrate evolutions |
| [[Simulation]] | Struct | 38 | Main simulation world state, tick loop, and population manager |
| [[PetriDish]] | Struct | 37 | Dish container geometry, boundary physics, and multi-dish layouts |
| [[GooGraphBrain]] | Struct | 36 | Dynamic graph neural architecture with primitive operator nodes |
| [[PopulationTensor]] | Struct | 34 | Dense memory packing for batched forward evaluations |
| [[MovementOutputMode]] | Enum | 32 | Translation of neural output activations to directional movement |
| `main()` | Function | 32 | Top-level entrypoints across CLI tools, viewers, and pipelines |
| `SpatialGraphBrain` | Struct | 31 | Landmark-based spatial neural graph implementation |
| [[HexCoord]] | Struct | 31 | Axial coordinate primitives and neighborhood offset tables |

---

## 3. Coupling & Cycle Resolution
All 13 multi-file circular dependency chains previously identified in `rust/src/brain/` and `hex_simulation.rs` have been eliminated:
- **Brain Subsystem Cycles (Resolved)**: Extracted `BrainOutput`, `ForwardContext`, `SpatialRuntime`, constants (`SENSOR_DIM`, `THOUGHT_DIM`, `EMBED_DIM`), and parameter math (`l2_slices`, `perturb_param`) into `rust/src/brain/types.rs`. All concrete brain models now import downstream from `types.rs`, eliminating bidirectional cycles with `genome.rs` and `perceptron.rs`.
- **Simulation Step Cycle (Resolved)**: Pruned dead legacy forwarders in `hex_simulation.rs`, breaking the circular dependency with `step_engine.rs`.
- **Legacy C++ Archival**: Legacy C++ sources (`src/`, `include/`, `tests/`) moved to `archive/cpp/` and excluded in `.graphifyignore`, resolving cross-language name collisions.

---

## 4. Relationship Categorization
- **EXTRACTED (97%)**: Exact AST-level symbol declarations, struct field types, direct function invocations, and module imports.
- **INFERRED (3%)**: High-confidence semantic caller/callee relationships verified by static call site proximity.
"""

# 3. Key Concepts.md
concepts_content = """---
project: Movement
type: graphify-note
source: graphify
updated: 2026-10-22
---

# Key Concepts: Movement

> Detailed breakdown of the primary domain abstractions extracted from the codebase knowledge graph.

---

## 1. [[BrainGenome]]
- **Location**: [rust/src/brain/genome.rs](file:///d:/Development/Movement/rust/src/brain/genome.rs#L47)
- **Role**: Polymorphic container representing an individual's nervous system.
- **Variants**:
  - `Perceptron`: Single-layer sensory-motor weight matrix.
  - `Mlp`: Multi-layer feedforward network with fixed layer sizes.
  - `SpatialGraph`: Graph where nodes possess spatial 2D coordinates in sensory space.
  - `GooGraph`: Dynamic directed graph of computational primitives.
- **Key Operations**: `.forward()`, `.reproduce_brain()`, `.genetic_distance()`, `.complexity()`, `.structural_budget()`.
- **Fact Status**: `EXTRACTED`

## 2. [[Config]]
- **Location**: [rust/src/config.rs](file:///d:/Development/Movement/rust/src/config.rs)
- **Role**: Complete parameter set for environmental dynamics and evolutionary selection.
- **Core Parameters**:
  - `burn_in`: Initial stabilization steps.
  - `carrying_capacity`: Maximum population threshold triggering competitive culling.
  - `env_perlin_spatial_scale` & `env_perlin_time_scale`: Rate of environmental drift.
  - `movement_cost_factor` & `reproduction_threshold`: Energy economics.
- **Fact Status**: `EXTRACTED`

## 3. [[ArchitecturePolicy]]
- **Location**: [rust/src/config.rs](file:///d:/Development/Movement/rust/src/config.rs)
- **Role**: Specifies the constraint profile for brain evolution in a run.
- **Policies**:
  - `PerceptronOnly`: Disallows structural expansion.
  - `EvolveMlp`: Allows adding hidden layers or neurons.
  - `EvolveSpatialGraph`: Permits edge and node addition in spatial coordinates.
  - `EvolveGoo`: Full topological mutation of Goo operators.
- **Fact Status**: `EXTRACTED`

## 4. [[Simulation]]
- **Location**: [rust/src/simulation.rs](file:///d:/Development/Movement/rust/src/simulation.rs)
- **Role**: Primary world state orchestrator.
- **Responsibilities**:
  - Maintains environment resource grids and population vectors.
  - Calls `step_engine` to resolve simultaneous agent sensory updates.
  - Manages reproduction, birth logging into [[Cladogram]], and culling.
- **Fact Status**: `EXTRACTED`

## 5. [[PetriDish]]
- **Location**: [rust/src/dish.rs](file:///d:/Development/Movement/rust/src/dish.rs)
- **Role**: Spatial boundary abstraction.
- **Modes**:
  - Open dish with edge collision/bouncing.
  - Toroidal wrap across boundary hexes.
  - Multi-dish configurations for parallel allopatric speciation experiments.
- **Fact Status**: `EXTRACTED`

## 6. [[GooGraphBrain]]
- **Location**: [rust/src/brain/goo_graph.rs](file:///d:/Development/Movement/rust/src/brain/goo_graph.rs)
- **Role**: High-capacity non-standard neural network architecture.
- **Primitives**:
  - `Generate`: Node creating candidate actions/values.
  - `Score`: Evaluator nodes calculating heuristic utility.
  - `KeepBest`: Selection node filtering highest-scoring candidate.
  - `Aggregate`: Blends multiple sensory inputs.
- **Fact Status**: `EXTRACTED`

## 7. [[HexCoord]] & [[HexGrid]]
- **Location**: [rust/src/hex_coord.rs](file:///d:/Development/Movement/rust/src/hex_coord.rs), [rust/src/hex_grid.rs](file:///d:/Development/Movement/rust/src/hex_grid.rs)
- **Role**: Axial coordinate system `(q, r)` and flat-top hexagon spatial container.
- **Operations**: Neighbor resolution (6 directions), Manhattan distance, wrapping arithmetic.
- **Fact Status**: `EXTRACTED`

## 8. [[Cladogram]]
- **Location**: [rust/src/cladogram.rs](file:///d:/Development/Movement/rust/src/cladogram.rs)
- **Role**: Phylogenetic tracking structure recording lineage trees, organism ancestry, and death times.
- **Fact Status**: `EXTRACTED`
"""

# 4. Communities.md
communities_content = """---
project: Movement
type: graphify-note
source: graphify
updated: 2026-10-22
---

# Communities: Movement

> High-level clustering decomposition across 106 detected functional communities in the codebase.

---

## Major Functional Clusters

### 1. Visualization & Diagramming (Community 0, 1, 25, 41)
- **Nodes**: `BrainDiagram`, `diagram_from_brain()`, `draw_brain_panel()`, `draw_nn_diagram()`, `on_slider()`, `on_text_submit()`
- **Purpose**: Rendering neural wiring, activation diagrams, interactive egui panels, and webviewer animations.

### 2. Neural Brain Substrates & Mutations (Community 26, 28, 45, 48)
- **Nodes**: [[BrainGenome]], `BrainInner`, `StructuralMutation`, [[ArchitecturePolicy]], `apply_goo_structural()`, `apply_mlp_structural()`, `init_founder_weights()`
- **Purpose**: Genotype definitions, forward pass evaluation, parametric weight mutations, and topological expansions.

### 3. Hexagonal Environment & Physics (Community 20, 22, 43, 51)
- **Nodes**: [[HexCoord]], [[HexGrid]], `apply_decision()`, `step_individuals_cpu()`, `neighbor_flat()`, `Environment`
- **Purpose**: 6-connected neighbor lookups, environmental resource distribution, organism movement execution, and collision resolution.

### 4. GPU & CUDA Acceleration (Community 12, 19)
- **Nodes**: `cached_cuda_module()`, `CachedCudaRuntime`, `DecisionRowGpu`, `brain_step_kernel()`, `build_sensors()`
- **Purpose**: High-throughput parallel sensor computation and brain forward evaluations on NVIDIA hardware.

### 5. Evolution Pipelines & Landscape Sweeps (Community 3, 7, 27, 34, 38)
- **Nodes**: `aggregate_cells()`, `build_cells_to_sweep()`, `select_promoted_cells()`, `evo_fitness_landscape`, `write_stage_report()`
- **Purpose**: Large-scale parameter grid searches, automated fitness evaluation, and multi-stage evolutionary runs.

### 6. Phylogeny & Lineage Tracking (Community 42, 56, 73)
- **Nodes**: [[Cladogram]], `LineageNode`, `apply_metabolism_and_cull()`, `run_reproduction_phase()`, `assign_genotype_colors()`
- **Purpose**: Tracking evolutionary descent trees, recording extinction times, and calculating genetic divergence.

---

## Cohesion & Structure Overview
- Total Communities: **106**
- Highly Cohesive Modules (> 0.40):
  - **Community 75** (Cohesion 0.60): Test harnesses for edge conditions.
  - **Community 73** (Cohesion 0.47): Genotype signature & color derivation.
  - **Community 65** (Cohesion 0.44): Snapshot & run replay discovery.
- Low Cohesion Modules (< 0.10):
  - Broad utility and visualization clusters combining loose interactive helper routines.
"""

# 5. Open Questions.md
questions_content = """---
project: Movement
type: graphify-note
source: graphify
updated: 2026-10-22
---

# Open Questions & Architectural Opportunities: Movement

> Structural risks, bottlenecks, and candidate refactorings identified by Graphify analysis.

---

## 1. High-Centrality Bottlenecks
Three nodes exhibit disproportionately high betweenness centrality:
1. **[[Simulation]]** (Centrality: 0.141): Bridges simulation mechanics, CUDA execution, telemetry logging, and dish setup.
   - *Risk*: High coupling makes independent refactoring of simulation physics difficult.
   - *Opportunity*: Extract distinct lifecycle stages into dedicated subsystems: `SensoryPipeline`, `MetabolismEngine`, `ReproductionManager`.
2. **[[BrainGenome]]** (Centrality: 0.064): Connects over 15 distinct functional communities.
   - *Status*: Cleanly decoupled via `rust/src/brain/types.rs`.
3. **[[ArchitecturePolicy]]** (Centrality: 0.054): Dispatched across configuration and evaluation pipelines.

---

## 2. Cyclic Dependencies Status
- **Resolved**: All 13 multi-file cycles between `genome.rs`, concrete brain variants, and `hex_simulation.rs` / `step_engine.rs` have been completely eliminated.
- **Current State**: 0 multi-file import cycles across the entire codebase.

---

## 3. Isolated & Weakly Connected Nodes
- **148 isolated symbol nodes** (with ≤1 edge):
  - E.g. Standalone enum constructors or utility functions in `scripts/`.
- *Action Item*: Check whether these isolated symbols reflect dead code, untyped dynamic dispatch in Python scripts, or unmodeled macro expansions.

---

## 4. Multi-Language Extractor Support
- **R Scripts**: Successfully extracted with `tree-sitter-language-pack` installed via `graphifyy[r]`.
- **WGSL Shaders**: `resource_update.wgsl` remains unclassified by tree-sitter.
"""

(VAULT_DIR / "Graphify Overview.md").write_text(overview_content, encoding="utf-8")
(VAULT_DIR / "Architecture Map.md").write_text(arch_content, encoding="utf-8")
(VAULT_DIR / "Key Concepts.md").write_text(concepts_content, encoding="utf-8")
(VAULT_DIR / "Communities.md").write_text(communities_content, encoding="utf-8")
(VAULT_DIR / "Open Questions.md").write_text(questions_content, encoding="utf-8")

print("Successfully written 5 core Obsidian notes to", VAULT_DIR)

# Copy and format wiki notes
src_wiki = Path(r"graphify-out/wiki")
count = 0
if src_wiki.exists():
    for f in src_wiki.glob("*.md"):
        content = f.read_text(encoding="utf-8")
        if not content.startswith("---"):
            header = "---\nproject: Movement\ntype: graphify-note\nsource: graphify\nupdated: 2026-10-22\n---\n\n"
            content = header + content
        content = re.sub(r"\[(.*?)\]\((.*?)\.md\)", r"[[\2|\1]]", content)
        content = content.replace("[[index|index]]", "[[Graphify Overview]]").replace("[[index]]", "[[Graphify Overview]]")
        
        out_file = WIKI_DIR / f.name
        out_file.write_text(content, encoding="utf-8")
        count += 1

print(f"Successfully formatted and synced {count} wiki notes to {WIKI_DIR}")
