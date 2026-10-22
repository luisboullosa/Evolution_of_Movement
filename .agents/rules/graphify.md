---
trigger: always_on
description: Graph-first navigation rules for Movement using graphify and Obsidian knowledge base.
---

## Graphify & Knowledge Graph Guidelines

This repository maintains a local-first code graph in `graphify-out/` and human-readable curated notes in the Obsidian vault at `D:\Development\KnowledgeBase\Projects\Movement\Graphify`.

### Rules for Agent Sessions:
1. **Graph-First Querying**:
   - Prefer `graphify query "<question>"`, `graphify explain "<Node>"`, and `graphify path "<Node A>" "<Node B>"` before initiating broad file searches or grepping across the repository.
   - For fast symbol navigation, query the graph first to get a focused subgraph.
2. **Architecture Orientation**:
   - Consult `graphify-out/GRAPH_REPORT.md` for high-level module architecture, god nodes, and coupling.
   - Consult `D:\Development\KnowledgeBase\Projects\Movement\Graphify/` (specifically `Graphify Overview.md`, `Architecture Map.md`, and `Key Concepts.md`) for human-readable project context and domain explanations.
3. **Graph Freshness & Stale Detection**:
   - Check if `graphify-out/graph.json` exists and verify its timestamp against recent source changes (or check `built_at_commit` vs `git rev-parse HEAD`).
   - If stale, run or offer to run: `graphify . --code-only --update --wiki`.
4. **Verification of Inferred Edges**:
   - Never treat `INFERRED` graph relationships as definitive ground truth without inspecting the cited source location. Verify against the active code before refactoring or removing connections.
5. **Session Maintenance**:
   - After making significant code changes during a session, run `graphify update .` to keep the graph synchronized.
