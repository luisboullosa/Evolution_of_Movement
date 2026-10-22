# Graphify Workflow & Maintenance Guide

This project maintains a local-first, privacy-preserving knowledge graph in `graphify-out/` and syncs curated notes to the Obsidian vault at `D:\Development\KnowledgeBase\Projects\Movement\Graphify`.

---

## 1. Commands by Scenario

### After Meaningful Code Changes
Update AST extraction incrementally and refresh the wiki:
```bash
graphify . --code-only --update --wiki
```

### After Dependency or Architecture Changes
Re-extract and update the full code graph and wiki:
```bash
graphify . --code-only --update --wiki
```

### After Pulling Git Changes
Quick incremental refresh of the graph:
```bash
graphify update .
```

### Querying the Code Graph
Use targeted graph queries before performing broad file greps:
- **Inspect a Concept**:
  ```bash
  graphify explain "<Concept>"
  # Example: graphify explain "BrainGenome"
  ```
- **Trace Relationships & Call Paths**:
  ```bash
  graphify path "<Node A>" "<Node B>"
  # Example: graphify path "Simulation" "GooGraphBrain"
  ```
- **Ask Architecture Questions**:
  ```bash
  graphify query "<question>"
  # Example: graphify query "Where is reproduction handled?"
  ```

---

## 2. Checklist: Regenerating & Syncing Obsidian Notes

When substantial architectural changes occur, follow this checklist to refresh both the graph and the Obsidian vault:

- [ ] **Step 1: Run Code Extraction**
  ```bash
  graphify . --code-only --no-viz --wiki
  ```
- [ ] **Step 2: Re-cluster Communities (Local, no LLM)**
  ```bash
  graphify cluster-only . --no-label --no-viz
  ```
- [ ] **Step 3: Export Fresh Wiki**
  ```bash
  graphify export wiki
  ```
- [ ] **Step 4: Sync Curated Notes to Obsidian Vault**
  ```bash
  python tools/export_obsidian_notes.py
  ```
- [ ] **Step 5: Verify Vault Notes**
  - Check `D:\Development\KnowledgeBase\Projects\Movement\Graphify\Graphify Overview.md`
  - Open `Architecture Map.md` or `Key Concepts.md` in Obsidian to confirm wikilinks and updated stats.

---

## 3. Privacy & Safety Guarantee
- **Local-only execution**: No source code, embeddings, or prompts are sent to external APIs or third-party services.
- **Ignored paths**: Ignored directories (build artifacts, caches, virtual environments, simulation data dumps, logs) are controlled strictly via `.graphifyignore`.
