# PATHFINDER — Phase 2 & 3 Execution Plan

## Phase 2: Mathematical Refinement & Calibration (IN PROGRESS)
The goal of this phase is to bridge the performance gap between PATHFINDER (pure structural traversal) and Naive RAG (pure dense retrieval) by dynamically hybridizing structural and semantic weights, and introducing "teleportation" jumps for disconnected graph components.

1. **Implement Adaptive Dense Weights (`sigma(q, d)`)**
   - Perform a calibration grid search for the `sigma` parameter.
   - Blend structural decay with semantic query similarity to overcome strict graph sparsity.
2. **Implement Dynamic Dense-Frontier Teleportation**
   - Allow the algorithm to "jump" out of local graph clusters when all adjacent node scores fall below a semantic threshold.
   - This directly addresses the submodularity limitations when ground-truth nodes are disconnected in the graph.

## Phase 3: Paper Manuscript & Benchmark Synchronization
Once the math refinements successfully boost PATHFINDER's R@5 scores past Naive RAG:
1. **Update Manuscript (`pathfinder-paper.md`)**
   - Update Section 7 (Empirical Results) with the finalized numbers from HotpotQA, 2WikiMultihopQA, and MuSiQue.
   - Detail the calibration mechanics.
2. **Update Future Work (`FUTURE_WORK.md`)**
   - Realign the future open problems based on the final benchmarks.
