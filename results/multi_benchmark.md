# Multi-Benchmark Evaluation Results

This document tracks the R@5 (Recall at 5) baseline evaluation across the three multi-hop question answering datasets.

## 1. HotpotQA (7,405 validation queries)
- **PATHFINDER (Semantic-Only)**: 0.7307
- **Naive RAG**: 0.7937
- **Spreading Activation**: 0.6974
- **BFS 2-Hop**: 0.6124

## 2. 2WikiMultihopQA (12,576 validation queries)
- **PATHFINDER (Semantic-Only)**: 0.2331
- **Naive RAG**: 0.3248
- **Spreading Activation**: 0.2358
- **BFS 2-Hop**: 0.1820

## 3. MuSiQue (2,417 validation queries)
- **PATHFINDER (Semantic-Only)**: 0.0087
- **Naive RAG**: 0.0041
- **Spreading Activation**: 0.0165
- **BFS 2-Hop**: 0.0141

### Observations & Next Steps
- **HotpotQA & 2WikiMultihopQA**: The trend holds. Naive RAG (dense-only) slightly outperforms the pure structural graph approaches (BFS) and PATHFINDER sits competitively in the middle. We will need the dynamic semantic-structural hybridization (Phase 2) to push PATHFINDER past Naive RAG.
- **MuSiQue**: The scores are near zero across the board. This is because MuSiQue requires up to 4 hops, and our ground truth mapping matched *every sentence* in the supporting paragraphs. Since `k=5`, if a query has more than 5 gold sentences, R@5 will strictly be 0. We will need to revise the evaluation metric for MuSiQue (e.g., Paragraph-level Recall, or R@10 / R@20) to get meaningful signal.
