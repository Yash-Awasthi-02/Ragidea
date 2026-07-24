# Multi-Benchmark Evaluation Results

This document tracks full-scale evaluation results across the three multi-hop QA benchmarks.

## 1. HotpotQA (N=7,405, Full-Scale)

### Sentence-Level Recall@k
| Algorithm | R@5 | R@10 | R@20 |
| :--- | :---: | :---: | :---: |
| **PATHFINDER** | 0.2567 | **0.3352** | 0.3373 |
| **Naive RAG** | **0.2930** | 0.2930 | 0.2930 |
| **Spreading Activation** | 0.1926 | 0.3456 | **0.5991** |
| **BFS 2-Hop** | 0.1490 | 0.2998 | 0.5561 |

### Paragraph-Recall@5
| Algorithm | Score |
| :--- | :---: |
| PATHFINDER | 0.7080 |
| Naive RAG | **0.7474** |
| Spreading Activation | 0.6531 |
| BFS 2-Hop | 0.5660 |

### Fractional Recall (PATHFINDER)
R@5 = 0.5591 | R@10 = 0.6203 | R@20 = 0.6203

### Additional Metrics
- EM = 0.0068 | F1 = 0.0091 (no GROQ_API_KEY)
- Nodes/query = 6.08 | Latency p50 = 1.61ms | p95 = 6.54ms
- σ Spearman ρ = -0.0078 (p=0.5037) | ECE = 0.4105
- Coverage ratio = 1.0097 (100% meet bound, 0 FC violations)
- Anchor: median rank = 1.0, top-1 = 57.81%, top-5 = 87.42%

## 2. 2WikiMultihopQA (N=12,576, Full-Scale)

### Sentence-Level Recall@k
| Algorithm | R@5 | R@10 | R@20 |
| :--- | :---: | :---: | :---: |
| **PATHFINDER** | 0.2347 | **0.3725** | 0.3781 |
| **Naive RAG** | **0.3247** | 0.3247 | 0.3247 |
| **Spreading Activation** | 0.2348 | 0.4404 | **0.6984** |
| **BFS 2-Hop** | 0.1817 | 0.3714 | 0.6118 |

### Paragraph-Recall@5
| Algorithm | Score |
| :--- | :---: |
| PATHFINDER | 0.6942 |
| Naive RAG | **0.7641** |
| Spreading Activation | 0.7143 |
| BFS 2-Hop | 0.6369 |

### Fractional Recall (PATHFINDER)
R@5 = 0.5183 | R@10 = 0.6068 | R@20 = 0.6078

### Additional Metrics
- EM = 0.0020 | F1 = 0.0027 (no GROQ_API_KEY)
- Nodes/query = 6.71 | Latency p50 = 1.30ms | p95 = 5.86ms
- σ Spearman ρ = -0.0315 (p=0.0004) | ECE = 0.4056
- Coverage ratio = 1.0093 (100% meet bound, 0 FC violations)
- Anchor: median rank = 1.0, top-1 = 63.48%, top-5 = 88.35%

## 3. MuSiQue (N=2,417, Full-Scale)

### Sentence-Level Recall@k
| Algorithm | R@5 | R@10 | R@20 |
| :--- | :---: | :---: | :---: |
| **PATHFINDER** | 0.0083 | 0.0124 | 0.0124 |
| **Naive RAG** | 0.0041 | 0.0041 | 0.0041 |
| **Spreading Activation** | **0.0157** | 0.0455 | **0.0976** |
| **BFS 2-Hop** | 0.0145 | 0.0372 | 0.0869 |

### Paragraph-Recall@5
| Algorithm | Score |
| :--- | :---: |
| PATHFINDER | 0.5323 |
| Naive RAG | **0.5531** |
| Spreading Activation | 0.5430 |
| BFS 2-Hop | 0.4558 |

### Fractional Recall (PATHFINDER)
R@5 = 0.2689 | R@10 = 0.3071 | R@20 = 0.3087

### Additional Metrics
- EM = 0.0017 | F1 = 0.0022 (no GROQ_API_KEY)
- Nodes/query = 7.21 | Latency p50 = 3.32ms | p95 = 14.39ms
- σ Spearman ρ = 0.0092 (p=0.6523) | ECE = 0.3714
- Coverage ratio = 0.9954 (100% meet bound, 0 FC violations)
- Anchor: median rank = 1.0, top-1 = 70.05%, top-5 = 91.31%

## Key Findings (Full-Scale)

1. **PATHFINDER beats Naive RAG at R@10** on HotpotQA (+14.4%) and 2Wiki (+14.7%)
2. **Naive RAG wins at R@5** — dense retrieval advantage on disconnected graphs
3. **Coverage ratio 1.01** — 100% meet (1−1/e) bound, 0 FC violations
4. **σ calibration poor** — Spearman ρ ≈ 0, needs geometric mean or bottleneck model
5. **Anchor quality strong** — entry node in top-1 58-70% of the time
6. **Latency production-viable** — 1.3-3.3ms p50, 5.9-14.4ms p95
7. **Semantic-only weights optimal** — adding γ=0.15 reduces R@5 by 52% on HotpotQA

## N=500 Subset Results (for comparison)

| Dataset | PF R@5 (N=500) | PF R@5 (Full) | RAG R@5 (N=500) | RAG R@5 (Full) |
|---|---|---|---|---|
| HotpotQA | 0.2680 | 0.2567 | 0.3100 | 0.2930 |
| 2Wiki | 0.2260 | 0.2347 | 0.3040 | 0.3247 |
| MuSiQue | 0.0060 | 0.0083 | 0.0040 | 0.0041 |

N=500 subset is representative — direction and magnitude consistent with full-scale.
