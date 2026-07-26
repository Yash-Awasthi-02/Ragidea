# PATHFINDER

Submodular Coverage Maximization over Multidimensional Knowledge Graphs for Multi-Hop Retrieval-Augmented Generation (RAG).

## Quick Start

```bash
# Install dependencies
pip install -r experiments/requirements.txt
python -m spacy download en_core_web_sm

# Build knowledge graphs (HotpotQA, 500 samples)
python experiments/01_build_kg.py --max_samples 500 --output data/hotpotqa_graphs.pkl
python experiments/02_load_2wiki_musique.py --dataset 2wiki --max_samples 500 --output data/2wiki_graphs.pkl
python experiments/02_load_2wiki_musique.py --dataset musique --max_samples 500 --output data/musique_graphs.pkl

# Run evaluation (no LLM needed)
python experiments/05_evaluate.py --graphs data/hotpotqa_graphs.pkl --max_samples 500 --output results/hotpotqa_eval.json

# Print consolidated metrics
python experiments/print_metrics.py

# Generate plots
python results/make_plots.py

# Run unit tests
pytest pathfinder/tests/
```

## Project Structure

```
pathfinder/           Core algorithm package
  algorithm.py        PATHFINDER-Greedy (Algorithm 1)
  graph.py            Knowledge graph construction
  config.py           Hyperparameters
  facets.py           Five-facet node representation
  baselines.py        Naive RAG, BFS, Spreading Activation
  feedback.py         Online feedback loop
  query_intel.py      Query classification & routing
  tests/              100 unit tests (formal properties)

experiments/          Evaluation pipeline
  01_build_kg.py      Build KGs from HotpotQA
  02_load_2wiki_musique.py  Load 2Wiki/MuSiQue
  03_grid_search.py   Hyperparameter grid search
  04_confidence_calibration.py  σ model comparison
  05_evaluate.py      Full evaluation (all metrics)
  05b-05d             Teleportation ablation & sensitivity
  07-21               Phase 5-12 experiment scripts
  run_pathfinder.py   Algorithm 1 implementation (experiment runner)
  run_baselines.py    Baseline implementations
  generate_answers.py Groq LLM generation + EM/F1

results/              Evaluation outputs
  raw/                JSON result files
  plots/              Generated PNG plots
  make_plots.py       Plot generation script

pathfinder-paper.md   Research manuscript
PLAN.md               Master plan & optimization pathway
```

## Key Results

Full-scale passage-level retrieval (sentence+passage hybrid graphs, BGE embeddings):

| Benchmark | N (queries) | Recall@5 | (1−1/e) coverage ratio |
|---|---|---|---|
| HotpotQA | 7,405 | 0.667–0.708 | 0.995–1.010 (bound held on 100% of graphs) |
| 2WikiMultihopQA | 12,576 | — | bound held on 100% of graphs |
| MuSiQue | 2,417 | — | bound held on 100% of graphs |

The greedy algorithm is validated by 100 formal-property unit tests (all passing). PATHFINDER is positioned as **retrieval parity + provable coverage certificate + interpretable graph structure**, not recall superiority. See `PLAN.md` for the optimization pathway and `pathfinder-paper.md` for the complete research paper.
