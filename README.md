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
  tests/              47 unit tests (formal properties)

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
  multi_benchmark.md  Results summary

pathfinder-paper.md   Research manuscript
PLAN.md               Master plan & optimization pathway
```

## Key Results (N=500 per dataset)

| System | HotpotQA R@5 | R@10 | 2Wiki R@5 | MuSiQue R@5 |
|---|---|---|---|---|
| PATHFINDER | 0.268 | **0.350** | 0.226 | 0.006 |
| Naive RAG | **0.310** | 0.310 | **0.304** | 0.004 |
| + LLM Rerank | **0.320** | — | — | — |

See `PLAN.md` for the full optimization pathway and `pathfinder-paper.md` for the complete research paper.
