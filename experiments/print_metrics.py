"""
PATHFINDER — Print Consolidated Metrics from All Benchmark Evaluations
======================================================================
Reads results/*_eval*.json files and prints a unified metrics table.

Usage:
    python experiments/print_metrics.py
"""
import json, os, sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Dataset configs: (display_name, [possible filenames])
DATASETS = [
    ("HotpotQA",      ["hotpotqa_eval_full.json", "hotpotqa_eval_passage.json", "hotpotqa_eval.json"]),
    ("2WikiMultihopQA", ["2wiki_eval_full.json", "2wiki_eval.json"]),
    ("MuSiQue",       ["musique_eval_full.json", "musique_eval.json"]),
]


def load_eval(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_dataset(name, data, label=""):
    s = data["summary"]
    n = s["n_queries"]
    print(f"\n📊 --- {name} (N={n}){label} ---")

    # Recall@k for all systems
    systems = [
        ("PATHFINDER",          "pathfinder"),
        ("Naive RAG",           "naive_rag"),
        ("Spreading Activation","spreading_activation"),
        ("BFS 2-Hop",           "bfs_2hop"),
    ]

    for sys_name, sys_key in systems:
        r5  = s["recall@5"].get(sys_key, 0)
        r10 = s["recall@10"].get(sys_key, 0) if "recall@10" in s else 0
        r20 = s["recall@20"].get(sys_key, 0) if "recall@20" in s else 0
        print(f"  • {sys_name:<20}: R@5 = {r5:.4f}  | R@10 = {r10:.4f}  | R@20 = {r20:.4f}")

    # Paragraph-Recall@5
    if "paragraph_recall@5" in s:
        pr5 = s["paragraph_recall@5"]
        print(f"  • Paragraph-Recall@5  : PF = {pr5.get('pathfinder',0):.4f}  | RAG = {pr5.get('naive_rag',0):.4f}  | SA = {pr5.get('spreading_activation',0):.4f}  | BFS = {pr5.get('bfs_2hop',0):.4f}")

    # Fractional Recall
    if "recall_frac@5" in s:
        print(f"  • Fractional Recall   : R@5 = {s['recall_frac@5']:.4f}  | R@10 = {s.get('recall_frac@10',0):.4f}  | R@20 = {s.get('recall_frac@20',0):.4f}")

    # EM/F1
    if "em_f1" in s:
        em = s["em_f1"].get("em", 0)
        f1 = s["em_f1"].get("f1", 0)
        if em > 0 or f1 > 0:
            print(f"  • EM/F1               : EM = {em:.4f}  | F1 = {f1:.4f}")


def main():
    print("=" * 80)
    print(f"{'':>40}ALL BENCHMARK EVALUATION METRICS")
    print("=" * 80)

    for ds_name, filenames in DATASETS:
        data = None
        label = ""
        for fname in filenames:
            data = load_eval(fname)
            if data:
                if "passage" in fname:
                    label = " [PASSAGE-LEVEL]"
                elif "full" in fname:
                    label = " [FULL-SCALE]"
                else:
                    label = " [N=500]"
                break
        if data:
            print_dataset(ds_name, data, label)
        else:
            print(f"\n📊 --- {ds_name} --- NOT FOUND")

    # Also check for passage-level eval
    passage_data = load_eval("hotpotqa_eval_passage.json")
    if passage_data:
        print_dataset("HotpotQA (Passage-Level)", passage_data, " [PASSAGE]")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
