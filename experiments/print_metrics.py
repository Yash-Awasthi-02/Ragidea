import json
from pathlib import Path

def main():
    files = [
        ("HotpotQA", "results/hotpotqa_eval.json"),
        ("2WikiMultihopQA", "results/2wiki_eval.json"),
        ("MuSiQue", "results/musique_eval.json")
    ]
    
    print("=" * 80)
    print("                      ALL BENCHMARK EVALUATION METRICS                  ")
    print("=" * 80)
    
    for name, filepath in files:
        p = Path(filepath)
        if not p.exists():
            continue
        with open(p, "r") as f:
            summary = json.load(f)["summary"]
            
        print(f"\n📊 --- {name} (N={summary['n_queries']}) ---")
        print(f"  • PATHFINDER          : R@5 = {summary['recall@5']['pathfinder']:.4f}  | R@10 = {summary['recall@10']['pathfinder']:.4f}  | R@20 = {summary['recall@20']['pathfinder']:.4f}")
        print(f"  • Naive RAG           : R@5 = {summary['recall@5']['naive_rag']:.4f}  | R@10 = {summary['recall@10']['naive_rag']:.4f}  | R@20 = {summary['recall@20']['naive_rag']:.4f}")
        print(f"  • Spreading Activation: R@5 = {summary['recall@5']['spreading_activation']:.4f}  | R@10 = {summary['recall@10']['spreading_activation']:.4f}  | R@20 = {summary['recall@20']['spreading_activation']:.4f}")
        print(f"  • BFS 2-Hop           : R@5 = {summary['recall@5']['bfs_2hop']:.4f}  | R@10 = {summary['recall@10']['bfs_2hop']:.4f}  | R@20 = {summary['recall@20']['bfs_2hop']:.4f}")
        print(f"  • Paragraph-Recall@5  : PF = {summary['paragraph_recall@5']['pathfinder']:.4f}  | RAG = {summary['paragraph_recall@5']['naive_rag']:.4f}  | SA = {summary['paragraph_recall@5']['spreading_activation']:.4f}  | BFS = {summary['paragraph_recall@5']['bfs_2hop']:.4f}")
        print(f"  • Fractional Recall   : R@5 = {summary['recall_frac@5']:.4f}  | R@10 = {summary['recall_frac@10']:.4f}  | R@20 = {summary['recall_frac@20']:.4f}")

    print("=" * 80)

if __name__ == "__main__":
    main()
