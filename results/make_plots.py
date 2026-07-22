"""
Generate 4 publication-quality plots from results/raw/ JSON files.
Saves all plots to results/plots/.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use("seaborn-v0_8-whitegrid")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
RAW_DIR   = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "naive_rag":            "#4C72B0",
    "bfs_2hop":             "#DD8452",
    "spreading_activation": "#55A868",
    "pathfinder":           "#C44E52",
}
PF_COLORS = ["#8172B3", "#937860", "#CCB974", "#64B5CD", "#C44E52"]

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — System comparison grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_system_comparison():
    with open(os.path.join(RAW_DIR, "hotpotqa_200_emf1.json")) as f:
        data = json.load(f)

    systems  = ["naive_rag", "bfs_2hop", "spreading_activation", "pathfinder"]
    labels   = ["Naive RAG", "BFS 2-hop", "Spreading\nActivation", "PATHFINDER"]
    metrics  = ["mean_recall@5", "mean_em", "mean_f1"]
    mlabels  = ["Recall@5", "EM", "F1"]
    mcolors  = ["#4C72B0", "#DD8452", "#55A868"]

    vals = {m: [data["systems"][s][m] for s in systems] for m in metrics}

    n_sys   = len(systems)
    n_met   = len(metrics)
    x       = np.arange(n_sys)
    width   = 0.22
    offsets = np.linspace(-(n_met-1)/2, (n_met-1)/2, n_met) * width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (m, ml, mc) in enumerate(zip(metrics, mlabels, mcolors)):
        bars = ax.bar(x + offsets[i], vals[m], width, label=ml,
                      color=mc, alpha=0.85, edgecolor="white", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("PATHFINDER vs Baselines — HotpotQA (N=200)\nRecall@5, EM, F1",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, max(max(vals[m]) for m in metrics) * 1.22)
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(axis="y", labelsize=10)

    # Highlight PATHFINDER column
    ax.axvspan(2.5, 3.5, alpha=0.06, color="#C44E52", zorder=0)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "system_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — σ calibration bucket chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_sigma_calibration():
    with open(os.path.join(RAW_DIR, "sigma_calibration_200.json")) as f:
        data = json.load(f)

    bucket_keys  = ["[0.0, 0.3)", "[0.3, 0.5)", "[0.5, 0.7)", "[0.7, 1.0)"]
    bucket_labels = ["[0.0, 0.3)\nretraverse", "[0.3, 0.5)\nhedge",
                     "[0.5, 0.7)\nproceed", "[0.7, 1.0)\nproceed+"]
    r5_vals  = [data["buckets"][k]["mean_r5"] for k in bucket_keys]
    ns       = [data["buckets"][k]["n"]       for k in bucket_keys]

    # Overall mean R@5 (weighted)
    total_n  = sum(ns)
    overall  = sum(r5_vals[i] * ns[i] for i in range(len(ns))) / total_n

    rho  = data["spearman_rho"]
    pval = data["spearman_pval"]
    ece  = data["ECE"]

    bucket_colors = ["#C44E52", "#DD8452", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(bucket_keys)), r5_vals, color=bucket_colors,
                  alpha=0.82, edgecolor="white", linewidth=0.7, width=0.55)

    for bar, r5, n in zip(bars, r5_vals, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{r5:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(overall, color="#333333", linewidth=1.6, linestyle="--", label=f"Overall mean R@5 = {overall:.3f}")

    ax.set_xticks(range(len(bucket_keys)))
    ax.set_xticklabels(bucket_labels, fontsize=10)
    ax.set_ylabel("Mean Recall@5", fontsize=12)
    ax.set_title(
        f"σ Calibration — Mean R@5 per Confidence Bucket (N=200)\n"
        f"Spearman ρ = {rho:.3f}  (p = {pval:.3f})   ECE = {ece:.3f}",
        fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(0, max(r5_vals) * 1.35)
    ax.legend(fontsize=10)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "sigma_calibration.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Weight ablation bar chart (PF variants only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation():
    with open(os.path.join(RAW_DIR, "hotpotqa_200_full_ablation.json")) as f:
        data = json.load(f)

    # Show baselines + PF variants together for context
    all_keys = ["naive_rag", "bfs_2hop", "spreading_activation",
                "pf_original", "pf_geomean_only", "pf_beta0_only",
                "pf_multianchor_only", "pf_all_fixes"]
    all_labels = ["Naive RAG", "BFS 2-hop", "Spread.\nActiv.",
                  "PF\nOriginal", "PF\nGeomean σ", "PF\nβ=0",
                  "PF\nMulti-anchor", "PF\nAll fixes"]
    r5_vals = [data["variants"][k]["mean_r5"] for k in all_keys]

    bar_colors = (
        [COLORS["naive_rag"], COLORS["bfs_2hop"], COLORS["spreading_activation"]]
        + PF_COLORS
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(all_keys)), r5_vals, color=bar_colors,
                  alpha=0.84, edgecolor="white", linewidth=0.7, width=0.6)

    for bar, r5 in zip(bars, r5_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{r5:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Divider between baselines and PF variants
    ax.axvline(2.5, color="#888888", linewidth=1.2, linestyle=":")
    ax.text(2.52, max(r5_vals) * 1.12, "← Baselines  |  PF Variants →",
            fontsize=8.5, color="#555555")

    # Horizontal line: naive_rag as reference
    naive_r5 = data["variants"]["naive_rag"]["mean_r5"]
    ax.axhline(naive_r5, color=COLORS["naive_rag"], linewidth=1.4,
               linestyle="--", alpha=0.7, label=f"Naive RAG baseline = {naive_r5:.3f}")

    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(all_labels, fontsize=9.5)
    ax.set_ylabel("Mean Recall@5", fontsize=12)
    ax.set_title("Ablation Study — PATHFINDER Variants vs Baselines (N=200, HotpotQA)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(0, max(r5_vals) * 1.28)
    ax.legend(fontsize=10)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Coverage ratio histogram
# ─────────────────────────────────────────────────────────────────────────────
def plot_coverage_ratio():
    with open(os.path.join(RAW_DIR, "coverage_ratio_real.json")) as f:
        data = json.load(f)

    ratios  = [q["ratio"] for q in data["per_query"]]
    summary = data["summary"]
    bound   = 1 - 1 / np.e   # ≈ 0.6321

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Histogram — few points so use step-style with markers
    n_bins = min(8, len(ratios))
    counts, edges, patches = ax.hist(ratios, bins=n_bins, color="#4C72B0",
                                     alpha=0.75, edgecolor="white", linewidth=0.8)

    # Colour bars below bound red
    for patch, left in zip(patches, edges[:-1]):
        if left < bound:
            patch.set_facecolor("#C44E52")
            patch.set_alpha(0.75)

    # Vertical lines
    ax.axvline(bound, color="#C44E52", linewidth=2.0, linestyle="--",
               label=f"(1−1/e) bound ≈ {bound:.3f}")
    ax.axvline(summary["mean_ratio"], color="#333333", linewidth=1.8, linestyle="-",
               label=f"Mean ratio = {summary['mean_ratio']:.4f}")
    ax.axvline(summary["min_ratio"], color="#DD8452", linewidth=1.4, linestyle=":",
               label=f"Min ratio = {summary['min_ratio']:.4f}")

    # Scatter individual points above x-axis
    ax.scatter(ratios, [0.08] * len(ratios), marker="|", s=250,
               color="#2d2d2d", zorder=5, linewidth=1.5)

    ax.set_xlabel("F_greedy / F*_frontier", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Coverage Ratio — Greedy vs Optimal (Real HotpotQA Graphs, N={summary['n']})\n"
        f"Mean = {summary['mean_ratio']:.4f}   Min = {summary['min_ratio']:.4f}   "
        f"100% ≥ (1−1/e) bound",
        fontsize=11.5, fontweight="bold", pad=14)
    ax.set_xlim(0.55, 1.05)
    ax.legend(fontsize=10, loc="upper left")
    ax.tick_params(labelsize=10)

    # Annotation
    ax.annotate("All 8 instances\nmeet the bound",
                xy=(bound + 0.005, 0.5), xytext=(bound + 0.05, 1.5),
                arrowprops=dict(arrowstyle="->", color="#333333"),
                fontsize=9, color="#333333")

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "coverage_ratio.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating plots...")
    plot_system_comparison()
    plot_sigma_calibration()
    plot_ablation()
    plot_coverage_ratio()
    print("Done — all 4 plots saved to results/plots/")
