"""
PATHFINDER Experiment — Step 6: σ Calibration Visualisation
=============================================================
Reads results/results.json and produces:
  1. Bucket bar chart  — mean EM per σ bucket (4 buckets)
  2. Calibration curve — σ vs accuracy with diagonal ideal line
  3. Scatter plot      — per-query σ vs EM (jittered for readability)
  4. Three-tier bar    — EM for proceed / hedge / re-traverse tiers

Requires matplotlib. Install: pip install matplotlib

Usage:
    python 06_plot_sigma.py --results results/results.json
    python 06_plot_sigma.py --results results/results.json --save plots/
"""

import json
import sys
import argparse
import random
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Install matplotlib:  pip install matplotlib numpy")
    sys.exit(1)


BUCKET_DEFS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
BUCKET_LABELS = ["[0, 0.3)\nRe-traverse", "[0.3, 0.5)\nHedge",
                 "[0.5, 0.7)\nProceed-low", "[0.7, 1.0]\nProceed-high"]
COLORS = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]


def load_results(path: str) -> tuple[list[float], list[int]]:
    """Load per-query sigma and EM from results JSON."""
    with open(path) as f:
        data = json.load(f)
    pq = data.get("per_query", [])
    sigmas = [r["pathfinder"]["sigma"] for r in pq if "sigma" in r["pathfinder"]]
    ems    = [r["pathfinder"]["em"]    for r in pq if "em"    in r["pathfinder"]]
    return sigmas, ems


def plot_bucket_bars(sigmas, ems, ax, title="Mean EM per σ Bucket"):
    """Bar chart: mean EM per σ bucket with n labels."""
    bucket_ems, bucket_ns = [], []
    for lo, hi in BUCKET_DEFS:
        vals = [e for s, e in zip(sigmas, ems) if lo <= s < hi]
        bucket_ems.append(float(np.mean(vals)) if vals else 0.0)
        bucket_ns.append(len(vals))

    bars = ax.bar(range(4), bucket_ems, color=COLORS, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(BUCKET_LABELS, fontsize=9)
    ax.set_ylabel("Mean EM", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(bucket_ems) * 1.35 + 0.05))
    ax.axhline(float(np.mean(ems)), color="gray", linestyle="--", linewidth=1,
               label=f"Overall mean EM = {float(np.mean(ems)):.3f}")
    ax.legend(fontsize=9)

    for bar, em_val, n in zip(bars, bucket_ems, bucket_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{em_val:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)


def plot_calibration_curve(sigmas, ems, ax, n_bins=10,
                           title="σ Calibration Curve"):
    """Reliability diagram: mean σ vs mean EM per equally-spaced bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_sigma, bin_em, bin_n = [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        vals = [(s, e) for s, e in zip(sigmas, ems) if lo <= s < hi]
        if vals:
            s_vals, e_vals = zip(*vals)
            bin_sigma.append(float(np.mean(s_vals)))
            bin_em.append(float(np.mean(e_vals)))
            bin_n.append(len(vals))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    scatter = ax.scatter(bin_sigma, bin_em, s=[n * 5 for n in bin_n],
                         c=bin_sigma, cmap="RdBu", vmin=0, vmax=1,
                         edgecolors="black", linewidths=0.5, zorder=3)
    ax.plot(bin_sigma, bin_em, "gray", linewidth=1, alpha=0.5)

    ax.set_xlabel("Mean σ (path confidence)", fontsize=11)
    ax.set_ylabel("Mean EM (accuracy)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    # ECE annotation
    ece = sum(
        (n / len(sigmas)) * abs(em - s)
        for s, em, n in zip(bin_sigma, bin_em, bin_n)
    )
    ax.text(0.05, 0.92, f"ECE = {ece:.4f}", transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def plot_scatter(sigmas, ems, ax, title="Per-query σ vs EM"):
    """Scatter: per-query σ vs EM (jittered EM since it's 0/1)."""
    rng = random.Random(42)
    jittered_ems = [e + rng.uniform(-0.03, 0.03) for e in ems]

    correct   = [(s, e) for s, e, em in zip(sigmas, jittered_ems, ems) if em == 1]
    incorrect = [(s, e) for s, e, em in zip(sigmas, jittered_ems, ems) if em == 0]

    if correct:
        ax.scatter(*zip(*correct),   alpha=0.3, s=12, c="#4575b4", label="Correct (EM=1)")
    if incorrect:
        ax.scatter(*zip(*incorrect), alpha=0.3, s=12, c="#d73027", label="Incorrect (EM=0)")

    # Vertical threshold lines
    ax.axvline(0.3, color="orange", linestyle="--", linewidth=1.2, label="τ_low=0.3")
    ax.axvline(0.5, color="green",  linestyle="--", linewidth=1.2, label="τ_high=0.5")

    ax.set_xlabel("σ (path confidence)", fontsize=11)
    ax.set_ylabel("EM (jittered)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)


def plot_three_tier(sigmas, ems, ax, title="Three-Tier Policy Validation"):
    """Bar chart: mean EM for σ<0.3, 0.3≤σ<0.5, σ≥0.5."""
    tiers = {"Re-traverse\n(σ<0.3)": [], "Hedge\n(0.3≤σ<0.5)": [], "Proceed\n(σ≥0.5)": []}
    for s, e in zip(sigmas, ems):
        if s < 0.3:
            tiers["Re-traverse\n(σ<0.3)"].append(e)
        elif s < 0.5:
            tiers["Hedge\n(0.3≤σ<0.5)"].append(e)
        else:
            tiers["Proceed\n(σ≥0.5)"].append(e)

    labels = list(tiers.keys())
    means  = [float(np.mean(v)) if v else 0.0 for v in tiers.values()]
    ns     = [len(v) for v in tiers.values()]
    colors = ["#d73027", "#fc8d59", "#4575b4"]

    bars = ax.bar(labels, means, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Mean EM", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(means) * 1.35 + 0.05))

    for bar, m, n in zip(bars, means, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)

    # Validation check
    if all(n > 0 for n in ns):
        validated = means[2] > means[1] > means[0]
        note = "✓ Three-tier ordering validated" if validated else "✗ Ordering not confirmed"
        color = "green" if validated else "red"
        ax.text(0.5, 0.95, note, transform=ax.transAxes, ha="center",
                fontsize=10, color=color, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Plot σ calibration from evaluation results")
    parser.add_argument("--results", default="results/results.json")
    parser.add_argument("--save",    default=None, help="Directory to save plots (default: show)")
    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Results file not found: {args.results}")
        print("Run 05_evaluate.py first.")
        sys.exit(1)

    print(f"Loading {args.results}...")
    sigmas, ems = load_results(args.results)
    n = len(sigmas)

    if n == 0:
        print("No per-query results found.")
        sys.exit(1)

    print(f"Loaded {n} queries")
    print(f"  Mean σ = {float(np.mean(sigmas)):.4f}")
    print(f"  Mean EM = {float(np.mean(ems)):.4f}")

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PATHFINDER σ Calibration Analysis  (n={n} queries)",
                 fontsize=14, fontweight="bold", y=1.01)

    plot_bucket_bars(sigmas, ems, axes[0, 0])
    plot_calibration_curve(sigmas, ems, axes[0, 1])
    plot_scatter(sigmas, ems, axes[1, 0])
    plot_three_tier(sigmas, ems, axes[1, 1])

    plt.tight_layout()

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "sigma_calibration.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
