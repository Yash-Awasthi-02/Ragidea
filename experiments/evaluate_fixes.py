"""
PATHFINDER Fixes Evaluation
============================
Evaluates all three fixes on 200 HotpotQA queries and compares against baselines.
"""
import sys, os, pickle, json, time
sys.path.insert(0, 'experiments')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
from run_pathfinder import run_pathfinder, run_pathfinder_multi_anchor, compute_sigma
from run_baselines import naive_rag

# Load data
with open('data/hotpotqa_graphs_200.pkl', 'rb') as f:
    records = pickle.load(f)

def get_gold_nodes(record):
    sf_titles = record['supporting_facts']['title']
    sf_sent_ids = record['supporting_facts']['sent_id']
    gold_set = set(zip(sf_titles, sf_sent_ids))
    return [i for i, node in enumerate(record['graph']['nodes'])
            if (node['doc_title'], node['sent_idx']) in gold_set]

def recall_at_k(retrieved, gold, k=5):
    if not gold:
        return 1
    return int(all(g in set(retrieved[:k]) for g in gold))

# Test all variants
variants = {
    'naive_rag': [],
    'pathfinder_full_fix': [],
}

pf_details = {
    'n_nodes': [],
    'sigmas': [],
    'flags': [],
    'Fs': [],
}

print("Running evaluation on 200 queries...")
t0 = time.time()

for idx, rec in enumerate(records):
    gd = rec['graph']
    gold = get_gold_nodes(rec)

    # Naive RAG baseline
    nr = naive_rag(gd, k=5)
    variants['naive_rag'].append(recall_at_k(nr, gold, k=5))

    # PATHFINDER with all three fixes
    pf = run_pathfinder(gd, k_tok=500)
    variants['pathfinder_full_fix'].append(recall_at_k(pf.S, gold, k=5))
    pf_details['n_nodes'].append(len(pf.S))
    pf_details['sigmas'].append(pf.sigma)
    pf_details['flags'].append(pf.confidence_flag)
    pf_details['Fs'].append(pf.F)

    if (idx + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  [{idx+1}/200] elapsed={elapsed:.1f}s")

elapsed = time.time() - t0

# Compute results
print()
print("=== RESULTS (N=200, HotpotQA) ===")
for name, r5s in variants.items():
    print(f"  {name:30s}  R@5={np.mean(r5s):.3f} +/- {np.std(r5s):.3f}")

# Breakdown by confidence flag
flags_arr = np.array(pf_details['flags'])
sigmas_arr = np.array(pf_details['sigmas'])
r5_arr = np.array(variants['pathfinder_full_fix'])

print()
print("=== CONFIDENCE BREAKDOWN ===")
for flag in ['HIGH', 'HEDGE', 'LOW']:
    mask = flags_arr == flag
    n = int(mask.sum())
    if n > 0:
        print(f"  {flag:10s}: n={n:4d}  R@5={r5_arr[mask].mean():.3f}  mean_sig={sigmas_arr[mask].mean():.4f}")

print()
print("=== NODE COUNTS ===")
mean_nodes = float(np.mean(pf_details['n_nodes']))
median_nodes = float(np.median(pf_details['n_nodes']))
print(f"  Mean nodes/query: {mean_nodes:.1f}")
print(f"  Median nodes/query: {median_nodes:.1f}")

# Early termination analysis (sigma < 0.3 after geometric mean)
sigma_low = sum(1 for s in pf_details['sigmas'] if s < 0.30)
sigma_low_pct = sigma_low / 200
print()
print("=== SIGMA LOW TERMINATION ===")
print(f"  sigma < 0.30: {sigma_low}/{200} ({sigma_low_pct:.1%})")

# Calibration
from scipy.stats import spearmanr
rho, pval = spearmanr(pf_details['sigmas'], variants['pathfinder_full_fix'])
print()
print("=== SIGMA CALIBRATION ===")
print(f"  Spearman rho(sig, R@5) = {rho:.4f} (p={pval:.4f})")

print()
print("=== TIMING ===")
print(f"  Total: {elapsed:.1f}s for 200 queries")

# Save results
output = {
    'n_queries': 200,
    'elapsed_s': round(elapsed, 1),
    'variants': {name: {'mean_r5': float(np.mean(r5s)), 'std_r5': float(np.std(r5s))}
                 for name, r5s in variants.items()},
    'pf_details': {
        'mean_nodes': mean_nodes,
        'median_nodes': median_nodes,
        'sigma_low_pct': sigma_low_pct,
        'mean_sigma': float(np.mean(pf_details['sigmas'])),
        'spearman_rho': float(rho),
        'spearman_pval': float(pval),
    },
    'confidence_breakdown': {
        flag: {
            'n': int((flags_arr == flag).sum()),
            'mean_r5': float(r5_arr[flags_arr == flag].mean()) if (flags_arr == flag).sum() > 0 else None,
            'mean_sigma': float(sigmas_arr[flags_arr == flag].mean()) if (flags_arr == flag).sum() > 0 else None,
        }
        for flag in ['HIGH', 'HEDGE', 'LOW']
    },
}

with open('results/raw/hotpotqa_200_fixes.json', 'w') as f:
    json.dump(output, f, indent=2)

print()
print("Saved to results/raw/hotpotqa_200_fixes.json")
