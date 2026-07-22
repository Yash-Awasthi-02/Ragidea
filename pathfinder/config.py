"""
PATHFINDER — Hyperparameters from the paper (Sections 3.3, 3.4, 4.2, 4.3, 7.3).

All defaults match the paper exactly. Override via constructor arguments.
"""

# ── Facet weights (Definition 4, Section 3.4) ─────────────────────────────
ALPHA   = 0.50   # φ_sem  — semantic coverage weight
BETA    = 0.15   # φ_temp — temporal recency weight
GAMMA   = 0.15   # φ_imp  — structural importance weight
DELTA   = 0.10   # φ_dom  — domain alignment weight
EPSILON = 0.10   # φ_conf — epistemic confidence weight

# ── Edge construction (Section 7.2) ───────────────────────────────────────
THETA_EDGE  = 0.30   # semantic edge admission threshold
W_ENTITY    = 0.70   # entity co-mention edge weight

# ── Temporal facet (Section 3.3) ──────────────────────────────────────────
LAMBDA_TEMP = 0.05   # decay rate for φ_temp = exp(-λ · age_days)

# ── Domain facet (Section 3.3, 7.3) ───────────────────────────────────────
PCA_K       = 16     # domain embedding dimensionality (paper uses K=16 in §7.3)

# ── Importance facet (Section 7.3) ────────────────────────────────────────
PR_DAMPING  = 0.85   # PageRank damping factor
PR_MAX_ITER = 100    # PageRank max iterations

# ── Confidence facet (Section 7.3) ────────────────────────────────────────
PHI_CONF_INIT = 0.95  # initial epistemic confidence (0.70 caused path-product σ to
                       # collapse: 0.70^5 × 0.50^4 ≈ 0.01, triggering re-traversal on
                       # every query and limiting mean node selection to 2.23/query vs
                       # 7.18 at 0.95 — see experiments/analysis.md)

# ── Algorithm 1 (Section 4.2) ─────────────────────────────────────────────
K_TOK             = 2048    # default token budget
TAU_HIGH          = 0.50    # σ ≥ τ_high → proceed without hedge
TAU_LOW           = 0.30    # σ < τ_low → trigger re-traversal
MAX_RETRIES       = 3       # bounded re-traversal limit
EPSILON_MAX       = 0.40    # max ε during re-traversal escalation
EPSILON_STEP      = 0.05    # ε increment per retry
SUFFICIENCY_THRES = 0.85    # coverage score threshold for sufficiency check

# ── Semantic cache (Section 4.4) ──────────────────────────────────────────
CACHE_THETA = 0.92   # cosine similarity threshold for cache hit

# ── Feedback loop (Section 4.5) ───────────────────────────────────────────
ETA_EDGE    = 0.05   # edge weight learning rate
MU_CONF     = 0.03   # confidence learning rate
ETA_W_DOM   = 0.001  # W_dom learning rate (optional)
