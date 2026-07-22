"""
Unit tests for PATHFINDER formal properties (Section 4.1, Theorem 1, Theorem 2).

Tests:
  1. F(∅, q) = 0  (Corollary to Theorem 1)
  2. Monotonicity: F(S) ≤ F(T) for S ⊆ T  (Lemma 1)
  3. Submodularity: Δ(v|S) ≥ Δ(v|T) for S ⊆ T  (Theorem 1)
  4. (1−1/e) approximation bound on small synthetic graphs  (Theorem 2)
  5. σ(S) computation correctness  (Section 4.3)
  6. Algorithm 1 guard conditions  (lines 0a, 0b, 3b)
  7. Marginal gain Δ_full = F(S∪{v}) − F(S)  (Definition 3)
  8. Frontier expansion produces tree-connected set  (Theorem 2 Step 1)
"""

import numpy as np
import pytest
from itertools import combinations

from pathfinder.algorithm import (
    PathfinderGreedy,
    compute_F,
    compute_sigma,
    marginal_gain,
    brute_force_optimum,
    _greedy_traverse,
)
from pathfinder.config import ALPHA, BETA, GAMMA, DELTA, EPSILON, K_TOK, TAU_HIGH, TAU_LOW
from pathfinder.tests.fixtures import (
    make_synthetic_kg,
    make_chain_graph,
    make_star_graph,
    make_random_graph,
)


# ── Test 1: F(∅, q) = 0 ────────────────────────────────────────────────────
class TestFEmpty:
    """Corollary to Theorem 1: F(∅, q) = 0 by empty-product convention."""

    def test_f_empty_chain(self):
        kg = make_chain_graph(n=5)
        assert compute_F([], kg) == 0.0

    def test_f_empty_star(self):
        kg = make_star_graph(n_branches=3, branch_len=2)
        assert compute_F([], kg) == 0.0

    def test_f_empty_random(self):
        kg = make_random_graph(n=10, seed=42)
        assert compute_F([], kg) == 0.0

    def test_f_empty_single_node(self):
        D = 8
        emb = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]])
        q = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
        kg = make_synthetic_kg(emb, q)
        assert compute_F([], kg) == 0.0


# ── Test 2: Monotonicity F(S) ≤ F(T) for S ⊆ T ─────────────────────────────
class TestMonotonicity:
    """Lemma 1: F is monotone non-decreasing."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 2024])
    def test_monotonicity_random_graphs(self, seed):
        kg = make_random_graph(n=8, seed=seed)
        N = kg.N

        # Test all subset pairs S ⊆ T
        for size_S in range(0, N):
            for S in combinations(range(N), size_S):
                S_list = list(S)
                F_S = compute_F(S_list, kg)
                # Add one more element
                for v in range(N):
                    if v not in S:
                        T_list = S_list + [v]
                        F_T = compute_F(T_list, kg)
                        assert F_S <= F_T + 1e-10, (
                            f"Monotonicity violated: F({S_list})={F_S:.6f} > "
                            f"F({T_list})={F_T:.6f} (seed={seed})"
                        )

    def test_monotonicity_chain(self):
        kg = make_chain_graph(n=5)
        N = kg.N
        for size_S in range(0, N):
            for S in combinations(range(N), size_S):
                S_list = list(S)
                F_S = compute_F(S_list, kg)
                for v in range(N):
                    if v not in S:
                        T_list = S_list + [v]
                        F_T = compute_F(T_list, kg)
                        assert F_S <= F_T + 1e-10

    def test_monotonicity_with_domain_floor(self):
        """Test that domain floor (max(0,·)) preserves monotonicity even with
        anti-aligned domain embeddings."""
        N = 4
        D = 8
        embeddings = np.eye(N, D)
        # L2-normalise (already unit vectors)
        q_emb = np.zeros(D)
        q_emb[0] = 1.0

        # Domain embeddings: some anti-aligned with q_dom
        phi_dom = np.zeros((N, 2))
        phi_dom[0] = [1.0, 0.0]   # aligned
        phi_dom[1] = [-1.0, 0.0]  # anti-aligned → domain term = 0
        phi_dom[2] = [0.5, 0.5]
        phi_dom[3] = [-0.5, -0.5]  # anti-aligned

        W_dom = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1.0, 0, 0, 0, 0, 0, 0]])
        q_dom = np.array([1.0, 0.0])

        edges = [(0, 1, 0.5), (1, 2, 0.5), (2, 3, 0.5),
                 (1, 0, 0.5), (2, 1, 0.5), (3, 2, 0.5)]

        kg = make_synthetic_kg(
            embeddings, q_emb, edges=edges, phi_dom=phi_dom,
        )
        # Override W_dom and q_dom
        kg.W_dom = W_dom
        kg.q_dom = q_dom
        for i in range(N):
            kg.G.nodes[i]["phi_dom"] = phi_dom[i]

        # Check monotonicity for all subsets
        for size_S in range(0, N):
            for S in combinations(range(N), size_S):
                S_list = list(S)
                F_S = compute_F(S_list, kg)
                for v in range(N):
                    if v not in S:
                        T_list = S_list + [v]
                        F_T = compute_F(T_list, kg)
                        assert F_S <= F_T + 1e-10, (
                            f"Domain floor monotonicity violated: "
                            f"F({S_list})={F_S:.6f} > F({T_list})={F_T:.6f}"
                        )


# ── Test 3: Submodularity Δ(v|S) ≥ Δ(v|T) for S ⊆ T ────────────────────────
class TestSubmodularity:
    """Theorem 1: F is submodular (diminishing marginal returns)."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_submodularity_random(self, seed):
        kg = make_random_graph(n=8, seed=seed)
        N = kg.N

        # For all S ⊆ T and v ∉ T: Δ(v|S) ≥ Δ(v|T)
        for size_S in range(0, N - 2):
            for S in combinations(range(N), size_S):
                S_list = list(S)
                S_set = set(S)
                for v in range(N):
                    if v in S_set:
                        continue
                    # Compute Δ(v|S)
                    F_S = compute_F(S_list, kg)
                    F_Sv = compute_F(S_list + [v], kg)
                    delta_S = F_Sv - F_S

                    # Now try adding an extra element u to make T = S ∪ {u}
                    for u in range(N):
                        if u in S_set or u == v:
                            continue
                        T_list = S_list + [u]
                        F_T = compute_F(T_list, kg)
                        F_Tv = compute_F(T_list + [v], kg)
                        delta_T = F_Tv - F_T

                        assert delta_S >= delta_T - 1e-10, (
                            f"Submodularity violated: Δ(v={v}|S={S_list})={delta_S:.6f} < "
                            f"Δ(v={v}|T={T_list})={delta_T:.6f} (seed={seed})"
                        )

    def test_submodularity_coverage_only(self):
        """Test submodularity of the coverage function f(S,q) specifically."""
        N = 5
        D = 8
        rng = np.random.RandomState(99)
        embeddings = rng.randn(N, D)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        q_emb = rng.randn(D)
        q_emb = q_emb / np.linalg.norm(q_emb)

        kg = make_synthetic_kg(embeddings, q_emb)

        # Extract just the coverage component: f(S,q) = 1 - prod(1-sim(v,q))
        def f_coverage(S):
            if not S:
                return 0.0
            product = 1.0
            for v in S:
                product *= (1.0 - kg.sim_to_query(v))
            return 1.0 - product

        for size_S in range(0, N - 1):
            for S in combinations(range(N), size_S):
                S_list = list(S)
                S_set = set(S)
                for v in range(N):
                    if v in S_set:
                        continue
                    delta_S = f_coverage(S_list + [v]) - f_coverage(S_list)
                    for u in range(N):
                        if u in S_set or u == v:
                            continue
                        T_list = S_list + [u]
                        delta_T = f_coverage(T_list + [v]) - f_coverage(T_list)
                        assert delta_S >= delta_T - 1e-10, (
                            f"Coverage submodularity violated: "
                            f"Δ(v={v}|S={S_list})={delta_S:.6f} < "
                            f"Δ(v={v}|T={T_list})={delta_T:.6f}"
                        )


# ── Test 4: (1−1/e) approximation bound ─────────────────────────────────────
class TestApproximationBound:
    """Theorem 2: F(S_greedy) ≥ (1−1/e) · F(S*_frontier) on small graphs."""

    def _run_greedy_and_brute(self, kg, k=5):
        """Run greedy traversal and brute-force optimum, return ratio."""
        # Get entry node
        phi_sem_q = kg.embeddings @ kg.q_emb
        v0 = int(np.argmax(phi_sem_q))

        # Run greedy with uniform token cost (k_tok large enough for k nodes)
        # Use uniform token cost: each node has ~11 tokens, so k_tok = k * 11
        k_tok = k * 15  # generous budget
        greedy = PathfinderGreedy(k_tok=k_tok, use_sufficiency=False, use_sigma_break=False)
        result = greedy.run(kg)
        F_greedy = compute_F(result.S, kg)

        # Brute-force optimum over connected subtrees
        F_opt, S_opt = brute_force_optimum(kg, v0, k=k)

        return F_greedy, F_opt, result.S, S_opt

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 2024, 31415, 2718])
    def test_approximation_bound_random_graphs(self, seed):
        """On random graphs, greedy should achieve ≥ (1−1/e) of optimum."""
        kg = make_random_graph(n=8, seed=seed, edge_prob=0.4)
        F_greedy, F_opt, S_greedy, S_opt = self._run_greedy_and_brute(kg, k=4)

        if F_opt < 1e-10:
            pytest.skip("Optimum is ~0, ratio undefined")

        ratio = F_greedy / F_opt
        bound = 1.0 - 1.0 / np.e  # ≈ 0.6321

        assert ratio >= bound - 1e-6, (
            f"(1−1/e) bound violated: F_greedy={F_greedy:.6f}, "
            f"F_opt={F_opt:.6f}, ratio={ratio:.4f} < {bound:.4f} (seed={seed})"
        )

    def test_approximation_bound_chain_graph(self):
        """Chain graph: greedy should achieve optimum (tree structure, Corollary 1)."""
        kg = make_chain_graph(n=5)
        F_greedy, F_opt, S_greedy, S_opt = self._run_greedy_and_brute(kg, k=3)

        if F_opt < 1e-10:
            pytest.skip("Optimum is ~0")

        ratio = F_greedy / F_opt
        # On tree graphs, Corollary 1 says the guarantee holds unconditionally
        # and greedy often achieves optimum
        bound = 1.0 - 1.0 / np.e
        assert ratio >= bound - 1e-6, (
            f"Chain graph bound violated: ratio={ratio:.4f} < {bound:.4f}"
        )

    def test_approximation_bound_star_graph(self):
        """Star graph: all branches accessible from center."""
        kg = make_star_graph(n_branches=3, branch_len=2)
        F_greedy, F_opt, S_greedy, S_opt = self._run_greedy_and_brute(kg, k=3)

        if F_opt < 1e-10:
            pytest.skip("Optimum is ~0")

        ratio = F_greedy / F_opt
        bound = 1.0 - 1.0 / np.e
        assert ratio >= bound - 1e-6, (
            f"Star graph bound violated: ratio={ratio:.4f} < {bound:.4f}"
        )

    def test_greedy_never_exceeds_optimum(self):
        """F_greedy should never exceed F_opt (optimum is the max)."""
        kg = make_random_graph(n=6, seed=777, edge_prob=0.5)
        F_greedy, F_opt, _, _ = self._run_greedy_and_brute(kg, k=3)
        assert F_greedy <= F_opt + 1e-10, (
            f"Greedy exceeded optimum: F_greedy={F_greedy:.6f} > F_opt={F_opt:.6f}"
        )


# ── Test 5: σ(S) computation ────────────────────────────────────────────────
class TestSigmaComputation:
    """Section 4.3: σ(S) = min_{v∈S} σ(v₀→v) over tree paths."""

    def test_sigma_single_node(self):
        """σ({v₀}) = φ_conf(v₀) (no edges, just node confidence)."""
        kg = make_chain_graph(n=3)
        v0 = 0
        sigma = compute_sigma([v0], {v0: None}, kg)
        expected = kg.G.nodes[v0]["phi_conf"]
        assert abs(sigma - expected) < 1e-10

    def test_sigma_chain(self):
        """σ for a chain 0→1→2: product of edge weights and node confidences."""
        N = 3
        D = 8
        emb = np.eye(N, D)
        q = np.zeros(D)
        q[0] = 1.0
        edges = [(0, 1, 0.8), (1, 2, 0.6)]
        phi_conf = [0.9, 0.8, 0.7]
        kg = make_synthetic_kg(emb, q, edges=edges, phi_conf=phi_conf)

        parent = {0: None, 1: 0, 2: 1}
        sigma = compute_sigma([0, 1, 2], parent, kg)

        # σ(0→0) = 0.9
        # σ(0→1) = 0.9 * 0.8 * 0.8 = 0.576
        # σ(0→2) = 0.9 * 0.8 * 0.8 * 0.6 * 0.7 = 0.24192
        # σ(S) = min = 0.24192
        expected = 0.9 * 0.8 * 0.8 * 0.6 * 0.7
        assert abs(sigma - expected) < 1e-10, f"σ={sigma:.6f}, expected={expected:.6f}"

    def test_sigma_empty_set(self):
        """σ(∅) = 1.0 (vacuous confidence by convention)."""
        kg = make_chain_graph(n=3)
        sigma = compute_sigma([], {}, kg)
        assert sigma == 1.0

    def test_sigma_takes_minimum(self):
        """σ(S) should be the minimum over all paths, not average."""
        N = 5
        D = 8
        emb = np.eye(N, D)
        q = np.zeros(D)
        q[0] = 1.0
        # Star: 0→1 (strong), 0→2 (weak), 0→3 (medium)
        edges = [(0, 1, 0.9), (0, 2, 0.1), (0, 3, 0.5)]
        phi_conf = [0.9, 0.9, 0.9, 0.9, 0.9]
        kg = make_synthetic_kg(emb, q, edges=edges, phi_conf=phi_conf)

        parent = {0: None, 1: 0, 2: 0, 3: 0}
        sigma = compute_sigma([0, 1, 2, 3], parent, kg)

        # σ(0→2) = 0.9 * 0.1 * 0.9 = 0.081 (weakest path)
        expected = 0.9 * 0.1 * 0.9
        assert abs(sigma - expected) < 1e-10, f"σ={sigma:.6f}, expected={expected:.6f}"


# ── Test 6: Algorithm 1 guard conditions ────────────────────────────────────
class TestGuardConditions:
    """Lines 0a, 0b, 3b of Algorithm 1."""

    def test_empty_graph_guard(self):
        """Line 0a: IF V = ∅ → RETURN ∅, 0, 1.0"""
        # Create a graph with 0 nodes
        D = 8
        emb = np.zeros((0, D))
        q = np.ones(D) / np.sqrt(D)
        kg = make_synthetic_kg(emb, q)
        # kg.N should be 0
        assert kg.N == 0

        greedy = PathfinderGreedy()
        result = greedy.run(kg)
        assert result.S == []
        assert result.F == 0.0
        assert result.sigma == 1.0

    def test_zero_query_embedding_guard(self):
        """Line 0b: IF ‖q_emb‖ = 0 → RETURN ∅, 0, 1.0"""
        N = 3
        D = 8
        emb = np.eye(N, D)
        q = np.zeros(D)  # zero query embedding
        kg = make_synthetic_kg(emb, q)

        greedy = PathfinderGreedy()
        result = greedy.run(kg)
        assert result.S == []
        assert result.F == 0.0
        assert result.sigma == 1.0

    def test_entry_node_exceeds_budget(self):
        """Line 3b: IF token_count(v₀) > K_tok → RETURN ∅, 0, 1.0"""
        N = 3
        D = 8
        emb = np.eye(N, D)
        q = np.zeros(D)
        q[0] = 1.0
        # Make node text very long (more than k_tok=5 tokens)
        texts = ["word " * 100 for _ in range(N)]
        kg = make_synthetic_kg(emb, q, texts=texts)

        greedy = PathfinderGreedy(k_tok=5)  # tiny budget
        result = greedy.run(kg)
        assert result.S == []
        assert result.F == 0.0
        assert result.sigma == 1.0


# ── Test 7: Marginal gain Δ_full = F(S∪{v}) − F(S) ──────────────────────────
class TestMarginalGain:
    """Definition 3: Δ_full(v|S,q) = F(S∪{v},q) − F(S,q)."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_marginal_gain_equals_F_difference(self, seed):
        """Δ_full(v|S,q) should exactly equal F(S∪{v}) − F(S)."""
        kg = make_random_graph(n=8, seed=seed)
        N = kg.N

        # Compute residual for S = {0}
        v0 = 0
        rho = 1.0 - kg.sim_to_query(v0)

        for v in range(1, N):
            # Compute Δ_full via the function
            delta = marginal_gain(v, rho, kg)

            # Compute F(S∪{v}) − F(S) directly
            F_S = compute_F([v0], kg)
            F_Sv = compute_F([v0, v], kg)
            delta_direct = F_Sv - F_S

            assert abs(delta - delta_direct) < 1e-8, (
                f"Δ_full mismatch: marginal_gain={delta:.8f} vs "
                f"F-difference={delta_direct:.8f} (v={v}, seed={seed})"
            )

    def test_marginal_gain_nonnegative(self):
        """Δ_full(v|S,q) ≥ 0 for all v and S (monotonicity)."""
        kg = make_random_graph(n=8, seed=42)
        N = kg.N

        rho = 1.0  # empty set residual
        for v in range(N):
            delta = marginal_gain(v, rho, kg)
            assert delta >= -1e-10, f"Negative marginal gain: Δ({v}|∅)={delta:.6f}"


# ── Test 8: Tree connectivity of greedy output ──────────────────────────────
class TestTreeConnectivity:
    """Theorem 2 Step 1: Algorithm 1 produces a tree-connected set rooted at v₀."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_greedy_output_is_tree_connected(self, seed):
        kg = make_random_graph(n=10, seed=seed, edge_prob=0.4)
        greedy = PathfinderGreedy(use_sufficiency=False, use_sigma_break=False)
        result = greedy.run(kg)

        if not result.S:
            pytest.skip("Empty result")

        # Check that every node (except v₀) has a parent in S
        v0 = result.S[0]
        parent = result.parent

        for v in result.S:
            if v == v0:
                assert parent[v] is None, f"Root {v0} should have null parent"
            else:
                assert parent[v] is not None, f"Node {v} has no parent"
                assert parent[v] in result.S, (
                    f"Parent {parent[v]} of node {v} is not in S"
                )
                # Check edge exists in graph
                assert kg.G.has_edge(parent[v], v), (
                    f"Edge ({parent[v]}, {v}) does not exist in G"
                )

    def test_greedy_output_root_is_entry_node(self):
        """First selected node should be argmax cosine similarity to query."""
        kg = make_random_graph(n=8, seed=42)
        phi_sem_q = kg.embeddings @ kg.q_emb
        expected_v0 = int(np.argmax(phi_sem_q))

        greedy = PathfinderGreedy(use_sufficiency=False, use_sigma_break=False)
        result = greedy.run(kg)

        assert result.S[0] == expected_v0, (
            f"Entry node mismatch: got {result.S[0]}, expected {expected_v0}"
        )


# ── Test 9: Re-traversal protocol ───────────────────────────────────────────
class TestRetraversal:
    """Section 4.3: bounded re-traversal protocol (MAX_RETRIES=3)."""

    def test_retraversal_terminates(self):
        """Re-traversal should always terminate within MAX_RETRIES."""
        # Create a graph where σ will be low (weak edges)
        N = 4
        D = 8
        emb = np.eye(N, D)
        q = np.zeros(D)
        q[0] = 1.0
        edges = [(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01)]  # very weak edges
        phi_conf = [0.1, 0.1, 0.1, 0.1]  # low confidence
        kg = make_synthetic_kg(emb, q, edges=edges, phi_conf=phi_conf)

        greedy = PathfinderGreedy(max_retries=3, use_sufficiency=False)
        result = greedy.run(kg)

        assert result.retries <= 3, f"Retries exceeded MAX_RETRIES: {result.retries}"

    def test_high_sigma_no_retraversal(self):
        """When σ ≥ τ_low, no re-traversal should occur."""
        # Use 2-node chain with strong edge: σ = 0.95 * 0.95 * 0.95 = 0.857 > τ_low
        N = 2
        D = 8
        emb = np.eye(N, D)
        q = np.zeros(D)
        q[0] = 1.0
        edges = [(0, 1, 0.95), (1, 0, 0.95)]
        phi_conf = [0.95, 0.95]
        kg = make_synthetic_kg(emb, q, edges=edges, phi_conf=phi_conf)

        greedy = PathfinderGreedy(max_retries=3, use_sufficiency=False)
        result = greedy.run(kg)
        # σ = 0.95 (conf v0) * 0.95 (edge) * 0.95 (conf v1) = 0.857 > τ_low=0.3
        assert result.sigma >= TAU_LOW, f"σ={result.sigma:.4f} should be ≥ τ_low={TAU_LOW}"
        assert result.retries == 0, f"Unexpected retries: {result.retries}"

    def test_low_confidence_flag(self):
        """When σ < τ_low after MAX_RETRIES, confidence_flag should be LOW."""
        N = 3
        D = 8
        emb = np.eye(N, D)
        q = np.zeros(D)
        q[0] = 1.0
        edges = [(0, 1, 0.01), (1, 2, 0.01)]
        phi_conf = [0.01, 0.01, 0.01]
        kg = make_synthetic_kg(emb, q, edges=edges, phi_conf=phi_conf)

        greedy = PathfinderGreedy(max_retries=2, use_sufficiency=False)
        result = greedy.run(kg)

        if result.sigma < TAU_LOW:
            assert result.confidence_flag == "LOW"


# ── Test 10: Coverage function properties ───────────────────────────────────
class TestCoverageFunction:
    """Definition 2: f(S,q) = 1 − ∏(1−sim(v,q))."""

    def test_coverage_empty(self):
        """f(∅, q) = 0 (empty product = 1, so 1−1=0)."""
        kg = make_chain_graph(n=3)
        # f is the coverage component: α·f(S,q) is the first term of F
        # F(∅) = 0 implies α·f(∅) = 0 implies f(∅) = 0
        assert compute_F([], kg) == 0.0

    def test_coverage_monotone_increasing(self):
        """f(S,q) is monotone non-decreasing."""
        kg = make_random_graph(n=6, seed=42)
        N = kg.N

        def f_cov(S):
            if not S:
                return 0.0
            product = 1.0
            for v in S:
                product *= (1.0 - kg.sim_to_query(v))
            return 1.0 - product

        prev = f_cov([])
        for size in range(1, N + 1):
            for S in combinations(range(N), size):
                val = f_cov(list(S))
                # Not all supersets, but adding elements should not decrease
                # Check specific chain: ∅ ⊂ {0} ⊂ {0,1} ⊂ ...
        # Simple chain test
        vals = [f_cov(list(range(i))) for i in range(N + 1)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-10

    def test_coverage_bounded_01(self):
        """f(S,q) ∈ [0, 1] for all S."""
        kg = make_random_graph(n=8, seed=42)
        N = kg.N

        def f_cov(S):
            if not S:
                return 0.0
            product = 1.0
            for v in S:
                product *= (1.0 - kg.sim_to_query(v))
            return 1.0 - product

        for size in range(0, N + 1):
            for S in combinations(range(N), size):
                val = f_cov(list(S))
                assert -1e-10 <= val <= 1.0 + 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
