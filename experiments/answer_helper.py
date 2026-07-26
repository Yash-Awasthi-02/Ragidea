"""
PATHFINDER — Unified Answer-Generation Context Path (Inspection §2.1 / §2.2)
============================================================================
Single canonical path from (retriever, records) → EM/F1 so that EM values are
COMPARABLE across scripts. Root cause of the EM "discrepancy" (C2-2, resolved —
see paper §7.6.10): every script fed a DIFFERENT node set and a
DIFFERENT effective context budget to the SAME scorer, so EM could not be
compared across scripts.

This module fixes three things:
  1. ONE context assembly (`build_context_budgeted`) used by every script.
  2. ONE scoring entry point (`answer_records`) that fixes retriever, node set,
     prompt, and context budget before scoring.
  3. A PER-NODE character budget (inspection §2.2) so passage-level nodes
     (≥80 words) are not truncated to 1–2 nodes while sentence nodes fit many —
     removing a granularity confound at answer time.

Nothing here calls a dataset or runs by itself; `answer_records` is invoked by
the eval scripts in PLAN §6 (user-driven).
"""

from __future__ import annotations

import networkx as nx

from generate_answers import (
    generate_answer, exact_match, f1_score, RATE_LIMITED, SYSTEM_PROMPT,
)


# ── Per-node-budgeted context builder (Inspection §2.2) ──────────────────────
def build_context_budgeted(node_indices: list[int],
                           G: nx.DiGraph,
                           total_chars: int = 12000,
                           per_node_chars: int = 1500) -> str:
    """
    Assemble context with BOTH a total cap and a per-node cap.

    Why per-node: the flat `max_chars=6000` in generate_answers.build_context
    lets the first 1–2 passage nodes consume the whole budget and truncates the
    rest, while sentence nodes fit many. At matched recall this presents
    DIFFERENT amounts of evidence to the generator for passage vs sentence
    granularity — a confound in any granularity comparison. Capping each node
    and raising the total budget makes the evidence presented a function of the
    RETRIEVED SET, not of node length.

    Args:
        node_indices  : selected node ids in selection order.
        G             : the query graph (node "text" attributes).
        total_chars   : hard cap on the assembled context.
        per_node_chars: hard cap on each individual node's text.

    Returns:
        Numbered context string "[i] text\\n..." within both budgets.
    """
    parts = []
    total = 0
    for idx, node_idx in enumerate(node_indices):
        text = (G.nodes[node_idx].get("text", "") or "").strip()
        if not text:
            continue
        if len(text) > per_node_chars:
            text = text[:per_node_chars].rstrip() + " …"
        part = f"[{idx+1}] {text}"
        if total + len(part) > total_chars:
            break
        parts.append(part)
        total += len(part) + 1
    return "\n".join(parts)


# ── Canonical answer path ─────────────────────────────────────────────────────
def answer_one(question: str,
               node_indices: list[int],
               G: nx.DiGraph,
               client,
               model: str,
               total_chars: int = 12000,
               per_node_chars: int = 1500) -> str:
    """
    Answer a single question from a retrieved node set, using the canonical
    budgeted context. Returns the model's raw prediction string (or RATE_LIMITED).
    """
    if not node_indices:
        return "unanswerable"
    context = build_context_budgeted(node_indices, G, total_chars, per_node_chars)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    # Reuse the rate-limit-safe retry loop from generate_answers by calling its
    # lower-level completion path through generate_answer is not possible (it
    # rebuilds context internally), so we delegate to generate_answer for the
    # API call but with our context. To keep ONE code path, generate_answer is
    # extended (below in generate_answers.py) to accept a prebuilt context.
    return generate_answer(question, node_indices, G, client, model=model,
                           _prebuilt_context=context)


# ── Batch scoring with a FIXED retriever + node set ──────────────────────────
def answer_records(records: list[dict],
                   retrieve_fn,
                   client,
                   model: str,
                   total_chars: int = 12000,
                   per_node_chars: int = 1500) -> dict:
    """
    The single entry point every EM-producing script should use.

    Args:
        records     : list of graph records ({"question","answer","graph"}).
        retrieve_fn : fn(record) -> list[int]  — the retrieved node set. THIS is
                      the only thing that varies between systems; context build,
                      prompt, and budget are held fixed here.
        client      : Groq client.
        model       : generator model id.

    Returns:
        {"em","f1","predictions","n_scored","n_rate_limited","quota_exhausted"}
        EM/F1 computed over NON-rate-limited predictions only (§2.1 guard).
    """
    ems, f1s, preds = [], [], []
    n_rate_limited = 0
    for rec in records:
        q    = rec["question"]
        gold = rec["answer"]
        G    = rec["graph"]["G"]
        nodes = retrieve_fn(rec)
        pred  = answer_one(q, nodes, G, client, model,
                           total_chars=total_chars, per_node_chars=per_node_chars)
        if pred == RATE_LIMITED:
            n_rate_limited += 1
            preds.append(RATE_LIMITED)
            continue
        ems.append(exact_match(pred, gold))
        f1s.append(f1_score(pred, gold))
        preds.append(pred)

    n_scored = len(ems)
    quota_exhausted = n_rate_limited > 0
    return {
        "em":              sum(ems) / n_scored if n_scored else 0.0,
        "f1":              sum(f1s) / n_scored if n_scored else 0.0,
        "predictions":     preds,
        "n_scored":        n_scored,
        "n_rate_limited":  n_rate_limited,
        "quota_exhausted": quota_exhausted,
        "context_budget":  {"total_chars": total_chars,
                            "per_node_chars": per_node_chars},
    }
