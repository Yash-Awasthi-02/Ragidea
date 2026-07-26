"""
PATHFINDER Experiment — Step 4: LLM Answer Generation + EM/F1 Evaluation
=========================================================================
Uses Groq API (free tier) — model: llama-3.3-70b-versatile
Falls back to llama3-8b-8192 if rate-limited (higher req/min).

EM/F1 normalization follows the HotpotQA official evaluation script:
  lowercase → remove articles → remove punctuation → whitespace-normalize
"""

import os
import re
import time
import string
import networkx as nx
from groq import Groq

# ── Model config ───────────────────────────────────────────────────────────────
PRIMARY_MODEL  = "llama-3.3-70b-versatile"   # best free model; 14,400 req/day
FALLBACK_MODEL = "llama3-8b-8192"            # higher rate limit if needed
TEMPERATURE    = 0
MAX_TOKENS     = 100

# Sentinel returned when the Groq quota is exhausted after all retries.
# Distinct from a genuine "unanswerable" so callers can detect corrupted runs.
RATE_LIMITED = "__RATE_LIMITED__"


class RateLimitExhausted(RuntimeError):
    """Raised when all retries fail due to rate-limit / quota exhaustion."""

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "Be concise — one sentence or a short phrase. "
    "If the answer cannot be determined from the context, reply exactly: unanswerable"
)


# ── Context builder ────────────────────────────────────────────────────────────
def build_context(node_indices: list[int], G: nx.DiGraph,
                  max_chars: int = 6000) -> str:
    """
    Concatenate node texts in selection order.
    Hard-cap at max_chars to stay within model context limits.
    """
    parts = []
    total = 0
    for idx, node_idx in enumerate(node_indices):
        text = G.nodes[node_idx].get("text", "").strip()
        if not text:
            continue
        part = f"[{idx+1}] {text}"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part) + 1
    return "\n".join(parts)


# ── LLM generation ─────────────────────────────────────────────────────────────
def generate_answer(question: str,
                    node_indices: list[int],
                    G: nx.DiGraph,
                    client: Groq,
                    model: str = PRIMARY_MODEL,
                    max_retries: int = 4,
                    _prebuilt_context: str = None) -> str:
    """
    Generate answer for a question given a list of retrieved node indices.
    Handles Groq rate limits with exponential backoff.

    Returns RATE_LIMITED sentinel if every retry failed due to rate-limit /
    quota exhaustion, so callers never mistake a corrupted run for EM=0.
    Returns "" only for non-rate-limit persistent failures.

    `_prebuilt_context` (optional): a pre-assembled context string supplied by
    answer_helper.answer_one so the UNIFIED context path and the legacy path
    share this single rate-limit-safe retry loop (no duplicated API logic).
    When provided, node_indices is used only for the empty-set guard.
    """
    if not node_indices:
        return "unanswerable"

    context = _prebuilt_context if _prebuilt_context is not None \
        else build_context(node_indices, G)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    current_model = model
    saw_rate_limit = False
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err or "quota" in err:
                saw_rate_limit = True
                wait = 2 ** attempt          # 1, 2, 4, 8 seconds
                time.sleep(wait)
                if attempt == 1 and current_model == PRIMARY_MODEL:
                    current_model = FALLBACK_MODEL   # try faster model
            elif "model" in err or "404" in err:
                current_model = FALLBACK_MODEL       # model not available
            else:
                # Non-rate-limit error; wait briefly and retry
                time.sleep(1)

    # All retries failed — flag quota exhaustion distinctly so the run is
    # detectable as corrupted rather than silently scored as EM=0.
    return RATE_LIMITED if saw_rate_limit else ""


# ── EM / F1 evaluation ─────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    """HotpotQA official normalization: lower → strip articles → strip punc → whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold: str) -> int:
    return int(_normalize(prediction) == _normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    recall    = sum(1 for t in gold_tokens if t in common) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Batch evaluation helper ────────────────────────────────────────────────────
def evaluate_batch(records: list[dict],
                   system_results: dict[str, list[list[int]]],
                   groq_api_key: str) -> dict:
    """
    Run LLM generation and compute EM/F1 for multiple systems.

    Args:
        records          : list of graph records (with "question", "answer", "graph")
        system_results   : {"system_name": [node_list_per_query, ...]}
        groq_api_key     : Groq API key

    Returns:
        dict: {"system_name": {"em", "f1", "predictions",
                                "n_rate_limited", "quota_exhausted"}}

    EM/F1 are computed over NON-rate-limited predictions only. If any query hit
    the rate limit, "quota_exhausted" is True and the run should be treated as
    INVALID and re-run with a fresh key — never silently scored as EM=0.
    """
    client = Groq(api_key=groq_api_key)
    output = {}

    for system_name, node_lists in system_results.items():
        print(f"  Generating answers: {system_name}...")
        ems, f1s, preds = [], [], []
        n_rate_limited = 0
        n_api_calls = 0
        for i, (rec, nodes) in enumerate(zip(records, node_lists)):
            G    = rec["graph"]["G"]
            q    = rec["question"]
            gold = rec["answer"]
            pred = generate_answer(q, nodes, G, client)
            n_api_calls += 1

            if pred == RATE_LIMITED:
                n_rate_limited += 1
                preds.append(RATE_LIMITED)
                continue  # exclude from EM/F1 — do not score as a wrong answer

            ems.append(exact_match(pred, gold))
            f1s.append(f1_score(pred, gold))
            preds.append(pred)

            if (i + 1) % 25 == 0:
                print(f"    ... {i+1}/{len(records)} queries "
                      f"({n_api_calls} API calls, {n_rate_limited} rate-limited)")

        quota_exhausted = n_rate_limited > 0
        n_scored = len(ems)
        output[system_name] = {
            "em":              sum(ems) / n_scored if n_scored else 0.0,
            "f1":              sum(f1s) / n_scored if n_scored else 0.0,
            "predictions":     preds,
            "n_scored":        n_scored,
            "n_rate_limited":  n_rate_limited,
            "quota_exhausted": quota_exhausted,
        }
        status = "VALID" if not quota_exhausted else \
                 f"INVALID — {n_rate_limited}/{len(records)} rate-limited"
        print(f"    EM={output[system_name]['em']:.3f}  "
              f"F1={output[system_name]['f1']:.3f}  "
              f"(scored {n_scored}/{len(records)})  [{status}]")
        if quota_exhausted:
            print(f"    ⚠ QUOTA EXHAUSTED for {system_name}: re-run with a fresh "
                  f"GROQ_API_KEY. Do NOT report these EM/F1 values.")

    return output


# ── CLI smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle, sys
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("Set GROQ_API_KEY environment variable first.")
        sys.exit(1)

    path = "data/hotpotqa_graphs.pkl"
    try:
        with open(path, "rb") as f:
            records = pickle.load(f)
    except FileNotFoundError:
        print(f"No cache at {path}. Run 01_build_kg.py first.")
        sys.exit(1)

    client = Groq(api_key=key)
    print("Testing LLM generation on first 3 queries...\n")
    for rec in records[:3]:
        gd   = rec["graph"]
        q    = rec["question"]
        gold = rec["answer"]
        # Use top-3 nodes by similarity as a quick test
        from naive_rag_helper import naive_rag
        nodes = []
        try:
            from run_baselines import naive_rag
            nodes = naive_rag(gd, k=3)
        except Exception:
            nodes = list(range(min(3, gd["N"])))

        pred = generate_answer(q, nodes, gd["G"], client)
        print(f"Q:    {q[:70]}...")
        print(f"Gold: {gold}")
        print(f"Pred: {pred}")
        print(f"EM={exact_match(pred, gold)}  F1={f1_score(pred, gold):.3f}\n")
