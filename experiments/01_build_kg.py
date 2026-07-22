"""
PATHFINDER Experiment — Step 1: Knowledge Graph Construction
============================================================
Builds per-query KGs from HotpotQA candidate document pool (distractor setting).
Each query gets a mini-KG from its 10 candidate documents (~48 sentence-level nodes).

Node facets computed:
  phi_sem  : sentence embedding (all-MiniLM-L6-v2, 384-dim)
  phi_temp : 1.0 constant (no timestamps in HotpotQA)
  phi_imp  : PageRank on W-weighted graph (damping=0.85)
  phi_dom  : PCA K=16 projection of phi_sem (per-query fit)
  phi_conf : 0.70 initial value

Edges:
  Semantic : cosine(phi_sem(u), phi_sem(v)) > 0.3 → W = max(0, cosine)
  Entity   : shared named entity (spaCy NER) → W = max(existing, 0.70)

Output: data/hotpotqa_graphs.pkl — list of per-query graph records
"""

import os
import pickle
import argparse
import numpy as np
import networkx as nx
import spacy
import nltk
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME     = "all-MiniLM-L6-v2"
THETA_EDGE     = 0.30   # semantic edge admission threshold
W_ENTITY       = 0.70   # entity co-mention edge weight
PHI_CONF_INIT  = 0.95   # initial epistemic confidence (0.70 caused σ collapse)
PCA_K          = 16     # domain embedding dimensionality
PR_DAMPING     = 0.85   # PageRank damping factor
MIN_SENT_LEN   = 8      # minimum words for a sentence to become a node


# ── KG Builder ────────────────────────────────────────────────────────────────
class KGBuilder:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.embedder = SentenceTransformer(MODEL_NAME)
        print("Loading spaCy NER pipeline")
        # Only NER — disable parser and tagger for speed
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

    def _segment(self, text: str) -> list[str]:
        """Sentence-tokenize a document; filter very short sentences."""
        sents = sent_tokenize(text.strip())
        return [s.strip() for s in sents if len(s.split()) >= MIN_SENT_LEN]

    def _entities(self, text: str) -> set[str]:
        """Extract lowercased named entity strings (spaCy NER)."""
        doc = self.nlp(text[:512])   # cap for speed on long sentences
        return {ent.text.lower().strip() for ent in doc.ents
                if len(ent.text.strip()) > 2}

    def build(self, query: str, docs: list[dict]) -> dict | None:
        """
        Build per-query KG.

        Args:
            query : question string
            docs  : list of {"title": str, "sentences": list[str]}

        Returns:
            dict with keys: nodes, embeddings, phi_dom_matrix, W_dom,
                            q_emb, q_dom, G, phi_imp, N
            or None if graph is empty.
        """
        # ── 1. Collect sentence nodes ────────────────────────────────────────
        nodes = []
        raw_texts = []

        for doc in docs:
            title = doc["title"]
            for sent_idx, sent in enumerate(doc["sentences"]):
                sent = sent.strip()
                if len(sent.split()) < MIN_SENT_LEN:
                    continue
                nodes.append({
                    "text":      sent,
                    "doc_title": title,
                    "sent_idx":  sent_idx,
                    "phi_conf":  PHI_CONF_INIT,
                    "phi_temp":  1.0,           # constant; no timestamps
                })
                raw_texts.append(sent)

        N = len(nodes)
        if N == 0:
            return None

        # ── 2. Embeddings ────────────────────────────────────────────────────
        embs = self.embedder.encode(
            raw_texts, normalize_embeddings=True,
            batch_size=128, show_progress_bar=False
        )  # shape (N, 384), L2-normalised → cosine = dot product

        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        )[0]  # (384,)

        # ── 3. Domain embeddings (PCA K=16) ──────────────────────────────────
        k = min(PCA_K, N - 1, embs.shape[1])
        pca = PCA(n_components=k)
        pca.fit(embs)
        W_dom = pca.components_                   # (k, 384)
        phi_dom_matrix = pca.transform(embs)      # (N, k)
        # L2-normalise each domain vector row
        row_norms = np.linalg.norm(phi_dom_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        phi_dom_matrix = phi_dom_matrix / row_norms

        q_dom_raw = W_dom @ q_emb               # (k,)
        q_dom_norm = np.linalg.norm(q_dom_raw)
        q_dom = q_dom_raw / q_dom_norm if q_dom_norm > 0 else q_dom_raw

        # ── 4. Similarity matrix ─────────────────────────────────────────────
        sim = embs @ embs.T       # (N, N) cosine similarities
        phi_sem_q = embs @ q_emb  # (N,)  cosine sim to query

        # ── 5. Build DiGraph ─────────────────────────────────────────────────
        G = nx.DiGraph()
        for i in range(N):
            G.add_node(i,
                       text=nodes[i]["text"],
                       doc_title=nodes[i]["doc_title"],
                       sent_idx=nodes[i]["sent_idx"],
                       phi_conf=nodes[i]["phi_conf"],
                       phi_temp=nodes[i]["phi_temp"],
                       phi_dom=phi_dom_matrix[i],
                       sim_to_query=float(max(0.0, phi_sem_q[i])))

        # Semantic edges
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = float(max(0.0, sim[i, j]))
                if w > THETA_EDGE:
                    if G.has_edge(i, j):
                        if G[i][j]["weight"] < w:
                            G[i][j]["weight"] = w
                    else:
                        G.add_edge(i, j, weight=w)

        # Entity co-mention edges
        entity_sets = [self._entities(n["text"]) for n in nodes]
        for i in range(N):
            for j in range(i + 1, N):
                if entity_sets[i] & entity_sets[j]:
                    for u, v in [(i, j), (j, i)]:
                        if G.has_edge(u, v):
                            G[u][v]["weight"] = max(G[u][v]["weight"], W_ENTITY)
                        else:
                            G.add_edge(u, v, weight=W_ENTITY)

        # ── 6. PageRank → phi_imp ────────────────────────────────────────────
        if G.number_of_edges() > 0:
            try:
                pr = nx.pagerank(G, alpha=PR_DAMPING, max_iter=100, weight="weight")
            except nx.PowerIterationFailedConvergence:
                pr = {i: 1.0 / N for i in range(N)}
        else:
            pr = {i: 1.0 / N for i in range(N)}

        pr_arr = np.array([pr[i] for i in range(N)])
        pr_min, pr_max = pr_arr.min(), pr_arr.max()
        phi_imp = ((pr_arr - pr_min) / (pr_max - pr_min)
                   if pr_max > pr_min else np.full(N, 0.5))

        for i in range(N):
            G.nodes[i]["phi_imp"] = float(phi_imp[i])

        return {
            "nodes":           nodes,
            "embeddings":      embs,          # (N, 384) float32
            "phi_dom_matrix":  phi_dom_matrix, # (N, k)
            "W_dom":           W_dom,          # (k, 384)
            "q_emb":           q_emb,          # (384,)
            "q_dom":           q_dom,          # (k,)
            "G":               G,
            "phi_imp":         phi_imp,        # (N,)
            "N":               N,
        }


# ── HotpotQA helpers ─────────────────────────────────────────────────────────
def load_hotpotqa(split: str = "validation", max_samples: int = None):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def parse_context(example: dict) -> list[dict]:
    """Convert HotpotQA context dict to list of {title, sentences}."""
    titles = example["context"]["title"]
    sent_lists = example["context"]["sentences"]
    return [{"title": t, "sentences": s} for t, s in zip(titles, sent_lists)]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-query KGs from HotpotQA")
    parser.add_argument("--split",       default="validation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of queries (None = full split)")
    parser.add_argument("--output",      default="data/hotpotqa_graphs.pkl")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    print(f"\nLoading HotpotQA ({args.split})...")
    dataset = load_hotpotqa(args.split, args.max_samples)
    print(f"  {len(dataset)} examples")

    builder = KGBuilder()
    records = []
    skipped = 0

    for ex in tqdm(dataset, desc="Building KGs"):
        docs = parse_context(ex)
        gd = builder.build(ex["question"], docs)
        if gd is None:
            skipped += 1
            continue
        records.append({
            "id":               ex["id"],
            "question":         ex["question"],
            "answer":           ex["answer"],
            "supporting_facts": ex["supporting_facts"],
            "graph":            gd,
        })

    print(f"\nBuilt {len(records)} graphs  |  skipped {skipped}")
    print(f"Avg nodes per query: {sum(r['graph']['N'] for r in records) / len(records):.1f}")

    with open(args.output, "wb") as f:
        pickle.dump(records, f, protocol=4)
    print(f"Saved → {args.output}")
