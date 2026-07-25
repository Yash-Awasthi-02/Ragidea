"""
PATHFINDER Experiment — Phase D1: BGE/E5 Embedding Upgrade
================================================================
Rebuild KGs with stronger embedding models (BGE-large, E5-large) and
evaluate whether better embeddings improve both graph quality and dense retrieval.

Supported models:
  - BAAI/bge-large-en-v1.5  (1024-dim, ~1.3GB download)
  - intfloat/e5-large-v2       (1024-dim, ~1.3GB download)
  - all-MiniLM-L6-v2           (384-dim, default baseline)

Usage:
    :: Sentence-level with BGE
    python 01c_build_kg_bge.py --max_samples 500 --output data/hotpotqa_graphs_bge.pkl --encoder_model BAAI/bge-large-en-v1.5

    :: Passage-level with BGE
    python 01c_build_kg_bge.py --max_samples 500 --output data/hotpotqa_graphs_bge_passage.pkl --encoder_model BAAI/bge-large-en-v1.5 --passage_level

    :: E5 alternative
    python 01c_build_kg_bge.py --max_samples 500 --output data/hotpotqa_graphs_e5.pkl --encoder_model intfloat/e5-large-v2
"""
import os, pickle, argparse, numpy as np, networkx as nx, spacy, nltk
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "BAAI/bge-large-en-v1.5"
THETA_EDGE      = 0.30
W_ENTITY        = 0.70
PHI_CONF_INIT   = 0.95
PCA_K          = 16
PR_DAMPING     = 0.85
MIN_SENT_LEN    = 8       # sentence-level minimum words
MIN_PASSAGE_WORDS = 15   # passage-level minimum words
PASSAGE_MAX_WORDS = 80   # flush passage at this word count


class UniversalKGBuilder:
    """
    KG builder that supports both sentence-level and passage-level node segmentation
    with any sentence-transformer embedding model.
    """

    def __init__(self, model_name: str, passage_level: bool = False):
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.model_name = model_name
        self.passage_level = passage_level
        print("Loading spaCy NER pipeline")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

    def _entities(self, text: str) -> set[str]:
        doc = self.nlp(text[:1000])
        return {ent.text.lower().strip() for ent in doc.ents if len(ent.text.strip()) > 2}

    def _segment_sentences(self, sentences: list[str]) -> list[str]:
        """Sentence-level: each sentence is a node (filter short ones)."""
        return [s.strip() for s in sentences if len(s.split()) >= MIN_SENT_LEN]

    def _segment_passages(self, sentences: list[str]) -> list[str]:
        """Passage-level: group sentences into passages of ~80 words."""
        passages = []
        current = []
        word_count = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                if current and word_count >= MIN_PASSAGE_WORDS:
                    passages.append(" ".join(current))
                current = []
                word_count = 0
                continue

            current.append(sent)
            word_count += len(sent.split())

            if word_count >= PASSAGE_MAX_WORDS:
                passages.append(" ".join(current))
                current = []
                word_count = 0

        if current and word_count >= MIN_PASSAGE_WORDS:
            passages.append(" ".join(current))

        return passages

    def build(self, query: str, docs: list[dict]) -> dict | None:
        """Build per-query KG."""
        nodes = []
        raw_texts = []

        for doc in docs:
            title = doc["title"]
            sentences = doc["sentences"]

            if self.passage_level:
                segments = self._segment_passages(sentences)
            else:
                segments = self._segment_sentences(sentences)

            for seg_idx, seg in enumerate(segments):
                nodes.append({
                    "text":      seg,
                    "doc_title": title,
                    "sent_idx":  seg_idx,
                    "phi_conf":  PHI_CONF_INIT,
                    "phi_temp":  1.0,
                    "is_passage": self.passage_level,
                })
                raw_texts.append(seg)

        N = len(nodes)
        if N == 0:
            return None

        # Embeddings
        embs = self.embedder.encode(
            raw_texts, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        )
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]

        # Domain embeddings (PCA)
        k = min(PCA_K, N - 1, embs.shape[1])
        k = max(1, k)
        pca = PCA(n_components=k)
        pca.fit(embs)
        W_dom = pca.components_
        phi_dom_matrix = pca.transform(embs)
        row_norms = np.linalg.norm(phi_dom_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        phi_dom_matrix = phi_dom_matrix / row_norms
        q_dom_raw = W_dom @ q_emb
        q_dom_norm = np.linalg.norm(q_dom_raw)
        q_dom = q_dom_raw / q_dom_norm if q_dom_norm > 0 else q_dom_raw

        # Similarity
        sim = embs @ embs.T
        phi_sem_q = embs @ q_emb

        # Build graph
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

        # PageRank
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
            "nodes":          nodes,
            "embeddings":      embs,
            "phi_dom_matrix":  phi_dom_matrix,
            "W_dom":           W_dom,
            "q_emb":           q_emb,
            "q_dom":           q_dom,
            "G":               G,
            "phi_imp":         phi_imp,
            "N":               N,
        }


def load_hotpotqa(split: str = "validation", max_samples: int = None):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def parse_context(example: dict) -> list[dict]:
    titles = example["context"]["title"]
    sent_lists = example["context"]["sentences"]
    return [{"title": t, "sentences": s} for t, s in zip(titles, sent_lists)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KGs with custom embedding model")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", default="data/hotpotqa_graphs_bge.pkl")
    parser.add_argument("--encoder_model", default=DEFAULT_MODEL,
                        help="Sentence-transformer model name (default: BAAI/bge-large-en-v1.5)")
    parser.add_argument("--passage_level", action="store_true",
                        help="Use passage-level node segmentation (default: sentence-level)")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    node_type = "passage" if args.passage_level else "sentence"
    print(f"\nLoading HotpotQA ({args.split})...")
    dataset = load_hotpotqa(args.split, args.max_samples)
    print(f"  {len(dataset)} examples")
    print(f"  Encoder: {args.encoder_model}")
    print(f"  Node type: {node_type}-level")

    builder = UniversalKGBuilder(args.encoder_model, passage_level=args.passage_level)
    records = []
    skipped = 0

    for ex in tqdm(dataset, desc=f"Building {node_type} KGs"):
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

    print(f"\nBuilt {len(records)} {node_type} KGs  |  skipped {skipped}")
    if records:
        print(f"Avg nodes per query: {sum(r['graph']['N'] for r in records) / len(records):.1f}")
    print(f"Encoder: {args.encoder_model}")
    print(f"Saved → {args.output}")

    with open(args.output, "wb") as f:
        pickle.dump(records, f, protocol=4)
