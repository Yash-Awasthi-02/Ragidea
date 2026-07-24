"""
PATHFINDER Experiment — Phase A2: Passage-Level KG Construction
================================================================
Build per-query KGs with passage-level nodes instead of sentence-level.
Each paragraph becomes a single node (~10 nodes/query vs ~40 nodes/query).

Usage:
    python 01b_build_kg_passage.py --max_samples 500 --output data/hotpotqa_graphs_passage.pkl
    python 01b_build_kg_passage.py --output data/hotpotqa_graphs_passage_full.pkl
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

MODEL_NAME     = "all-MiniLM-L6-v2"
THETA_EDGE     = 0.30
W_ENTITY       = 0.70
PHI_CONF_INIT  = 0.95
PCA_K          = 16
PR_DAMPING     = 0.85
MIN_PASSAGE_WORDS = 15  # minimum words for a passage to become a node


class PassageKGBuilder:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.embedder = SentenceTransformer(MODEL_NAME)
        print("Loading spaCy NER pipeline")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

    def _entities(self, text: str) -> set[str]:
        doc = self.nlp(text[:1000])
        return {ent.text.lower().strip() for ent in doc.ents if len(ent.text.strip()) > 2}

    def _segment_passages(self, sentences: list[str]) -> list[str]:
        """Group sentences into passages by paragraph breaks.
        HotpotQA sentences are already paragraph-split — each list item is a paragraph.
        We join consecutive short sentences and keep long ones as standalone passages.
        """
        passages = []
        current_passage = []
        word_count = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                if current_passage and word_count >= MIN_PASSAGE_WORDS:
                    passages.append(" ".join(current_passage))
                current_passage = []
                word_count = 0
                continue

            current_passage.append(sent)
            word_count += len(sent.split())

            # If passage is long enough, flush it
            if word_count >= 80:
                passages.append(" ".join(current_passage))
                current_passage = []
                word_count = 0

        # Flush remaining
        if current_passage and word_count >= MIN_PASSAGE_WORDS:
            passages.append(" ".join(current_passage))

        return passages

    def build(self, query: str, docs: list[dict]) -> dict | None:
        """Build per-query KG with passage-level nodes."""
        nodes = []
        raw_texts = []

        for doc in docs:
            title = doc["title"]
            passages = self._segment_passages(doc["sentences"])

            for sent_idx, passage in enumerate(passages):
                nodes.append({
                    "text":      passage,
                    "doc_title": title,
                    "sent_idx":  sent_idx,  # passage index within doc
                    "phi_conf":  PHI_CONF_INIT,
                    "phi_temp":  1.0,
                    "is_passage": True,
                })
                raw_texts.append(passage)

        N = len(nodes)
        if N == 0:
            return None

        # Embeddings
        embs = self.embedder.encode(
            raw_texts, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False
        )
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]

        # Domain embeddings
        k = min(PCA_K, N - 1, embs.shape[1])
        if k < 1:
            k = 1
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

        # Semantic edges (lower threshold for passages since they're longer)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = float(max(0.0, sim[i, j]))
                if w > THETA_EDGE:
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
            "nodes":           nodes,
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
    parser = argparse.ArgumentParser(description="Build passage-level KGs from HotpotQA")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", default="data/hotpotqa_graphs_passage.pkl")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    print(f"\nLoading HotpotQA ({args.split})...")
    dataset = load_hotpotqa(args.split, args.max_samples)
    print(f"  {len(dataset)} examples")

    builder = PassageKGBuilder()
    records = []
    skipped = 0

    for ex in tqdm(dataset, desc="Building passage KGs"):
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

    print(f"\nBuilt {len(records)} passage KGs  |  skipped {skipped}")
    if records:
        print(f"Avg nodes per query: {sum(r['graph']['N'] for r in records) / len(records):.1f}")
    print(f"Saved → {args.output}")

    with open(args.output, "wb") as f:
        pickle.dump(records, f, protocol=4)
