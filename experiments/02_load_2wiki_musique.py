"""
PATHFINDER Experiment — Step 2: Loader for 2WikiMultihopQA and MuSiQue
========================================================================
Loads 2WikiMultihopQA / MuSiQue validation split and converts records to the
standard PATHFINDER knowledge graph record format used by 01_build_kg.py.
"""

import os
import argparse
import pickle
import importlib
from tqdm import tqdm
from datasets import load_dataset

kg_mod = importlib.import_module("01_build_kg")
KGBuilder = kg_mod.KGBuilder
MODEL_NAME = kg_mod.MODEL_NAME

def load_2wiki_records(split="validation", max_samples=None):
    print(f"Loading 2WikiMultihopQA ({split})...")
    # Load 2WikiMultihopQA dataset from community HF mirror
    ds = load_dataset("framolfese/2WikiMultihopQA", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds

def load_musique_records(split="validation", max_samples=None):
    print(f"Loading MuSiQue ({split})...")
    # Load MuSiQue dataset from community HF mirror
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["2wiki", "musique"], required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    builder = KGBuilder()
    
    if args.dataset == "2wiki":
        ds = load_2wiki_records(args.split, args.max_samples)
    else:
        ds = load_musique_records(args.split, args.max_samples)

    records = []
    print(f"Building KGs for {len(ds)} {args.dataset} examples...")
    for idx, item in enumerate(tqdm(ds, desc=f"Building {args.dataset} KGs")):
        question = item["question"]
        docs = []

        if args.dataset == "2wiki":
            # 2Wiki context is a dict {"title": [...], "sentences": [...]}
            ctx = item.get("context", {})
            titles = ctx.get("title", [])
            sentences_list = ctx.get("sentences", [])
            for t, s_list in zip(titles, sentences_list):
                docs.append({"title": t, "sentences": s_list})
        else:
            # MuSiQue paragraphs is a list of dicts [{"title": ..., "paragraph_text": ...}, ...]
            paragraphs = item.get("paragraphs", [])
            for p in paragraphs:
                t = p.get("title", "")
                text = p.get("paragraph_text", "")
                # Segment paragraph_text into sentences
                s_list = builder._segment(text)
                docs.append({"title": t, "sentences": s_list})

        rec = builder.build(question, docs)
        if rec is not None:
            if args.dataset == "2wiki":
                sf = item.get("supporting_facts", {"title": [], "sent_id": []})
                if isinstance(sf, list):
                    # 2Wiki supporting_facts is list of [title, sent_idx]
                    sf_dict = {
                        "title": [s[0] for s in sf if len(s) > 1],
                        "sent_id": [s[1] for s in sf if len(s) > 1]
                    }
                else:
                    sf_dict = sf
            else:
                # MuSiQue supporting facts are in paragraphs with is_supporting
                gold_titles = [p["title"] for p in item.get("paragraphs", []) if p.get("is_supporting")]
                # We record the matching titles and sentences directly from the built graph
                t_list, s_list = [], []
                for node in rec["nodes"]:
                    if node["doc_title"] in gold_titles:
                        t_list.append(node["doc_title"])
                        s_list.append(node["sent_idx"])
                sf_dict = {"title": t_list, "sent_id": s_list}
                
            records.append({
                "id": item.get("id", str(idx)),
                "question": question,
                "answer": item.get("answer", ""),
                "supporting_facts": sf_dict,
                "graph": rec
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(records, f)

    print(f"Successfully saved {len(records)} KGs to {args.output}")
