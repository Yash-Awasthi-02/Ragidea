"""
PATHFINDER Experiment — Step 2: Loader for 2WikiMultihopQA and MuSiQue
========================================================================
Loads 2WikiMultihopQA / MuSiQue validation split and converts records to the
standard PATHFINDER knowledge graph record format used by 01_build_kg.py.
"""

import os
import argparse
import pickle
from datasets import load_dataset
from 01_build_kg import KGBuilder, MODEL_NAME

def load_2wiki_records(split="validation", max_samples=None):
    print(f"Loading 2WikiMultihopQA ({split})...")
    # Load 2WikiMultihopQA dataset
    ds = load_dataset("2wiki_multihop_qa", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds

def load_musique_records(split="validation", max_samples=None):
    print(f"Loading MuSiQue ({split})...")
    # Load MuSiQue dataset
    ds = load_dataset("musique", "ans", split=split)
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
    for idx, item in enumerate(ds):
        # Format candidate docs from dataset sample
        question = item["question"]
        if "context" in item:
            docs = [{"title": ctx[0], "sentences": ctx[1]} for ctx in item["context"]]
        else:
            docs = []

        rec = builder.build(question, docs)
        if rec is not None:
            rec["id"] = item.get("id", str(idx))
            rec["answer"] = item.get("answer", "")
            rec["supporting_facts"] = item.get("supporting_facts", {"title": [], "sent_id": []})
            records.append(rec)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(records, f)

    print(f"Successfully saved {len(records)} KGs to {args.output}")
