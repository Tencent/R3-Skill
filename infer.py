"""
infer.py · R3 two-stage skill retrieval (R3-Embedding recall + R3-Reranker rerank)
==================================================================================
Input : one/more queries + a skill library (jsonl)
Flow  : query --[R3-Embedding]--> recall top-N --[R3-Reranker]--> rerank --> top-K
Output: final top-K skills per query (id + reranker score)

Models load from ./models/r3-embedding and ./models/r3-reranker by default
(relative to this file), so the folder is fully self-contained.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# Train/inference-consistent prompts (do NOT change):
#   embedding-side query prefix (= R3-Embedding training INSTR)
EMB_INSTR = "Instruct: Given a user request, retrieve the agent skill that solves it.\nQuery: "
#   reranker-side instruct (= R3-Reranker training INSTRUCT; note: no "Query:" segment)
RR_INSTRUCT = "Given a user request, retrieve the agent skill that solves it."


def load_queries(query, query_file):
    if query:
        return [query]
    out = []
    for line in open(query_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            out.append(o["query"] if isinstance(o, dict) and "query" in o else line)
        except json.JSONDecodeError:
            out.append(line)
    return out


def truncate_body(text, tokenizer, body_max_tokens):
    """Match reranker training: truncate only the body (3rd field of 'name | desc | body')."""
    parts = text.split(" | ", 2)
    if len(parts) < 3:
        return text
    ids = tokenizer.encode(parts[2], add_special_tokens=False)
    if len(ids) <= body_max_tokens:
        return text
    return parts[0] + " | " + parts[1] + " | " + tokenizer.decode(ids[:body_max_tokens], skip_special_tokens=True)


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_model",    default=str(here / "models" / "r3-embedding"))
    ap.add_argument("--rerank_model", default=str(here / "models" / "r3-reranker"))
    ap.add_argument("--query",        help="single query string")
    ap.add_argument("--query_file",   help="multi-query file (txt one-per-line, or jsonl with 'query')")
    ap.add_argument("--skills",       required=True,
                    help="skill library jsonl, each line {\"id\": ..., \"text\": ...}")
    ap.add_argument("--recall_n",     type=int, default=20, help="embedding recall depth fed to the reranker")
    ap.add_argument("--top_k",        type=int, default=10, help="final number returned after rerank")
    ap.add_argument("--out",          help="output jsonl (prints to stdout if omitted)")
    ap.add_argument("--batch_size",   type=int, default=32)
    ap.add_argument("--emb_max_seq",  type=int, default=4096)
    ap.add_argument("--rr_max_seq",   type=int, default=4096)
    ap.add_argument("--body_max_tokens", type=int, default=4096)
    args = ap.parse_args()

    if not args.query and not args.query_file:
        ap.error("must give --query or --query_file")

    skills = [json.loads(l) for l in open(args.skills, encoding="utf-8") if l.strip()]
    sids = [s["id"] for s in skills]
    docs = [s["text"] for s in skills]
    queries = load_queries(args.query, args.query_file)
    print(f"[data] {len(skills)} skills, {len(queries)} queries", flush=True)

    recall_n = min(args.recall_n, len(skills))
    top_k = min(args.top_k, recall_n)

    n_gpu = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu > 0 else None
    multi = devices is not None and len(devices) > 1
    if multi:
        print(f"[device] multi-GPU {devices}", flush=True)

    # ===== Stage 1: R3-Embedding recall top-N =====
    print(f"[stage1] load embedding: {args.emb_model}", flush=True)
    emb = SentenceTransformer(args.emb_model, trust_remote_code=True,
                              device=(devices[0] if devices else None))
    emb.max_seq_length = args.emb_max_seq
    pool = emb.start_multi_process_pool(target_devices=devices) if multi else None

    def encode(texts):
        if pool:
            return emb.encode_multi_process(texts, pool=pool, batch_size=args.batch_size,
                                            normalize_embeddings=True)
        return emb.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True)

    skill_emb = encode(docs)
    query_emb = encode([EMB_INSTR + q for q in queries])
    if pool:
        emb.stop_multi_process_pool(pool)
    scores = query_emb @ skill_emb.T
    recall_idx = np.argsort(-scores, axis=1)[:, :recall_n]

    # ===== Stage 2: R3-Reranker rerank =====
    print(f"[stage2] load reranker: {args.rerank_model}", flush=True)
    rr = CrossEncoder(args.rerank_model, trust_remote_code=True,
                      device=(devices[0] if devices else None))
    rr.max_length = args.rr_max_seq
    doc_trunc = {sid: truncate_body(text, rr.tokenizer, args.body_max_tokens)
                 for sid, text in zip(sids, docs)}

    pairs = [(queries[qi], doc_trunc[sids[j]])
             for qi in range(len(queries)) for j in recall_idx[qi]]
    rpool = rr.start_multi_process_pool(target_devices=devices) if multi else None
    rr_flat = rr.predict(pairs, batch_size=args.batch_size, prompt=RR_INSTRUCT,
                         show_progress_bar=True, convert_to_numpy=True,
                         **({"pool": rpool} if rpool else {}))
    if rpool:
        rr.stop_multi_process_pool(rpool)
    rr_scores = rr_flat.reshape(len(queries), recall_n)

    # ===== output: rerank-sorted top-K =====
    results = []
    for qi, q in enumerate(queries):
        order = np.argsort(-rr_scores[qi])[:top_k]
        results.append({
            "query": q,
            "top_k": [{"id": sids[recall_idx[qi][o]], "score": float(rr_scores[qi][o])} for o in order],
        })

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[done] saved {len(results)} results to {args.out}", flush=True)
    else:
        for r in results:
            head = r["query"][:80] + ("..." if len(r["query"]) > 80 else "")
            print(f"\n=== Query: {head} ===")
            for rank, item in enumerate(r["top_k"], 1):
                print(f"  {rank:2d}. [{item['score']:+.3f}] {item['id']}")


if __name__ == "__main__":
    main()
