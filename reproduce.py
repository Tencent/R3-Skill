"""
reproduce.py · reproduce the R3-Skill test numbers from the paper
=================================================================
Embedding-only  -> paper Table 5 (R3-Embedding row)
Embedding+Reranker -> paper Table 7 (R3-Reranker row)

Uses data/test.jsonl (queries + GT skill_ids) and data/skills_test.jsonl (skill pool).
Self-contained: models load from ./models/ by default.

  python3 reproduce.py            # both stages
  python3 reproduce.py --stage emb     # embedding only
  python3 reproduce.py --stage rerank  # embedding + reranker
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

EMB_INSTR = "Instruct: Given a user request, retrieve the agent skill that solves it.\nQuery: "
RR_INSTRUCT = "Given a user request, retrieve the agent skill that solves it."

# paper reference numbers (R3-Skill test): Table 5 (emb) and Table 7 R3-Reranker row (rerank)
PAPER = {
    "emb": {
        "hit@1": 0.7207,
        "ndcg@5": 0.8006, "ndcg@10": 0.8160, "ndcg@15": 0.8205,
        "recall@5": 0.8722, "recall@10": 0.9177, "recall@15": 0.9338,
        "comp@5": 0.8722, "comp@10": 0.9177, "comp@15": 0.9338,
        "set_compat": 0.2812,
    },
    "rerank": {
        "hit@1": 0.7521,
        "ndcg@5": 0.8008, "ndcg@10": 0.8173, "ndcg@15": 0.8254,
        "recall@5": 0.8481, "recall@10": 0.8969, "recall@15": 0.9264,
        "comp@5": 0.8481, "comp@10": 0.8969, "comp@15": 0.9264,
        "set_compat": 0.3188,
    },
}

METRIC_KEYS = ["hit@1", "ndcg@5", "ndcg@10", "ndcg@15",
               "recall@5", "recall@10", "recall@15",
               "comp@5", "comp@10", "comp@15", "set_compat"]


def metrics(gt_list, scores, sids):
    """Full R3-Skill metrics (@5/@10/@15) matching the paper protocol."""
    sid_arr = np.array(sids)
    K = min(15, scores.shape[1])
    top = np.argpartition(-scores, K - 1, axis=1)[:, :K]
    row = np.arange(len(gt_list))[:, None]
    top = top[row, np.argsort(-scores[row, top], axis=1)]
    out = {k: 0.0 for k in METRIC_KEYS}
    n_multi = 0
    for qi, gtl in enumerate(gt_list):
        gt = set(gtl)
        tk = sid_arr[top[qi]]
        if tk[0] in gt:
            out["hit@1"] += 1.0
        for k in (5, 10, 15):
            dcg = sum(1.0 / math.log2(r + 2) for r, s in enumerate(tk[:k]) if s in gt)
            idcg = sum(1.0 / math.log2(r + 2) for r in range(min(k, len(gt))))
            out[f"ndcg@{k}"] += dcg / idcg if idcg > 0 else 0.0
            out[f"recall@{k}"] += len(gt & set(tk[:k])) / max(1, len(gt))
            out[f"comp@{k}"] += len(gt & set(tk[:k])) / min(len(gt), k)
        if len(gt) >= 2:
            n_multi += 1
            if gt.issubset(set(tk[:len(gt)])):
                out["set_compat"] += 1.0
    n = len(gt_list)
    for k in out:
        out[k] = out[k] / n_multi if k == "set_compat" else out[k] / n
    return out


def truncate_body(text, tok, n):
    p = text.split(" | ", 2)
    if len(p) < 3:
        return text
    ids = tok.encode(p[2], add_special_tokens=False)
    return text if len(ids) <= n else p[0] + " | " + p[1] + " | " + tok.decode(ids[:n], skip_special_tokens=True)


def show(name, m, ref):
    print(f"\n=== {name} ===")
    print(f"{'metric':<12}{'ours':>10}{'paper':>10}")
    for k in METRIC_KEYS:
        print(f"{k:<12}{m[k]*100:>10.2f}{ref[k]*100:>10.2f}")


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_model",    default=str(here / "models" / "r3-embedding"))
    ap.add_argument("--rerank_model", default=str(here / "models" / "r3-reranker"))
    ap.add_argument("--queries", default=str(here / "data" / "test.jsonl"))
    ap.add_argument("--skills",  default=str(here / "data" / "skills_test.jsonl"))
    ap.add_argument("--stage", choices=["emb", "rerank", "both"], default="both")
    ap.add_argument("--recall_n", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    skills = [json.loads(l) for l in open(args.skills, encoding="utf-8") if l.strip()]
    sids = [s["id"] for s in skills]
    docs = [s["text"] for s in skills]
    gt_list = [q["skill_ids"] for q in queries]
    q_texts = [q["query"] for q in queries]
    print(f"[data] {len(queries)} queries, {len(skills)} skills", flush=True)

    n_gpu = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu > 0 else None
    multi = devices is not None and len(devices) > 1
    print(f"[device] {'multi-GPU '+str(devices) if multi else (devices[0] if devices else 'cpu')}", flush=True)

    emb = SentenceTransformer(args.emb_model, trust_remote_code=True,
                              device=(devices[0] if devices else None))
    emb.max_seq_length = 4096
    pool = emb.start_multi_process_pool(target_devices=devices) if multi else None

    def encode(texts):
        if pool:
            return emb.encode_multi_process(texts, pool=pool, batch_size=args.batch_size,
                                            normalize_embeddings=True)
        return emb.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True)

    d = encode(docs)
    qv = encode([EMB_INSTR + q for q in q_texts])
    if pool:
        emb.stop_multi_process_pool(pool)
    emb_scores = qv @ d.T

    if args.stage in ("emb", "both"):
        show("Embedding only (paper Table 5)", metrics(gt_list, emb_scores, sids), PAPER["emb"])

    if args.stage in ("rerank", "both"):
        recall_n = min(args.recall_n, len(skills))
        recall_idx = np.argsort(-emb_scores, axis=1)[:, :recall_n]
        rr = CrossEncoder(args.rerank_model, trust_remote_code=True,
                          device=(devices[0] if devices else None))
        rr.max_length = 4096
        dt = {sids[j]: truncate_body(docs[j], rr.tokenizer, 4096) for j in range(len(sids))}
        pairs = [(q_texts[qi], dt[sids[j]]) for qi in range(len(queries)) for j in recall_idx[qi]]
        rpool = rr.start_multi_process_pool(target_devices=devices) if multi else None
        flat = rr.predict(pairs, batch_size=args.batch_size, prompt=RR_INSTRUCT,
                          show_progress_bar=True, convert_to_numpy=True,
                          **({"pool": rpool} if rpool else {}))
        if rpool:
            rr.stop_multi_process_pool(rpool)
        rr_scores = flat.reshape(len(queries), recall_n)
        # scatter reranker scores back to full matrix (non-recalled = -inf)
        full = np.full_like(emb_scores, -1e9)
        for qi in range(len(queries)):
            full[qi, recall_idx[qi]] = rr_scores[qi]
        show("Embedding + Reranker (paper Table 7)", metrics(gt_list, full, sids), PAPER["rerank"])


if __name__ == "__main__":
    main()
