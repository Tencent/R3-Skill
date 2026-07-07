# Skill Is Not Document

A query-conditional benchmark (**R3-Skill**) and a two-stage retriever (**R3-Embedding + R3-Reranker**)
for LLM agent skill routing.

- 📄 Paper: [Skill Is Not Document: A Query-Conditional Benchmark and Two-Stage Retriever for LLM Agent Skill Routing](https://arxiv.org/abs/2606.03565)
- 🧩 This repository contains the code, docs, R3-Skill data, and a toy example. Model weights are
  hosted on Hugging Face — [R3-Embedding-0.6B](https://huggingface.co/tencent/R3-embedding-0.6b) /
  [R3-Rerank-0.6B](https://huggingface.co/tencent/R3-rerank-0.6b) — and should be placed under
  `models/` for reproduction.

---

## 1. Layout

```
├── infer.py              two-stage retrieval (embedding recall → reranker rerank)
├── reproduce.py          reproduce the paper's R3-Skill test numbers
├── example_skills.jsonl  toy skill library for a quick test
├── requirements.txt
├── README.md / CONTRIBUTING.md / CODE_OF_CONDUCT.md / LICENSE
├── models/               not tracked by Git; place downloaded model weights here
│   ├── r3-embedding/     bi-encoder    (fine-tuned from Qwen3-Embedding-0.6B)
│   └── r3-reranker/      cross-encoder (fine-tuned from Qwen3-Reranker-0.6B)
└── data/                 R3-Skill benchmark data
    ├── test.jsonl        5,696 test queries
    └── skills_test.jsonl  2,050 test-pool skills
```

Skill identifiers are released as opaque ids (`skill-00000`, …); each id maps consistently
across the query and skill-pool files. Train and test skill pools are disjoint.

The skill pool and query set draw on the following public resources:

- https://huggingface.co/datasets/ThakiCloud/SKILLRET
- https://huggingface.co/datasets/pipizhao/SkillRouter-Eval-Core
- https://gitcode.com/zhoukc42/AiSkill
- https://github.com/jnMetaCode/agency-agents-zh

---

## 2. Architecture

```
                    query
                      │
                      ▼
        ┌─────────────────────────────┐
        │  Stage 1 · R3-Embedding      │   bi-encoder, cosine over the whole pool
        │  (recall)                    │   prompt: "Instruct: ... \nQuery: "
        └─────────────┬───────────────┘
                      │  top-N candidates (default N=20)
                      ▼
        ┌─────────────────────────────┐
        │  Stage 2 · R3-Reranker       │   cross-encoder, scores each (query, skill) pair
        │  (rerank)                    │   prompt: "Given a user request, ..."
        └─────────────┬───────────────┘
                      │  reordered
                      ▼
                 top-K skills
```

- **Stage 1 (recall)** embeds the query and every skill independently and ranks by cosine
  similarity — cheap, scales to the full skill pool.
- **Stage 2 (rerank)** re-scores only the top-N candidates with a cross-encoder that reads the
  query and skill jointly, so it can judge query-conditional compatibility that a bi-encoder cannot.

---

## 3. Design notes

- **Why two stages.** A bi-encoder is fast but scores each skill in isolation; whether a *set* of
  skills should be retrieved together depends on the query (skill compatibility). The cross-encoder
  reranker repairs this on a small candidate set.
- **SKIP-as-resource.** LLM-rejected skill combinations are kept as negative supervision for the
  reranker during training, rather than discarded — this is the paper's core idea.

---

## 4. Modules

| File | Role |
|---|---|
| `infer.py` | End-to-end retrieval over an arbitrary skill library; CLI in / jsonl out. |
| `reproduce.py` | Loads `data/test.jsonl` + `data/skills_test.jsonl`, runs both stages, prints metrics next to the paper values. |
| `models/r3-embedding/` | `SentenceTransformer` bi-encoder. [Download](https://huggingface.co/tencent/R3-embedding-0.6b) and place here. |
| `models/r3-reranker/` | `CrossEncoder` cross-encoder. [Download](https://huggingface.co/tencent/R3-rerank-0.6b) and place here. |
| `data/` | R3-Skill test queries and the test skill pool. |

---

## 5. Quick start

```bash
pip install -r requirements.txt
```

Download the released model weights and place them under:

```bash
pip install -U huggingface_hub
hf download tencent/R3-embedding-0.6b --local-dir models/r3-embedding
hf download tencent/R3-rerank-0.6b --local-dir models/r3-reranker
```

### Smoke test
```bash
python3 infer.py --query "I need to compose music" --skills example_skills.jsonl --recall_n 6 --top_k 3
```
Expected top-1: `music-composer`.

### Reproduce paper numbers (R3-Skill test)
The R3-Skill data is included under `data/`. Run:

```bash
python3 reproduce.py
```
Prints embedding-only (Table 5) and embedding+reranker (Table 7) metrics next to the paper values.
Uses all GPUs visible via `CUDA_VISIBLE_DEVICES`.

---

## 6. Programming guide

### Retrieve over your own skills
```bash
python3 infer.py --query_file my_queries.txt --skills my_skills.jsonl --top_k 10 --out results.jsonl
```

**Skill library** (`--skills`), one JSON per line:
```json
{"id": "music-composer", "text": "name | description | body"}
```
The `text` field is recommended to follow `name | description | body` (matches training; the
reranker truncates only the `body` segment).

**Query input** — either `--query "..."` (single) or `--query_file FILE` (txt one-per-line, or
jsonl with a `query` field).

**Output** (`--out`), one JSON per line:
```json
{"query": "...", "top_k": [{"id": "...", "score": 0.83}, ...]}
```

### Embedding only (no reranker)
```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("models/r3-embedding", trust_remote_code=True)
qv = m.encode(["Instruct: Given a user request, retrieve the agent skill that solves it.\nQuery: " + q],
              normalize_embeddings=True)
```

### Key flags
| flag | default | meaning |
|---|---|---|
| `--recall_n` | 20 | embedding recall depth fed to the reranker |
| `--top_k` | 10 | final results after rerank |
| `--batch_size` | 32 | lower it on OOM |

---

## 7. Contributing & conduct

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 8. License

Apache License 2.0 (see [LICENSE](LICENSE)).

## 9. Questions about models & datasets

For any questions regarding the use of the released models or datasets, please contact the
author at **jawnrwen@tencent.com**.

## 10. Citation

```bibtex
@inproceedings{r3skill2026,
  title  = {Skill Is Not Document: A Query-Conditional Benchmark and Two-Stage Retriever for LLM Agent Skill Routing},
  author = {Wang, Zifei and Wen, Wei and Ji, Qiang and Qiao, Ruizhi and Sun, Xing},
  year   = {2026},
  url    = {https://arxiv.org/abs/2606.03565},
}
```
