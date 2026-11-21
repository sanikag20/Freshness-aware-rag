"""Latency benchmark for the retrieval configurations and the final answer path: at least 100 timed calls after a
warm-up of 10 (harness.measure_latency). The configurations are the notebook's 2.1.3 ablation rows
(same functions, from northbeam_system.py) plus the final answer() for 2.2.6.

Run from the repo root:   python latency_benchmark.py
Writes outputs/q2_latency_100calls.csv.
"""
import os
import platform

import numpy as np
import pandas as pd

import harness as H
import northbeam_system as S

S.build()
kb = S._STATE["kb"]
DEV = H.load_golden("dev")
ALL = np.ones(len(kb.live), dtype=bool)


def bm25_only(q, a, k=5):
    s = kb.bm25.scores(q)
    top = np.argsort(-s)[:k]
    return [kb.live["chunk_id"].iloc[int(i)] for i in top if s[i] > 0]


def dense_only(q, a, k=5):
    hits = S._dense_rank(kb, S.expand_query(q), ALL, n=k)
    return [kb.live["chunk_id"].iloc[i] for i, _s in hits]


def hybrid(q, a, k=5, use_filter=True):
    fused, _c = S.hybrid_idx(kb, q, a, use_filter)
    return [kb.live["chunk_id"].iloc[i] for i, _s in fused[:k]]


queries = [(q["question"], q["as_of"]) for q in DEV] * 2          # 132 timed calls
rows = []
for name, fn in {"BM25": bm25_only, "dense": dense_only,
                 "hybrid": lambda q, a: hybrid(q, a, use_filter=False),
                 "hybrid+filter": hybrid,
                 "answer() (final system, 2.2.6)": S.answer}.items():
    lat = H.measure_latency(fn, queries, warmup=10)
    rows.append({"configuration": name, "timed_calls": lat["n"],
                 "p50_ms": round(lat["p50_ms"], 1), "p95_ms": round(lat["p95_ms"], 1)})

hw = f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}"
df = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/q2_latency_100calls.csv", index=False)
print("hardware:", hw)
print("10 warm-up calls, then", rows[0]["timed_calls"], "timed calls per configuration (66 dev questions x 2)")
print(df.to_string(index=False))
