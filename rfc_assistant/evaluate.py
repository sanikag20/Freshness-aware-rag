"""Q3 evaluation: baseline vs. system, ablations, held-out test, latency and the cache pattern.

Usage (from the repo root or from rfc_assistant/):   python rfc_assistant/evaluate.py
Writes every table to rfc_assistant/outputs/ and prints them. Uses the unmodified harness harness.

Protocol
  * golden.jsonl (34 q) was labelled before the system was first run. The v2 answer path was
    written after looking at the v1 results on it, so golden numbers for v2 are optimistic.
  * golden_test.jsonl (14 q) was written and frozen (sha1 recorded in the README) before v2 existed
    and is evaluated once, here. It is the unbiased estimate.
"""
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import harness as H          # noqa: E402
import ingest_new as ingest   # noqa: E402
import system_new as S       # noqa: E402

OUT = os.path.join(HERE, "outputs")
N_DATE, LATER = "2024-06-01", "2026-11-01"
SHIPPED = dict(S.CONFIG)


def hardware():
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, Python {platform.python_version()}"


def set_config(**kw):
    S.CONFIG.clear()
    S.CONFIG.update(SHIPPED)
    S.CONFIG.update(kw)


# ---------------------------------------------------------------- baseline -------------------
def baseline_retrieve(question, as_of, k=5):
    """Plain BM25 over every chunk: no freshness filter, no tiers, no query expansion."""
    s = S._STATE["bm25"].scores(question)
    top = np.argsort(-s)[:k]
    return [S._STATE["chunks"]["chunk_id"].iloc[int(i)] for i in top if s[i] > 0]


def baseline_answer(question, as_of):
    """Best-overlap sentence of the top BM25 chunk; never abstains."""
    ids = baseline_retrieve(question, as_of, 1)
    if not ids:
        return {"answer": "", "citations": [], "abstained": False, "confidence": 0.0}
    row = S._STATE["chunks"].set_index("chunk_id").loc[ids[0]]
    sent, _sc = S._best_sentence(question, row["text"])
    return {"answer": f"{sent} [{row['doc_id']}]", "citations": ids, "abstained": False, "confidence": 0.0}


# ---------------------------------------------------------------- metrics --------------------
def stale_citation_rate(df, golden):
    """Domain metric: share of ANSWERED questions that cite an RFC which is not current on as_of
    under the full-index obsolescence rule (catches citing RFC 793 in 2024, RFC 8446 in 2026, ...)."""
    cat = S._STATE["catalog"]
    asof = {q["qid"]: q["as_of"] for q in golden}
    bad = n = 0
    for _, r in df.iterrows():
        if r["abstained"] or not r["citations"]:
            continue
        n += 1
        cur = ingest.current_docs(cat, asof[r["qid"]], successors="index")
        bad += any(H.doc_of(c) not in cur for c in r["citations"])
    return bad / n if n else float("nan")


def run(name, answer_fn, retrieve_fn, golden):
    s, df = H.evaluate_answers(answer_fn, golden, verbose=False)
    r, _rd = H.evaluate_retrieval(retrieve_fn, golden, verbose=False)
    bt = s["by_type"]
    return {"system": name, "accuracy": round(s["accuracy"], 3),
            "acc_answerable": bt.get("answerable"), "acc_stale_trap": bt.get("stale_trap"),
            "acc_unanswerable": bt.get("unanswerable"),
            "stale_answer_rate": round(s["stale_answer_rate"], 3),
            "stale_citation_rate": round(stale_citation_rate(df, golden), 3),
            "abstain_precision": round(s["abstain_precision"], 3) if s["abstain_precision"] == s["abstain_precision"] else None,
            "abstain_recall": round(s["abstain_recall"], 3),
            "citation_precision": round(s["citation_precision"], 3),
            "recall@1": round(r["recall@1"], 3), "recall@3": round(r["recall@3"], 3), "mrr@5": round(r["mrr@5"], 3)}, df


def latency(fn, golden, reps=4):
    qs = [(q["question"], q["as_of"]) for q in golden] * reps          # >= 100 timed calls
    lat = H.measure_latency(fn, qs, warmup=10)
    return {"n_calls": lat["n"], "p50_ms": round(lat["p50_ms"], 1), "p95_ms": round(lat["p95_ms"], 1)}


# ---------------------------------------------------------------- cache pattern ---------------
PARAPHRASES = {
    "What TCP port does an SMTP server listen on?": "Which port do SMTP servers listen on?",
    "What is the maximum length of a single DNS label?": "How long can one DNS label be?",
    "Which TCP port is used by default for http URIs when no port is given?": "What's the default port for an http URI?",
    "How large is an IPv6 address?": "How many bits are in an IPv6 address?",
    "What is the maximum number of characters allowed on one line of an email message?": "How long may a line in an email be?",
    "What is the maximum length of an SMTP command line?": "Max length of an SMTP command line?",
    "Which cookie attribute stops scripts from reading a cookie?": "Which attribute keeps JavaScript from reading a cookie?",
    "What port is assigned to the FTP control connection?": "Which port does FTP use for its control connection?",
    "How many bytes long is the random value in a TLS 1.3 ClientHello?": "Length of the ClientHello random in TLS 1.3?",
    "What does HTTP status code 422 mean?": "What is HTTP 422?",
    "Can a JSON text be a single value such as a number, or must it be an object or array?": "Is a bare number a valid JSON text?",
    "What is the reason phrase for HTTP status code 413?": "What's the name of status 413 in HTTP?",
    "What is the maximum length of an SMTP reply line?": "How long can an SMTP reply line be?",
    "What port does MQTT use?": "Which port is MQTT on?",
    "How does Encrypted Client Hello hide the server name in TLS?": "How does ECH hide the SNI in TLS?",
}


def query_log(golden, n_per_phase=150, seed=0):
    """Two phases: users asking on 2024-06-01, then on 2026-11-01 (after RFC 9846 obsoleted RFC 8446).
    Each query is a golden question asked on N_DATE or its paraphrase; repeats are frequent."""
    rng = random.Random(seed)
    base = sorted({q["question"] for q in golden if q["as_of"] == N_DATE})
    pool = base + [PARAPHRASES[q] for q in base if q in PARAPHRASES]
    log = []
    for phase, as_of in (("phase1", N_DATE), ("phase2", LATER)):
        for _ in range(n_per_phase):
            log.append({"phase": phase, "as_of": as_of, "query": rng.choice(pool)})
    return pd.DataFrame(log)


class NoRecheckCache(S.AnswerCache):
    """Ablation: the same cache without the freshness re-check on read."""
    def _valid(self, out, as_of):
        return True


def cache_experiment(golden):
    log = query_log(golden)
    rows = []
    fresh_cache = {}

    def fresh(q, a):
        if (q, a) not in fresh_cache:
            fresh_cache[(q, a)] = S.answer(q, a)
        return fresh_cache[(q, a)]

    # no cache
    ms = []
    for _, r in log.iterrows():
        t0 = time.perf_counter()
        S.answer(r["query"], r["as_of"])
        ms.append((time.perf_counter() - t0) * 1000)
    rows.append({"cache": "none", "tau": None, "hit_l1": 0.0, "hit_l2": 0.0,
                 "p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
                 "wrong_hit_rate": None, "stale_served": 0})
    cur_later = ingest.current_docs(S._STATE["catalog"], LATER)
    for label, cls in (("exact+semantic, re-check on read (shipped)", S.AnswerCache),
                       ("exact+semantic, NO re-check", NoRecheckCache)):
        for tau in (0.90, 0.95):
            cache = cls(tau=tau)
            recs = []
            for _, r in log.iterrows():
                t0 = time.perf_counter()
                out, tier = cache.get(r["query"], r["as_of"])
                if out is None:
                    out = S.answer(r["query"], r["as_of"])
                    cache.put(r["query"], r["as_of"], out)
                ms_ = (time.perf_counter() - t0) * 1000
                f = fresh(r["query"], r["as_of"])
                served = sorted({H.doc_of(c) for c in out["citations"]})
                recs.append({"tier": tier, "ms": ms_, "phase": r["phase"],
                             "wrong": tier != "miss" and (served != sorted({H.doc_of(c) for c in f["citations"]})
                                                          or out["abstained"] != f["abstained"]),
                             "stale": r["phase"] == "phase2" and any(d not in cur_later for d in served)})
            d = pd.DataFrame(recs)
            hits = d[d["tier"] != "miss"]
            rows.append({"cache": label, "tau": tau, "hit_l1": round(float((d["tier"] == "l1").mean()), 3),
                         "hit_l2": round(float((d["tier"] == "l2").mean()), 3),
                         "p50_ms": round(float(d["ms"].median()), 2), "p95_ms": round(float(d["ms"].quantile(0.95)), 2),
                         "wrong_hit_rate": round(float(hits["wrong"].mean()), 3) if len(hits) else 0.0,
                         "stale_served": int(d["stale"].sum())})
    return pd.DataFrame(rows), log


# ---------------------------------------------------------------- main -----------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    golden = H.load_golden(path=os.path.join(HERE, "golden.jsonl"))
    test = H.load_golden(path=os.path.join(HERE, "golden_test.jsonl"))
    t0 = time.perf_counter()
    S.build()
    build_s = time.perf_counter() - t0
    print(f"hardware: {hardware()} | build: {build_s:.1f} s | chunks: {len(S._STATE['chunks'])}")

    configs = [
        ("baseline: BM25, no filter, never abstains", None),
        ("v1: original system (in-corpus successors, v1 extractor)", dict(answer="v1", successors="corpus")),
        ("v2: final system", {}),
        ("v2 - scope guard", dict(scope_guard=False)),
        ("v2 - trusted evidence (joke RFCs citable)", dict(trusted_evidence=False)),
        ("v2 with in-corpus successors only", dict(successors="corpus")),
        ("v2 - freshness filter", dict(use_filter=False)),
        ("v2 - source tiers", dict(use_tiers=False)),
        ("v2 - abstention (min_cov = 0)", dict(min_cov=0.0)),
    ]
    rows, test_rows, lat_rows = [], [], []
    for name, cfg in configs:
        if cfg is None:
            a_fn, r_fn = baseline_answer, baseline_retrieve
        else:
            set_config(**cfg)
            a_fn, r_fn = S.answer, S.retrieve
        row, df = run(name, a_fn, r_fn, golden)
        rows.append(row)
        if name.startswith(("baseline", "v1:", "v2: final")):
            trow, tdf = run(name, a_fn, r_fn, test)
            test_rows.append(trow)
            lat_rows.append({"system": name, **latency(a_fn, golden)})
            tag = name.split(":")[0]
            df.to_csv(os.path.join(OUT, f"answers_golden_{tag}.csv"), index=False)
            tdf.to_csv(os.path.join(OUT, f"answers_test_{tag}.csv"), index=False)
    set_config()
    comp = pd.DataFrame(rows)
    test_df = pd.DataFrame(test_rows)
    lat_df = pd.DataFrame(lat_rows)
    for name, t in (("q3_golden_comparison", comp), ("q3_test_comparison", test_df), ("q3_latency", lat_df)):
        t.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)

    cache_df, log = cache_experiment(golden)
    cache_df.to_csv(os.path.join(OUT, "q3_cache.csv"), index=False)
    log.to_csv(os.path.join(OUT, "q3_query_log.csv"), index=False)

    fails = pd.read_csv(os.path.join(OUT, "answers_golden_v2.csv"))
    fails = fails[fails["correct"] < 1][["qid", "type", "question", "answer", "citations"]]
    fails.to_csv(os.path.join(OUT, "q3_failures_v2.csv"), index=False)

    meta = {"hardware": hardware(), "build_s": round(build_s, 2), "n_chunks": int(len(S._STATE["chunks"])),
            "n_docs": int(S._STATE["chunks"]["doc_id"].nunique()), "golden_n": len(golden), "test_n": len(test)}
    with open(os.path.join(OUT, "q3_run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n== Golden set (34 q): baseline vs systems and ablations ==\n", comp.to_string(index=False))
    print("\n== Held-out test set (14 q), evaluated once ==\n", test_df.to_string(index=False))
    print("\n== Latency (golden x4 = 136 timed calls after 10 warm-up) ==\n", lat_df.to_string(index=False))
    print("\n== Cache (300 queries; phase 2 is after RFC 9846 obsoleted RFC 8446) ==\n", cache_df.to_string(index=False))
    print("\n== v2 failures on golden ==\n", fails.to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = comp[comp["system"].str.match(r"(baseline|v1:|v2: final)")].set_index("system")
        ax = sel[["acc_answerable", "acc_stale_trap", "acc_unanswerable", "accuracy"]].T.plot.bar(figsize=(8, 4), rot=0)
        ax.set_ylabel("share correct"); ax.set_title("Q3 golden set: accuracy by question type")
        ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(OUT, "q3_accuracy_by_type.png"), dpi=150)
    except Exception as exc:          # plotting is optional
        print("plot skipped:", exc)
    return comp, test_df, lat_df, cache_df


if __name__ == "__main__":
    main()
