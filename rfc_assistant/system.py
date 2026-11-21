"""Q3 query path: hybrid retrieval and grounded extractive answers over the RFC corpus.

Interfaces match the evaluation harness:
    build()
    retrieve(question, as_of, k) -> [chunk ids]
    answer(question, as_of)      -> {"answer", "citations", "abstained", "confidence"}

Design: BM25 + LSA dense vectors fused with reciprocal rank fusion; only documents
current on `as_of` (obsolescence rule) can be returned; source tiers downweight
informational / experimental / historic / April-1st documents; an idf-weighted
coverage test triggers abstention; answers are sentences extracted from the cited
chunk, so every factual claim carries a citation.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import harness as H

import ingest

TIER_W = {"standards": 1.0, "informational": 0.7, "experimental": 0.5,
          "historic": 0.45, "april1": 0.15}
SYNONYMS = {
    "email": "mail smtp message", "e-mail": "mail smtp", "website": "http server",
    "web": "http", "browser": "http user agent", "hostname": "domain name dns",
    "lookup": "resolution query", "secure": "tls security", "encrypt": "tls encryption",
    "port": "port number", "login": "authentication user", "file": "file transfer ftp",
}
MIN_COVERAGE = 0.6
_STATE = {}


def _expand(question):
    toks = H.tokenize(question, drop_stopwords=False)
    extra = [SYNONYMS[t] for t in toks if t in SYNONYMS]
    return question + " " + " ".join(dict.fromkeys(extra))


def _idf(t):
    idf = _STATE["bm25"].idf
    cands = [idf[v] for v in (t, t + "s", t.rstrip("s")) if v in idf]
    return min(cands) if cands else max(idf.values())


def _match(t, have):
    alts = {t} | set(H.tokenize(SYNONYMS.get(t, "")))
    return any(a in have or a + "s" in have or a.rstrip("s") in have for a in alts)


def build(embedder=None):
    chunks = pd.read_parquet(os.path.join(HERE, "data", "knowledge_chunks.parquet"))
    catalog = ingest.load_catalog()
    bm25 = H.BM25Retriever.from_chunks(chunks)
    if embedder is None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        embedder = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=2,
                            tokenizer=H.tokenize, preprocessor=lambda x: x, token_pattern=None),
            TruncatedSVD(n_components=256, random_state=0))
        embedder.fit(chunks["text"].tolist())
    mat = np.asarray(embedder.transform(chunks["text"].tolist()))
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    _STATE.update(chunks=chunks, catalog=catalog, bm25=bm25, embedder=embedder, matrix=mat,
                  tier_w=chunks["tier"].map(TIER_W).fillna(0.5).to_numpy())


def _embed_query(q):
    v = np.asarray(_STATE["embedder"].transform([q]))[0]
    return v / max(np.linalg.norm(v), 1e-12)


def _fused(question, as_of, use_filter=True, use_tiers=True, n=50):
    chunks = _STATE["chunks"]
    q = _expand(question)
    if use_filter:
        cur = ingest.current_docs(_STATE["catalog"], as_of)
        mask = chunks["doc_id"].isin(cur).to_numpy()
    else:
        mask = np.ones(len(chunks), dtype=bool)
    lists = []
    for scores in (_STATE["bm25"].scores(q), _STATE["matrix"] @ _embed_query(q)):
        s = np.where(mask, scores, -1.0)
        if use_tiers:
            s = s * _STATE["tier_w"]
        idx = np.argsort(-s)[:n]
        lists.append([(int(i), float(s[i])) for i in idx if s[i] > 0])
    sc = {}
    for lst in lists:
        for rank, (i, _s) in enumerate(lst):
            sc[i] = sc.get(i, 0.0) + 1.0 / (60 + rank + 1)
    return sorted(sc.items(), key=lambda kv: -kv[1])


def retrieve(question, as_of, k=5, use_filter=True, use_tiers=True):
    fused = _fused(question, as_of, use_filter, use_tiers)
    return [_STATE["chunks"]["chunk_id"].iloc[i] for i, _s in fused[:k]]


def _coverage(question, idxs):
    qt = [t for t in set(H.tokenize(question)) if len(t) > 2]
    if not qt:
        return 1.0
    have = set()
    for i in idxs:
        have |= set(H.tokenize(_STATE["chunks"]["text"].iloc[i]))
    num = den = 0.0
    for t in qt:
        w = _idf(t)
        den += w
        if _match(t, have):
            num += w
    return num / den if den else 0.0


def _best_sentence(question, text):
    q = {t for t in H.tokenize(question) if len(t) > 2}
    best, bs = "", 0.0
    for li, line in enumerate(text.split("\n")):
        if li == 0 or line.startswith("## "):
            continue
        for s in H._SENT.split(line):
            s = s.strip()
            st = set(H.tokenize(s))
            if len(st) < 3:
                continue
            score = sum(_idf(t) for t in q if _match(t, st))
            if re.search(r"\d", s):
                score *= 1.3
            if score > bs:
                best, bs = s, score
    return best, bs


def answer(question, as_of):
    fused = _fused(question, as_of)
    idxs = [i for i, _s in fused[:8]]
    if not idxs:
        return {"answer": "The corpus does not cover this.", "citations": [],
                "abstained": True, "confidence": 0.0}
    cov = _coverage(question, idxs[:3])
    if cov < MIN_COVERAGE:
        return {"answer": "The corpus does not cover this.", "citations": [],
                "abstained": True, "confidence": cov}
    chunks = _STATE["chunks"]
    cur = ingest.current_docs(_STATE["catalog"], as_of)
    best = ("", 0.0, None)
    for j, i in enumerate(idxs[:4]):
        row = chunks.iloc[i]
        if row["doc_id"] not in cur:
            continue
        s, sc = _best_sentence(question, row["text"])
        sc *= (1.0, 0.9, 0.8, 0.7)[j]
        if sc > best[1]:
            best = (s, sc, i)
    sent, score, i = best
    if not sent or score <= 0:
        return {"answer": "The corpus does not cover this.", "citations": [],
                "abstained": True, "confidence": cov}
    row = chunks.iloc[i]
    return {"answer": f"{sent} [{row['doc_id']}]", "citations": [row["chunk_id"]],
            "abstained": False, "confidence": cov}


class AnswerCache:
    """Production pattern measured in evaluate.ipynb: exact + semantic cache whose
    entries self-invalidate when a cited document stops being current on as_of."""

    def __init__(self, tau=0.95):
        self.tau = tau
        self.store, self.keys, self.vecs = {}, [], None

    def _valid(self, out, as_of):
        if not out["citations"]:
            return True
        cur = ingest.current_docs(_STATE["catalog"], as_of)
        return all(H.doc_of(c) in cur for c in out["citations"])

    def get(self, question, as_of):
        k = H.normalize(question)
        if k in self.store and self._valid(self.store[k], as_of):
            return dict(self.store[k]), "l1"
        if self.keys:
            qv = _embed_query(question)
            sims = self.vecs @ qv
            j = int(np.argmax(sims))
            if sims[j] >= self.tau and self._valid(self.store[self.keys[j]], as_of):
                return dict(self.store[self.keys[j]]), "l2"
        return None, "miss"

    def put(self, question, as_of, out):
        k = H.normalize(question)
        if k in self.store:
            return
        self.store[k] = dict(out)
        self.keys.append(k)
        v = _embed_query(question)[None, :]
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])


if __name__ == "__main__":
    t0 = time.time()
    build()
    print("build s:", round(time.time() - t0, 1))
    for q, a in [("What port does SMTP use?", "2024-01-01"),
                 ("What does HTTP status code 404 mean?", "2024-01-01"),
                 ("How do I bake a cake?", "2024-01-01")]:
        print(q, "->", answer(q, a))
