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

#-----
import ingest_new as ingest
#-----

TIER_W = {"standards": 1.0, "informational": 0.7, "experimental": 0.5,
          #-----
          "historic": 0.45, "legacy": 0.45, "april1": 0.15}
          #-----
SYNONYMS = {
    "email": "mail smtp message", "e-mail": "mail smtp", "website": "http server",
    "web": "http", "browser": "http user agent", "hostname": "domain name dns",
    "lookup": "resolution query", "secure": "tls security", "encrypt": "tls encryption",
    "port": "port number", "login": "authentication user", "file": "file transfer ftp",
}
MIN_COVERAGE = 0.6
#-----
# Switches used by evaluate.py for the ablations; the defaults are the shipped system.
CONFIG = {"successors": "index",   # "index": successors dated from the full rfc-index; "corpus": original rule
          "use_filter": True,      # only documents current on as_of can be retrieved or cited
          "use_tiers": True,       # source-tier weights (standards > informational > ... > april1)
          "min_cov": MIN_COVERAGE, # abstention threshold on idf-weighted coverage
          "answer": "v2",          # "v1": first extractor (single best sentence); "v2": see _answer_v2
          "scope_guard": True,     # v2: a question that names a protocol is answered only from that protocol's RFCs
          "trusted_evidence": True}  # v2: April-1st joke RFCs can be retrieved but never cited
#-----
_STATE = {}
#-----
_CUR = {}


def current(as_of):
    key = (as_of, CONFIG["successors"])
    if key not in _CUR:
        _CUR[key] = ingest.current_docs(_STATE["catalog"], as_of, successors=CONFIG["successors"])
    return _CUR[key]
#-----


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
    #-----
    if os.path.isdir(ingest.RAW) and any(f.startswith("rfc") for f in os.listdir(ingest.RAW)):
        valid, _q = ingest.validate(ingest.load_catalog())     # raw texts present: rebuild (<1 s)
        chunks = ingest.build_chunks(valid)
    else:                                                       # submission without raw/: use the table
        chunks = pd.read_parquet(os.path.join(HERE, "data", "knowledge_chunks_new.parquet"))
    #-----
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


#-----
def _fused(question, as_of, use_filter=None, use_tiers=None, n=50):
    use_filter = CONFIG["use_filter"] if use_filter is None else use_filter
    use_tiers = CONFIG["use_tiers"] if use_tiers is None else use_tiers
#-----
    chunks = _STATE["chunks"]
    q = _expand(question)
    if use_filter:
        #-----
        cur = current(as_of)
        #-----
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


#-----
def retrieve(question, as_of, k=5, use_filter=None, use_tiers=None):
#-----
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


#-----
def _answer_v1(question, as_of):
#-----
    fused = _fused(question, as_of)
    idxs = [i for i, _s in fused[:8]]
    if not idxs:
        return {"answer": "The corpus does not cover this.", "citations": [],
                "abstained": True, "confidence": 0.0}
    cov = _coverage(question, idxs[:3])
    #-----
    if cov < CONFIG["min_cov"]:
    #-----
        return {"answer": "The corpus does not cover this.", "citations": [],
                "abstained": True, "confidence": cov}
    chunks = _STATE["chunks"]
    #-----
    cur = current(as_of) if CONFIG["use_filter"] else {d for d in chunks["doc_id"].unique()}
    #-----
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


#-----
# ---------------------------------------------------------------------------------------------
# v2 answer path (written after the first evaluation on golden.jsonl; see README for the protocol)
# ---------------------------------------------------------------------------------------------
ABSTAIN = "The corpus does not cover this."
# protocol named in a question -> pattern its RFC title must match (scope guard)
SCOPE = [
    (r"\btcp\b|transmission control protocol", r"transmission control protocol|\btcp\b"),
    (r"\budp\b|user datagram", r"user datagram|\budp\b"),
    (r"\btls\b|\bssl\b|transport layer security", r"transport layer security|\btls\b"),
    (r"\bhttps?\b|hypertext", r"hypertext transfer|\bhttp\b"),
    (r"\bsmtp\b", r"simple mail transfer|\bsmtp\b"),
    (r"\be-?mail\b", r"\bmail\b|message format"),
    (r"\bdns\b|domain name", r"domain name|\bdns\b"),
    (r"\bntp\b|network time protocol", r"network time protocol|\bntp\b"),
    (r"\bipv6\b", r"\bipv6\b|version 6"),
    (r"\bipv4\b", r"^internet protocol$"),
    (r"\bftp\b|file transfer", r"file transfer"),
    (r"\boauth\b", r"oauth"),
    (r"\bwebsocket\b", r"websocket"),
    (r"\bjson\b", r"\bjson\b|javascript object notation"),
    (r"\bcookies?\b", r"state management|cookie"),
    (r"\btelnet\b", r"telnet"),
    (r"\bssh\b|secure shell", r"secure shell|\bssh\b"),
    (r"\bicmp\b", r"internet control message"),
    (r"\barp\b|address resolution", r"address resolution"),
]
_QTY = re.compile(r"\bhow (many|much|long|large|big)\b|\bmaximum\b|\bminimum\b|\bport\b|\blength\b|"
                  r"\bsize\b|\bnumber\b|\bvalue\b|\bstatus code\b|\bversion\b|\bbytes?\b|\bbits?\b", re.I)


def _scope_titles(question):
    """Title patterns for every protocol the question names (empty list = no constraint)."""
    return [tp for qp, tp in SCOPE if re.search(qp, question, re.I)]


def _in_scope(title, pats):
    return not pats or any(re.search(p, (title or "").lower()) for p in pats)


def _sentences_v2(question, text):
    """Score every body sentence of a chunk: idf overlap with the question, plus the section
    heading, a preference for sentences that state a number when the question asks for one,
    and a penalty for ASCII-art / ABNF fragments and very long run-ons."""
    q = {t for t in H.tokenize(question) if len(t) > 2}
    lines = text.split("\n")
    heading = lines[1][3:] if len(lines) > 1 and lines[1].startswith("## ") else ""
    ht = set(H.tokenize(heading))
    hb = 0.5 * sum(_idf(t) for t in q if _match(t, ht))
    quant = bool(_QTY.search(question))
    out = []
    for line in lines[2:]:
        for sent in H._SENT.split(line):
            sent = sent.strip()
            st = set(H.tokenize(sent))
            if len(st) < 3:
                continue
            clean = sum(c.isalnum() or c.isspace() for c in sent) / max(len(sent), 1)
            if clean < 0.8:                       # tables, figures, ABNF
                continue
            score = hb + sum(_idf(t) for t in q if _match(t, st))
            if quant and re.search(r"\d", sent):
                score *= 1.5
            if len(sent.split()) > 60:
                score *= 0.7
            out.append((score, sent))
    return out, heading


def _answer_v2(question, as_of):
    chunks = _STATE["chunks"]
    fused = _fused(question, as_of)
    idxs = [i for i, _s in fused]                 # filter the whole candidate list, then cut
    if CONFIG["trusted_evidence"]:
        idxs = [i for i in idxs if chunks["tier"].iloc[i] != "april1"]
    pats = _scope_titles(question) if CONFIG["scope_guard"] else []
    idxs = [i for i in idxs if _in_scope(chunks["title"].iloc[i], pats)][:10]
    if not idxs:
        return {"answer": ABSTAIN, "citations": [], "abstained": True, "confidence": 0.0}
    cov = _coverage(question, idxs[:3])
    if cov < CONFIG["min_cov"]:
        return {"answer": ABSTAIN, "citations": [], "abstained": True, "confidence": cov}
    cur = current(as_of) if CONFIG["use_filter"] else set(chunks["doc_id"])
    best = (0.0, None, None, None)
    for j, i in enumerate(idxs[:5]):
        if chunks["doc_id"].iloc[i] not in cur:
            continue
        scored, heading = _sentences_v2(question, chunks["text"].iloc[i])
        if not scored:
            continue
        top = max(scored)
        sc = top[0] * (1.0, 0.9, 0.8, 0.7, 0.6)[j]
        if sc > best[0]:
            best = (sc, i, scored, heading)
    score, i, scored, heading = best
    if i is None or score <= 0:
        return {"answer": ABSTAIN, "citations": [], "abstained": True, "confidence": cov}
    ranked = sorted(scored, reverse=True)
    picked = [ranked[0][1]]
    if len(ranked) > 1 and ranked[1][0] >= 0.6 * ranked[0][0]:   # a second supporting sentence
        picked.append(ranked[1][1])
    order = [snt for _sc, snt in scored]
    picked.sort(key=order.index)
    row = chunks.iloc[i]
    cite = f"[{row['doc_id']}, section {heading}]" if heading else f"[{row['doc_id']}]"
    return {"answer": " ".join(picked) + " " + cite, "citations": [row["chunk_id"]],
            "abstained": False, "confidence": cov}


def answer(question, as_of):
    """Return {"answer", "citations", "abstained", "confidence"}; every answer cites one chunk."""
    return (_answer_v2 if CONFIG["answer"] == "v2" else _answer_v1)(question, as_of)


#-----
class AnswerCache:
    """Production pattern measured in evaluate.ipynb: exact + semantic cache whose
    entries self-invalidate when a cited document stops being current on as_of."""

    def __init__(self, tau=0.95):
        self.tau = tau
        self.store, self.keys, self.vecs = {}, [], None

    def _valid(self, out, as_of):
        if not out["citations"]:
            return True
        #-----
        cur = current(as_of)
        #-----
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
