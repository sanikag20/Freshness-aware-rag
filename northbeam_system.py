"""
northbeam_system.py -- your final system's entry point.

Keep this file importable from the folder you submit. It must work like this, with no manual steps:

    import northbeam_system
    northbeam_system.build()                                              # called once
    northbeam_system.answer("How long do I have ...", "2026-10-29")       # one call per question
    northbeam_system.retrieve("How long do I have ...", "2026-10-29", 5)  # optional, for retrieval metrics

Rules
  * Read data only through the harness loaders, and apply ALL events in kb_updates.jsonl inside build().
  * No network calls are required. If you call an LLM, keep the extractive path working without it.
  * build() plus answering the 66 development questions must finish well within 15 minutes on a CPU.
  * Answers must come from retrieved evidence: no hard-coded question -> answer/document mappings.

Move the pieces you wrote in northbeam_pipeline.ipynb into this file (or import them from your own
modules that you include in the zip).
"""
import harness as H

_STATE = {}

#------
import hashlib
import html
import os
import re
import time

import numpy as np
import pandas as pd

VALID_STATUS = {"active", "retired", "draft"}
VALID_CATEGORY = {"refund", "shipping", "product", "account", "promo", "other"}
CHUNK_COLS = ["chunk_id", "doc_id", "article_id", "version", "title", "category", "status",
              "source", "effective_date", "kind", "text", "content_hash"]

SYNONYMS = {
    "britain": "united kingdom", "england": "united kingdom", "uk": "united kingdom",
    "u.k": "united kingdom", "scotland": "united kingdom", "wales": "united kingdom",
    "germany": "european union eu", "france": "european union eu", "spain": "european union eu",
    "italy": "european union eu", "ireland": "european union eu", "netherlands": "european union eu",
    "america": "united states", "usa": "united states", "u.s": "united states", "us": "united states",
    "boots": "footwear", "boot": "footwear", "sandals": "footwear", "sandal": "footwear",
    "sneakers": "footwear", "shoes": "footwear", "shoe": "footwear", "hiker": "footwear",
    "daypack": "packs bags", "backpack": "packs bags", "rucksack": "packs bags",
    "jacket": "apparel clothing", "fleece": "apparel clothing", "pants": "apparel clothing",
    "hoodie": "apparel clothing", "shirt": "apparel clothing",
    "tent": "camping gear", "tents": "camping gear",
    "stove": "water cooking", "cook": "water cooking cookware", "filter": "water cooking",
    "watch": "electronics", "headlamp": "electronics", "gps": "electronics",
    "send": "return", "money": "refund", "delete": "deleting deletion", "cancel": "cancelling",
    "match": "matching", "adjustment": "adjustments", "cost": "costs price",
    "price": "cost costs", "minimum": "over orders", "store": "retailer retailers",
    "dollar": "1 spent", "wash": "washing machine", "fix": "troubleshooting",
    "much": "cost costs price fee", "kingdom": "uk", "states": "us", "union": "eu",
}
EXT_STOP = H.STOPWORDS | set("""how long take takes much many get see need still offer really per
now value buy bought buying live day days way find lost give also just even right after until any
some lower found another above gets won what's whats you your do does did""".split())
TRUST = {"official": 1.0, "ticket": 0.5, "community": 0.25}

PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PII_PHONE = re.compile(r"(\(\d{3}\)\s?\d{3}-\d{4})|(\b\d{3}-\d{3}-\d{4}\b)|(\+44\s?\d{2}\s?\d{4}\s?\d{4})")
PII_ORDER = re.compile(r"\bNB-\d{7}\b")
_W = r"[A-Z][\wÀ-ɏ'’-]*"
_NAME_CTX = [
    re.compile(rf"\b(?:Hi|Hello|Hey)[,!]?\s+({_W}(?:\s+{_W})?)\s+here\b"),
    re.compile(rf"\b({_W}(?:\s+{_W})?)\s+here\b"),
    re.compile(rf"\bPackage for\s+({_W}(?:\s+{_W})*)"),
    re.compile(rf"\bCustomer:\s*({_W},\s*{_W})"),
    re.compile(rf"\bwas for\s+({_W}(?:\s+{_W})*)"),
    re.compile(rf"\b({_W}(?:\s+{_W})+)\s+called\b"),
    re.compile(rf"\bSpoke (?:with|to)\s+({_W}(?:\s+{_W})*)"),
    re.compile(rf"\b(?:This is|I am|I'm|My name is|name is)\s+({_W}(?:\s+{_W})*)"),
    re.compile(rf"\b(?:Hi|Hello|Hey|Dear|Thanks|Thank you|Regards|Best|Cheers|help)[,]?\s+({_W}(?:\s+{_W})?)\s*[,.!;)]"),
    re.compile(rf",\s+({_W}(?:\s+{_W})?)\s+(?:again|speaking)\b"),
]
_NOT_NAMES = set("""january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday northbeam outfitters order history start return
summit down jacket trailrunner gtx basecamp tent lumen headlamp orbit gps watch alpine pack daytrip
cascade water filter ember stove halo ridgecrest hiker drift shell i it the a is""".split())

_MONEY = re.compile(r"(?:C\$|\$|€|£)\s?\d+(?:[.,]\d+)?")
_PCT = re.compile(r"\d+(?:\.\d+)?\s?%")
_DUR = re.compile(r"\d+(?:\.\d+)?\s?(?:business |working )?(?:day|week|month|year|minute|hour)s?", re.I)
_NUMTOK = re.compile(r"\d+(?:[.,:]\d+)?")
_NEG = {"no", "not", "nothing", "never", "cannot", "longer"}
_POLAR = _NEG | {"included", "free", "yes"}
_QUANT = re.compile(r"\bhow (many|much|long|often)\b|\bdays?\b|\bcost|price|fee|value|minimum|"
                    r"when\b|rating|weigh|torso|sleep", re.I)
PII_REQUEST = re.compile(
    r"(email|e-mail|phone|contact|address)\b.*\b(customer|someone|person|who|complained|reported)"
    r"|\b(customer|someone|person|who|complained|reported)\b.*(email|e-mail|phone|number|address)", re.I)
ABSTAIN = {"answer": "I don't have enough information in the current knowledge base to answer that.",
           "citations": [], "abstained": True}
_REGION_ALIAS = {"uk": {"united", "kingdom"}, "us": {"united", "states"},
                 "eu": {"european", "union"}}
_COLLECTION = re.compile(r"(?:belongs to the|part of our)\s+([a-z &]+?)\s+(?:line|collection)", re.I)
MIN_COVERAGE = 0.62


def validate_articles(raw: pd.DataFrame):
    """Pandas version of the notebook's Spark validation (same rules, same counts)."""
    df = raw.copy()
    df["status_norm"] = df["status"].fillna("").str.strip().str.lower()
    df["eff_ok"] = pd.to_datetime(df["effective_date"], format="%Y-%m-%d", errors="coerce").notna()
    viol = pd.Series([None] * len(df), index=df.index, dtype=object)
    viol[df["article_id"].isna() | (df["article_id"].astype(str).str.strip() == "")] = "missing_article_id"
    viol[viol.isna() & (df["body"].isna() | (df["body"].astype(str).str.strip() == ""))] = "empty_body"
    viol[viol.isna() & ~df["eff_ok"]] = "impossible_date"
    viol[viol.isna() & ~df["status_norm"].isin(VALID_STATUS)] = "invalid_status"
    viol[viol.isna() & ~df["category"].fillna("").str.strip().str.lower().isin(VALID_CATEGORY)] = "invalid_category"
    ok = df[viol.isna()].copy()
    ok["_hash"] = (ok["title"].fillna("") + "\x00" + ok["body"].fillna("") + "\x00" +
                   ok["effective_date"].fillna("") + ok["status_norm"] + ok["source"].fillna("")
                   ).map(lambda s: hashlib.sha1(s.encode()).hexdigest())
    exact_dup = ok.duplicated(["article_id", "version", "_hash"], keep="first")
    viol[ok.index[exact_dup]] = "duplicate_key_exact"
    ok2 = ok[~exact_dup]
    nuniq = ok2.groupby(["article_id", "version"])["_hash"].nunique()
    conflict_keys = set(nuniq[nuniq > 1].index)
    mask = ok2.set_index(["article_id", "version"]).index.isin(conflict_keys)
    viol[ok2.index[mask]] = "duplicate_key_conflicting_content"
    valid = df[viol.isna()].copy()
    valid["status"] = valid["status_norm"]
    valid = valid.drop(columns=["status_norm", "eff_ok"])
    quar = df[viol.notna()].drop(columns=["status_norm", "eff_ok"]).copy()
    quar["violation"] = viol[viol.notna()]
    return valid, quar


def clean_text(body: str) -> str:
    t = body or ""
    t = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", t)
    t = re.sub(r"(?i)<\s*h[1-6]\s*>", "\n## ", t)
    t = re.sub(r"(?i)</\s*h[1-6]\s*>", "\n", t)
    t = re.sub(r"(?i)<\s*(p|li|div|ul|ol)\s*>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"(?im)(was this article helpful\?.*?$)|(northbeam outfitters help center\.?.*?$)|"
               r"(last reviewed by the customer care team\.?)|(^-{2,}\s*$)", " ", t)
    lines = []
    for ln in t.split("\n"):
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln and ln != "---":
            lines.append(ln)
    return "\n".join(lines)


def find_pii_spans(text: str):
    spans = []
    for rx, typ in ((PII_EMAIL, "EMAIL"), (PII_PHONE, "PHONE"), (PII_ORDER, "ORDER_ID")):
        for m in rx.finditer(text or ""):
            spans.append((m.start(), m.end(), typ))
    taken = [(s, e) for s, e, _ in spans]
    for rx in _NAME_CTX:
        for m in rx.finditer(text or ""):
            g0 = m.span(1)[0]
            for wm in re.finditer(_W, m.group(1)):
                ws, we = g0 + wm.start(), g0 + wm.end()
                if wm.group(0).lower() in _NOT_NAMES:
                    continue
                if any(ws < e and s < we for s, e in taken):
                    continue
                taken.append((ws, we))
                spans.append((ws, we, "NAME"))
    return sorted(set(spans))


def mask_pii(text: str) -> str:
    out = text or ""
    for s, e, typ in sorted(find_pii_spans(out), key=lambda x: -x[0]):
        out = out[:s] + f"[{typ}]" + out[e:]
    return re.sub(r"\[NAME\](\s*\[NAME\])+", "[NAME]", out)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _mk_chunk(cid, doc, aid, ver, title, cat, status, src, eff, kind, text):
    return dict(zip(CHUNK_COLS, [cid, doc, aid, int(ver), title, cat, status, src, eff,
                                 kind, text, _sha1(text)]))


def chunk_article(rec: dict) -> list:
    cleaned = clean_text(rec.get("body") or "")
    title = (rec.get("title") or "").strip()
    sections, cur = [], []
    for ln in cleaned.split("\n"):
        if ln.startswith("## "):
            if cur:
                sections.append("\n".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append("\n".join(cur))
    doc = f"{rec['article_id']}@v{int(rec['version'])}"
    out = []
    for k, sec in enumerate(sections):
        text = f"{title}\n{sec}" if title else sec
        out.append(_mk_chunk(f"{doc}#c{k}", doc, rec["article_id"], rec["version"], title,
                             rec.get("category"), rec.get("status"), rec.get("source"),
                             rec.get("effective_date"), "article", text))
    return out


def chunk_ticket(rec: dict) -> list:
    msg = mask_pii(str(rec.get("customer_message") or ""))
    rep = mask_pii(str(rec.get("agent_reply") or ""))
    text = f"Support ticket\nCustomer: {msg}\nAgent: {rep}"
    tid = rec["ticket_id"]
    return [_mk_chunk(f"{tid}#c0", tid, tid, 1, "Support ticket", rec.get("category"),
                      "resolved", "ticket", str(rec.get("created_at") or "")[:10], "ticket", text)]


_EMB = {"model": None, "kind": None}


def _fit_embedder(texts):
    if os.environ.get("HW5_USE_ST") == "1":
        from sentence_transformers import SentenceTransformer
        _EMB["model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _EMB["kind"] = "sentence-transformers/all-MiniLM-L6-v2"
        return
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, tokenizer=H.tokenize,
                        preprocessor=lambda x: x, token_pattern=None),
        TruncatedSVD(n_components=256, random_state=0))
    pipe.fit(texts)
    _EMB["model"] = pipe
    _EMB["kind"] = "TF-IDF(1-2grams)+TruncatedSVD(256), i.e. LSA"


def embed_texts(texts: list) -> np.ndarray:
    if _EMB["model"] is None:
        _fit_embedder(list(texts))
    if _EMB["kind"].startswith("sentence"):
        v = np.asarray(_EMB["model"].encode(list(texts), show_progress_bar=False))
    else:
        v = np.asarray(_EMB["model"].transform(list(texts)))
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _resolve_current(records, as_of: str, deleted=frozenset()) -> set:
    best = {}
    for r in records:
        aid = r["article_id"]
        if aid in deleted or str(r.get("status") or "").strip().lower() != "active":
            continue
        eff = str(r.get("effective_date") or "")
        if not eff or eff > as_of:
            continue
        key = (eff, int(r["version"]))
        if aid not in best or key > best[aid]:
            best[aid] = key
    return {f"{aid}@v{key[1]}" for aid, key in best.items()}


REQUIRED_EVENT_FIELDS = {"upsert": ["article_id", "version", "body", "title"],
                         "delete": ["article_id"]}


class KnowledgeBase:
    def __init__(self, chunks: pd.DataFrame, records: list):
        self.chunks = chunks.reset_index(drop=True)
        self.records = {(r["article_id"], int(r["version"])): r for r in records}
        self.deleted, self.processed = set(), set()
        self.vectors = {}
        self.embed_count = 0
        self.quarantined_events = []
        self._embed_new(self.chunks)
        self._rebuild_indexes()

    def _embed_new(self, chunk_df):
        todo = [h for h in chunk_df["content_hash"].unique() if h not in self.vectors]
        if todo:
            texts = chunk_df.drop_duplicates("content_hash").set_index("content_hash")["text"]
            for h, v in zip(todo, embed_texts([texts[h] for h in todo])):
                self.vectors[h] = v
            self.embed_count += len(todo)
        return len(todo)

    def _rebuild_indexes(self):
        self.live = self.chunks[~self.chunks["article_id"].isin(self.deleted)].reset_index(drop=True)
        self.bm25 = H.BM25Retriever.from_chunks(self.live)
        self.matrix = (np.stack([self.vectors[h] for h in self.live["content_hash"]])
                       if len(self.live) else np.zeros((0, 1)))

    def apply_event(self, ev: dict) -> dict:
        t0 = time.perf_counter()
        eid = ev.get("event_id")
        row = {"event_id": eid, "op": ev.get("op"), "status": "applied",
               "rechunked": 0, "reembedded": 0, "ms": 0.0}
        if eid in self.processed:
            row["status"] = "skipped_duplicate"
            row["ms"] = (time.perf_counter() - t0) * 1000
            return row
        art = ev.get("article") or {}
        need = REQUIRED_EVENT_FIELDS.get(ev.get("op"))
        if need is None or any(not str(art.get(f) or "").strip() for f in need):
            self.quarantined_events.append({**ev, "violation": "missing_required_fields"})
            self.processed.add(eid)
            row["status"] = "quarantined"
            row["ms"] = (time.perf_counter() - t0) * 1000
            return row
        if ev["op"] == "delete":
            self.deleted.add(art["article_id"])
        else:
            self.records[(art["article_id"], int(art["version"]))] = art
            doc = f"{art['article_id']}@v{int(art['version'])}"
            new_chunks = pd.DataFrame(chunk_article(art), columns=CHUNK_COLS)
            self.chunks = pd.concat([self.chunks[self.chunks["doc_id"] != doc], new_chunks],
                                    ignore_index=True)
            row["rechunked"] = len(new_chunks)
            row["reembedded"] = self._embed_new(new_chunks)
            self.deleted.discard(art["article_id"])
        self._rebuild_indexes()
        self.processed.add(eid)
        row["ms"] = (time.perf_counter() - t0) * 1000
        return row

    def current_docs(self, as_of: str) -> set:
        cur = _resolve_current(self.records.values(), as_of, self.deleted)
        cur |= set(self.live.loc[self.live["kind"] == "ticket", "doc_id"])
        return cur


def expand_query(question: str) -> str:
    toks = H.tokenize(question, drop_stopwords=False)
    extra = [SYNONYMS[t] for t in toks if t in SYNONYMS]
    return question + " " + " ".join(dict.fromkeys(extra))


def _idf(kb, t):
    idf = kb.bm25.idf
    cands = [idf[v] for v in (t, t + "s", t.rstrip("s"), t + "ing") if v in idf]
    return min(cands) if cands else max(idf.values())


def _match(t, have):
    alts = {t} | set(H.tokenize(SYNONYMS.get(t, "")))
    for a in alts:
        if a in have or a + "s" in have or a.rstrip("s") in have or a + "ing" in have:
            return True
        if len(a) >= 5 and any(h.startswith(a[:5]) and abs(len(h) - len(a)) <= 3 for h in have):
            return True
    return False


def _allowed_mask(kb, as_of, use_filter=True):
    if not use_filter:
        return np.ones(len(kb.live), dtype=bool)
    return kb.live["doc_id"].isin(kb.current_docs(as_of)).to_numpy()


def _ranked(kb, scores, mask, n=50, trust=False):
    s = np.where(mask, scores, -1.0)
    if trust:
        s = s * kb.live["source"].map(TRUST).fillna(1.0).to_numpy()
    idx = np.argsort(-s)[:n]
    return [(int(i), float(s[i])) for i in idx if s[i] > 0]


def _bm25_rank(kb, query, mask, n=50, trust=False):
    return _ranked(kb, kb.bm25.scores(query), mask, n, trust)


def _dense_rank(kb, query, mask, n=50, trust=False):
    qv = embed_texts([query])[0]
    return _ranked(kb, kb.matrix @ qv, mask, n, trust)


def _title_bonus(kb, question, idx):
    q = set(H.tokenize(question))
    t = set(H.tokenize(kb.live["title"].iloc[idx] or ""))
    b = 0.0015 * sum(min(_idf(kb, w), 6.0) for w in q if _match(w, t))
    if kb.live["category"].iloc[idx] == "refund" and re.search(r"\brefund|return", question, re.I):
        b += 0.004
    return b


def _rrf_fuse(lists, kb=None, question=None, k_rrf=60):
    sc = {}
    for lst in lists:
        for rank, (i, _s) in enumerate(lst):
            sc[i] = sc.get(i, 0.0) + 1.0 / (k_rrf + rank + 1)
    if kb is not None and question is not None:
        sc = {i: v + _title_bonus(kb, question, i) for i, v in sc.items()}
    return sorted(sc.items(), key=lambda kv: -kv[1])


def hybrid_idx(kb, question, as_of, use_filter=True):
    q = expand_query(question)
    mask = _allowed_mask(kb, as_of, use_filter)
    fused = _rrf_fuse([_bm25_rank(kb, q, mask, trust=use_filter),
                       _dense_rank(kb, q, mask, trust=use_filter)], kb, question)
    chased = None
    if use_filter and fused and re.search(r"\breturn|refund|restock", question, re.I):
        prod = None
        for i, _s in fused[:3]:
            m = _COLLECTION.search(kb.live["text"].iloc[i])
            if m:
                prod, chased = i, m.group(1)
                break
        if prod is None:
            rare = [t for t in H.tokenize(question) if t in kb.bm25.idf and _idf(kb, t) >= 4.5]
            if rare:
                for i, _s in _bm25_rank(kb, " ".join(rare), mask, n=5):
                    m = _COLLECTION.search(kb.live["text"].iloc[i])
                    if m:
                        prod, chased = i, m.group(1)
                        break
        if chased is not None:
            q2 = q + " returns refunds policy " + chased
            fused2 = _rrf_fuse([_bm25_rank(kb, q2, mask, trust=True),
                                _dense_rank(kb, q2, mask, trust=True)], kb, q2)
            fused = [(prod, fused[0][1])] + [f for f in fused2 if f[0] != prod]
    return fused, chased


def _facts(sent):
    toks = set(H.tokenize(sent, drop_stopwords=False))
    return set(_NUMTOK.findall(sent)) | (toks & _POLAR)


def typed_facts(sent):
    return {"money": set(_MONEY.findall(sent)), "pct": set(_PCT.findall(sent)),
            "dur": set(f.lower().rstrip("s") for f in _DUR.findall(sent))}


def _qtokens(question):
    return {t for t in H.tokenize(question) if t not in EXT_STOP or t in SYNONYMS}


def _wanted_fact(question):
    if re.search(r"how long|how many (?:business )?days|\bdays?\b|\bminutes?\b|\bweeks?\b", question, re.I):
        return "dur"
    if re.search(r"how much|\bcost|price|fee\b|\bminimum\b|order value", question, re.I):
        return "money"
    return None


def _coverage(kb, question, idxs):
    qt = [t for t in set(H.tokenize(question)) if t not in EXT_STOP]
    if not qt:
        return 1.0
    have = set()
    for i in idxs:
        have |= set(H.tokenize(kb.live["text"].iloc[i]))
    num = den = 0.0
    for t in qt:
        w = _idf(kb, t)
        den += w
        if _match(t, have):
            num += w
    return num / den if den else 0.0


def _body_sentences(text):
    out = []
    for li, line in enumerate((text or "").split("\n")):
        if li == 0 or line.startswith("## "):
            continue
        out.extend(s.strip() for s in H._SENT.split(line) if s.strip())
    return out


def _score_sentences(kb, question, text, quantitative, title, dw=True):
    ttl = set(H.tokenize(title or "")) if dw else set()
    q = _qtokens(question)
    heading = next((ln[3:] for ln in (text or "").split("\n") if ln.startswith("## ")), "")
    ht = set(H.tokenize(heading))
    hb = 0.6 * sum(_idf(kb, t) for t in q if _match(t, ht))
    scored = []
    for s in _body_sentences(text):
        st = set(H.tokenize(s))
        if not st or s.lower().startswith("this article explains"):
            continue
        score = hb + sum(_idf(kb, t) * (0.3 if t in ttl else 1.0) for t in q if _match(t, st))
        if len(st) < 4:
            score *= 0.5
        if quantitative and _facts(s):
            score *= 1.5
        want = _wanted_fact(question)
        if want and typed_facts(s)[want]:
            score *= 1.4
        scored.append((score, s))
    return scored


def _extract_from_doc(kb, question, doc_id):
    rows = kb.live[kb.live["doc_id"] == doc_id]
    quant = bool(_QUANT.search(question))
    title = rows.iloc[0]["title"]
    sent, score, cid, pool = "", 0.0, None, []
    for _, r in rows.iterrows():
        scored = _score_sentences(kb, question, r["text"], quant, title)
        pool.extend(scored)
        if scored:
            sc, s = max(scored)
            if sc > score:
                sent, score, cid = s, sc, r["chunk_id"]
    if sent and quant and not any(typed_facts(sent).values()):
        withf = [(sc, s) for sc, s in pool if any(typed_facts(s).values()) and s != sent]
        if withf:
            sent = sent + " " + max(withf)[1]
    return sent, score, cid


def _alias(tokens):
    out = set(tokens)
    for t in tokens:
        out |= _REGION_ALIAS.get(t, set())
    return out


def _is_peer_out_of_scope(question_exp, r1, r2):
    t1 = _alias(set(H.tokenize(r1["title"] or "")))
    t2 = _alias(set(H.tokenize(r2["title"] or "")))
    if len(t1 & t2) < 2:
        return False
    q = _alias(set(H.tokenize(question_exp)))
    only1, only2 = t1 - t2, t2 - t1
    return bool(only1 & q) and not (only2 & q)


def _answer_impl(kb, question, as_of, min_cov=MIN_COVERAGE, defenses=True):
    if defenses and PII_REQUEST.search(question):
        return {"answer": "I can't share customers' personal contact details.",
                "citations": [], "abstained": True, "confidence": 0.0}
    fused, chased = hybrid_idx(kb, question, as_of, use_filter=True)
    idxs = [i for i, _s in fused[:8]]
    if defenses:
        idxs = [i for i in idxs if kb.live["source"].iloc[i] != "community"]
    if not idxs:
        return {**ABSTAIN, "confidence": 0.0}
    cov = _coverage(kb, question, idxs[:3])
    if cov < min_cov:
        return {**ABSTAIN, "confidence": cov}
    rows = kb.live.iloc[idxs]
    docs = rows.drop_duplicates("doc_id")
    current = kb.current_docs(as_of)

    def citable(doc_id):
        return doc_id in current and doc_id in set(rows["doc_id"])

    off = docs[docs["source"] == "official"]
    if len(off) >= 2 and chased is None:
        q_exp = expand_query(question)
        r1 = off.iloc[0]
        s1, sc1, c1 = _extract_from_doc(kb, question, r1["doc_id"])
        s1t = set(H.tokenize(s1))
        q_in_s1 = sorted({t for t in _qtokens(question) if _match(t, s1t)},
                         key=lambda t: (-_idf(kb, t), t))
        for _, r2 in off.iloc[1:4].iterrows():
            if r2["article_id"] == r1["article_id"] or not citable(r2["doc_id"]):
                continue
            if _is_peer_out_of_scope(q_exp, r1, r2):
                continue
            s2, sc2, c2 = _extract_from_doc(kb, question, r2["doc_id"])
            s2t = set(H.tokenize(s2))
            shared = (s1t & s2t) - EXT_STOP
            tf1, tf2 = typed_facts(s1), typed_facts(s2)
            numeric = any(tf1[k] and tf2[k] and tf1[k] != tf2[k] for k in tf1) and len(shared) >= 2
            neg1 = bool(set(H.tokenize(s1, False)) & _NEG)
            neg2 = bool(set(H.tokenize(s2, False)) & _NEG)
            polarity = (neg1 != neg2) and any(_match(t, s2t) for t in q_in_s1[:2])
            if sc1 > 0 and sc2 >= 0.5 * sc1 and (numeric or polarity) and citable(r1["doc_id"]):
                ans = (f"Official sources disagree on this. {s1} [{r1['doc_id']}] "
                       f"However: {s2} [{r2['doc_id']}]")
                return {"answer": ans, "citations": [c1, c2],
                        "abstained": False, "confidence": cov}

    cites = []
    if chased is not None:
        pol = docs[(docs["category"] == "refund") & (docs["source"] == "official")]
        if len(pol):
            cites.append(docs.iloc[0]["chunk_id"])
            target = pol.iloc[0]
        else:
            target = docs.iloc[0]
        sent, score, cid = _extract_from_doc(kb, question, target["doc_id"])
    else:
        target, score = None, 0.0
        for j, (_, r) in enumerate(docs.head(3).iterrows()):
            if not citable(r["doc_id"]):
                continue
            _s, sc_j, _c = _extract_from_doc(kb, question, r["doc_id"])
            ttl = set(H.tokenize(r["title"] or ""))
            sc_j += 0.5 * sum(_idf(kb, t) for t in _qtokens(question) if _match(t, ttl))
            sc_j *= (1.0, 0.85, 0.75)[j]
            if sc_j > score:
                target, score = r, sc_j
        sent, score, cid = ("", 0.0, None) if target is None else \
            _extract_from_doc(kb, question, target["doc_id"])
    if target is None or not sent or score <= 0 or not citable(target["doc_id"]):
        return {**ABSTAIN, "confidence": cov}
    if defenses:
        sent = mask_pii(sent)
    cites.append(cid)
    return {"answer": f"{sent} [{target['doc_id']}]", "citations": cites,
            "abstained": False, "confidence": cov}
#------


def build():
    """Load data, validate, mask, chunk, apply kb_updates.jsonl, build indexes. Called once."""
    # TODO: replace this naive baseline with your Q1/Q2 pipeline.
    #------
    valid, _quar = validate_articles(H.load_articles())
    tickets = H.load_tickets()
    chunks = pd.DataFrame(
        [c for r in valid.to_dict("records") for c in chunk_article(r)] +
        [c for r in tickets.to_dict("records") for c in chunk_ticket(r)], columns=CHUNK_COLS)
    kb = KnowledgeBase(chunks, valid.to_dict("records"))
    event_log = [kb.apply_event(e) for e in H.load_updates()]
    _STATE["kb"] = kb
    _STATE["event_log"] = event_log
    _STATE["retrieve"] = lambda q, a, k=5: [kb.live["chunk_id"].iloc[i]
                                            for i, _s in hybrid_idx(kb, q, a, True)[0][:k]]
    _STATE["answer"] = lambda q, a: _answer_impl(kb, q, a)
    #------


def retrieve(question: str, as_of: str, k: int = 5) -> list:
    """Return chunk ids, best first."""
    if not _STATE:
        build()
    return _STATE["retrieve"](question, as_of, k)


def answer(question: str, as_of: str) -> dict:
    """Return {"answer": str, "citations": [chunk ids], "abstained": bool, "confidence": float}."""
    if not _STATE:
        build()
    return _STATE["answer"](question, as_of)
