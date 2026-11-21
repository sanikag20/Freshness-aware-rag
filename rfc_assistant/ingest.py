"""Q3 ingestion: validate, clean, chunk and version the RFC corpus.

Pandas is used instead of Spark: the corpus is 242 documents / ~19 MB / ~12k chunks,
three orders of magnitude below where a single machine stops being adequate, and the
whole pipeline runs in under a minute. The stage structure (validate -> quarantine ->
clean -> chunk -> freshness) mirrors the Spark pipeline from Q1 one-to-one, so it
could be ported back if the corpus grew to the full 9,000-RFC archive.

Freshness model: an RFC is never edited in place; it is *obsoleted* by a successor.
A document is current on `as_of` when it was published on or before `as_of` and no
RFC that obsoletes it had been published by `as_of`. Chunk ids follow the harness
convention: "RFC9112#c4".
"""
import hashlib
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data")

CHUNK_COLS = ["chunk_id", "doc_id", "rfc", "title", "status", "tier", "effective_date",
              "obsoleted_by", "section", "text", "content_hash"]
STATUS_TIER = {"INTERNET STANDARD": "standards", "DRAFT STANDARD": "standards",
               "PROPOSED STANDARD": "standards", "BEST CURRENT PRACTICE": "standards",
               "INFORMATIONAL": "informational", "EXPERIMENTAL": "experimental",
               "HISTORIC": "historic"}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"])}
_HEADING = re.compile(r"^((?:\d+(?:\.\d+)*)|(?:Appendix [A-Z](?:\.\d+)*))\.?\s{1,2}(\S.*)$")
_SKIP_SECTIONS = re.compile(r"references|acknowledg|authors?'? address|full copyright|"
                            r"table of contents|index$|contributors", re.I)
MAX_WORDS = 180


def load_catalog():
    with open(os.path.join(HERE, "data", "catalog.json"), encoding="utf-8") as fh:
        return json.load(fh)


def parse_date(s):
    m = re.match(r"(?:(\d{1,2}) )?([A-Z][a-z]+) (\d{4})", s or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    return f"{m.group(3)}-{_MONTHS[m.group(2)]:02d}-{int(m.group(1) or 1):02d}"


def clean_rfc_text(raw: str) -> str:
    """Strip BOM, running page headers, '[Page N]' footers and form feeds."""
    pages = raw.lstrip("﻿").split("\f")
    kept = []
    for pi, page in enumerate(pages):
        lines = page.split("\n")
        if pi > 0:
            while lines and not lines[0].strip():
                lines.pop(0)
            if lines and re.search(r"RFC ?\d+", lines[0]) and len(lines[0]) < 90:
                lines.pop(0)
        lines = [ln for ln in lines if not re.search(r"\[Page \d+\]\s*$", ln)]
        kept.extend(ln.rstrip() for ln in lines)
    return "\n".join(kept)


def validate(catalog) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for num, meta in catalog["rfcs"].items():
        path = os.path.join(RAW, f"rfc{int(num)}.txt")
        text = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        violation = None
        if not text.strip():
            violation = "missing_or_empty_text"
        elif not meta.get("title", "").strip():
            violation = "missing_title"
        elif parse_date(meta.get("date")) is None:
            violation = "unparseable_date"
        elif meta.get("status", "UNKNOWN") not in STATUS_TIER:
            violation = "unknown_status"
        elif len(text.split()) < 300:
            violation = "implausibly_short_document"
        april1 = bool(re.search(r"\b1 April (?:19|20)\d{2}\b", text[:2500]))
        rows.append({**meta, "rfc": int(num), "text": text, "violation": violation,
                     "effective_date": parse_date(meta.get("date")),
                     "tier": "april1" if april1 else STATUS_TIER.get(meta.get("status"), "unknown")})
    df = pd.DataFrame(rows).sort_values("rfc").reset_index(drop=True)
    dup = df.duplicated("rfc", keep="first")
    df.loc[dup & df["violation"].isna(), "violation"] = "duplicate_rfc_number"
    valid = df[df["violation"].isna()].drop(columns=["violation"])
    quarantine = df[df["violation"].notna()].drop(columns=["text"])
    return valid, quarantine


def chunk_rfc(rec: dict) -> list:
    cleaned = clean_rfc_text(rec["text"])
    doc = f"RFC{rec['rfc']}"
    header = f"RFC {rec['rfc']}: {rec['title']}"
    sections, cur, cur_name = [], [], "Front matter"
    in_toc = False
    for ln in cleaned.split("\n"):
        m = _HEADING.match(ln)
        if re.match(r"^\s*Table of Contents\s*$", ln):
            in_toc = True
            continue
        if m and not ln.startswith(" "):
            in_toc = False
            if cur:
                sections.append((cur_name, "\n".join(cur)))
            cur, cur_name = [], f"{m.group(1)} {m.group(2)}"
        elif not in_toc:
            cur.append(ln)
    if cur:
        sections.append((cur_name, "\n".join(cur)))
    out = []
    k = 0
    for name, body in sections:
        if _SKIP_SECTIONS.search(name):
            continue
        words = " ".join(body.split()).split()
        if not words:
            continue
        for i in range(0, len(words), MAX_WORDS):
            part = " ".join(words[i:i + MAX_WORDS])
            text = f"{header}\n## {name}\n{part}"
            out.append(dict(zip(CHUNK_COLS, [
                f"{doc}#c{k}", doc, rec["rfc"], rec["title"], rec["status"], rec["tier"],
                rec["effective_date"], ",".join(str(x) for x in rec["obsoleted_by"]),
                name, text, hashlib.sha1(text.encode()).hexdigest()])))
            k += 1
    return out


def build_chunks(valid: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([c for r in valid.to_dict("records") for c in chunk_rfc(r)],
                        columns=CHUNK_COLS)


def current_docs(catalog, as_of: str, tiers=None) -> set:
    """Doc ids current on as_of under the obsolescence rule (optionally tier-limited)."""
    pub = {int(n): parse_date(m.get("date")) for n, m in catalog["rfcs"].items()}
    cur = set()
    for n, meta in catalog["rfcs"].items():
        n = int(n)
        if pub[n] is None or pub[n] > as_of:
            continue
        superseded = any(pub.get(ob) is not None and pub[ob] <= as_of
                         for ob in meta.get("obsoleted_by", []))
        if not superseded:
            cur.add(f"RFC{n}")
    return cur


def main():
    catalog = load_catalog()
    valid, quarantine = validate(catalog)
    chunks = build_chunks(valid)
    chunks.to_parquet(os.path.join(OUT, "knowledge_chunks.parquet"), index=False)
    quarantine.to_csv(os.path.join(OUT, "quarantine.csv"), index=False)
    print("valid RFCs:", len(valid), "| quarantined:", len(quarantine))
    if len(quarantine):
        print(quarantine.groupby("violation").size().to_string())
    print("chunks:", len(chunks), "| tiers:", chunks.drop_duplicates("doc_id")["tier"].value_counts().to_dict())
    print("current on 1999-01-01:", len(current_docs(catalog, "1999-01-01")),
          "| on 2024-01-01:", len(current_docs(catalog, "2024-01-01")))


if __name__ == "__main__":
    main()
