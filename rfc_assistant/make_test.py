"""Writes rfc_assistant/golden_test.jsonl: a HELD-OUT set written after the first evaluation run on golden.jsonl and
BEFORE any change to the system. It is evaluated once, after the system is frozen, and never used for
tuning. Same labelling method as make_golden.py."""
import json, os
import ingest_new as ingest
N = "2024-06-01"
Q = [
 ("answerable", "What is the largest DNS message that can be sent over UDP?", N, ["RFC1035"], "512 octets", None, ""),
 ("answerable", "Which TCP port is the default for https URIs?", N, ["RFC9110"], "port 443", None, ""),
 ("answerable", "What value must the Sec-WebSocket-Version header field have?", N, ["RFC6455"], "13", None, ""),
 ("answerable", "Which port does a DNS name server listen on?", N, ["RFC1035"], "port 53", None, ""),
 ("answerable", "How long is an IPv4 address?", N, ["RFC791"], "32 bits", None, ""),
 ("answerable", "Which grant_type value requests an OAuth 2.0 client credentials grant?", N, ["RFC6749"], "client_credentials", None, ""),
 ("answerable", "Which HTTP status code says the request was sent to a server that cannot produce a response for that origin?", N,
  ["RFC9110", "RFC9113", "RFC9114"], "421", None, ""),
 ("answerable", "Are IPv6 extension headers examined by nodes along a packet's delivery path?", "2010-01-01", ["RFC2460"],
  "not examined or processed", None, "pair of the stale trap"),
 ("stale_trap", "Are IPv6 extension headers examined by nodes along a packet's delivery path?", N, ["RFC8200"],
  "not processed, inserted, or deleted", "not examined or processed", "RFC 2460 -> 8200"),
 ("unanswerable", "Which port does the XMPP client protocol use?", N, [], None, None, "outside the corpus"),
 ("unanswerable", "How does DNS over HTTPS encode queries?", N, [], None, None, "outside the corpus"),
 ("unanswerable", "What does HTTP status code 425 Too Early mean?", N, [], None, None, "RFC 8470 is not in the corpus"),
 ("unanswerable", "How does the Telnet randomly-lose option work?", N, [], None, None, "only an April-1st joke RFC (748)"),
 ("unanswerable", "What does the TCP Urgent Pointer field contain?", N, [], None, None,
  "RFC 793 obsoleted by RFC 9293 (Aug 2022), not in the corpus"),
]

def main():
    cat = ingest.load_catalog()
    valid, _q = ingest.validate(cat)
    text = {f"RFC{r}": " ".join(t.split()).lower() for r, t in zip(valid.rfc, valid.text)}
    rows, problems, cnt = [], [], {}
    for t, q, a, gold, key, stale, note in Q:
        cnt[t] = cnt.get(t, 0) + 1
        qid = f"Q3T-{ {'answerable': 'A', 'stale_trap': 'S', 'unanswerable': 'U'}[t] }{cnt[t]:02d}"
        cur = ingest.current_docs(cat, a)
        for g in gold:
            if g not in cur:
                problems.append(f"{qid}: {g} not current on {a}")
            if key and key.lower() not in text.get(g, ""):
                problems.append(f"{qid}: key not in {g}")
        rows.append({"qid": qid, "question": q, "as_of": a, "type": t, "gold_doc_ids": gold,
                     "answer_key": key, "stale_key": stale, "note": note})
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_test.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(len(rows), "held-out questions |", cnt, "| label problems:", problems or "none")

if __name__ == "__main__":
    main()
