"""Writes rfc_assistant/golden.jsonl and checks every label against the corpus.

Labelling method: every answer_key is copied verbatim from the section of the gold RFC that
answers the question; for stale traps the stale_key is the wording of the superseded RFC.
gold_doc_ids are the RFCs that state the fact AND are current on as_of under the obsolescence
rule. Unanswerable questions are either outside the corpus, only "answered" by an April-1st joke
RFC, or asked on a date when the only source had been obsoleted by an RFC that is not in the corpus
or did not exist yet. The questions were written before the system was run on them and were not
edited afterwards.
"""
import json, os, re
import ingest_new as ingest

N = "2024-06-01"   # default "today" for questions about the current specs
Q = [
 # ---- answerable: current specs -------------------------------------------------------------
 ("answerable", "What TCP port does an SMTP server listen on?", N, ["RFC5321"], "port 25", None, ""),
 ("answerable", "What is the maximum length of a single DNS label?", N, ["RFC1035"], "63 octets", None, ""),
 ("answerable", "What is the maximum length of a full domain name?", N, ["RFC1035"], "255 octets", None, ""),
 ("answerable", "Which TCP port is used by default for http URIs when no port is given?", N, ["RFC9110"], "port 80", None, ""),
 ("answerable", "Which port is assigned to the Network Time Protocol?", N, ["RFC5905"], "123", None, ""),
 ("answerable", "How large is an IPv6 address?", N, ["RFC8200"], "128 bits", None, ""),
 ("answerable", "What is the maximum number of characters allowed on one line of an email message?", N, ["RFC5322"], "998 characters", None, ""),
 ("answerable", "What is the maximum length of an SMTP command line?", N, ["RFC5321"], "512 octets", None, ""),
 ("answerable", "Which OAuth 2.0 error code means the authorization grant is invalid, expired or revoked?", N, ["RFC6749"], "invalid_grant", None, ""),
 ("answerable", "Which cookie attribute stops scripts from reading a cookie?", N, ["RFC6265"], "httponly", None, ""),
 ("answerable", "What port is assigned to the FTP control connection?", N, ["RFC959"], "port 21", None, ""),
 ("answerable", "How many bytes long is the random value in a TLS 1.3 ClientHello?", N, ["RFC8446"], "32 bytes", None, ""),
 ("answerable", "What does HTTP status code 422 mean?", N, ["RFC9110"], "unprocessable content", None, "added in RFC 9110"),
 ("answerable", "Can I assign HTTP status code 418 to my own application?", N, ["RFC9110"], "reserved", None,
  "hard case: the only other source is the April-1st joke RFC 2324"),
 ("answerable", "What status code does a server return to accept a WebSocket opening handshake?", N, ["RFC6455"], "101", None, ""),
 # ---- answerable on an older date (the older spec was current then) -------------------------
 ("answerable", "Can a JSON text be a single value such as a number, or must it be an object or array?", "2010-01-01", ["RFC4627"], "serialized object or array", None, "pair of Q3-S01"),
 ("answerable", "What is the reason phrase for HTTP status code 413?", "2010-01-01", ["RFC2616"], "request entity too large", None, "pair of Q3-S02"),
 ("answerable", "What is the maximum length of an SMTP reply line?", "1995-01-01", ["RFC821"], "512 characters", None, "pair of Q3-S03"),
 ("answerable", "What NTP version number goes in the version field of an NTP packet?", "2000-01-01", ["RFC1305"], "currently three", None, "pair of Q3-S04"),
 ("answerable", "What does the TCP Data Offset field count?", "2010-01-01", ["RFC793"], "32 bit words", None, "pair of Q3-U08"),
 # ---- stale traps: same question, the spec changed ------------------------------------------
 ("stale_trap", "Can a JSON text be a single value such as a number, or must it be an object or array?", N, ["RFC8259"], "serialized value", "serialized object or array", "RFC 4627 -> 7159 -> 8259"),
 ("stale_trap", "What is the reason phrase for HTTP status code 413?", N, ["RFC9110"], "content too large", "request entity too large", "RFC 2616 -> 7231 -> 9110"),
 ("stale_trap", "What is the maximum length of an SMTP reply line?", N, ["RFC5321"], "512 octets", "512 characters", "RFC 821 -> 2821 -> 5321"),
 ("stale_trap", "What NTP version number goes in the version field of an NTP packet?", N, ["RFC5905"], "currently 4", "currently three", "RFC 1305 -> 5905"),
 # ---- unanswerable --------------------------------------------------------------------------
 ("unanswerable", "What port does MQTT use?", N, [], None, None, "outside the corpus"),
 ("unanswerable", "What is the SameSite cookie attribute for?", N, [], None, None, "cookie RFC is in the corpus, SameSite is not"),
 ("unanswerable", "Which response header turns on HTTP Strict Transport Security?", N, [], None, None, "outside the corpus"),
 ("unanswerable", "Can I send IP datagrams using carrier pigeons?", N, [], None, None, "only an April-1st joke RFC (1149) covers it"),
 ("unanswerable", "What does the BREW method do in a coffee pot protocol?", N, [], None, None, "only an April-1st joke RFC (2324) covers it"),
 ("unanswerable", "What is the default port for PostgreSQL?", N, [], None, None, "outside the corpus"),
 ("unanswerable", "How does Encrypted Client Hello hide the server name in TLS?", N, [], None, None, "outside the corpus"),
 ("unanswerable", "What does the TCP Data Offset field count?", N, [], None, None,
  "RFC 793 was obsoleted by RFC 9293 (Aug 2022), which is not in the corpus"),
 ("unanswerable", "How many bytes long is the random value in a TLS 1.3 ClientHello?", "2026-11-01", [], None, None,
  "RFC 8446 was obsoleted by RFC 9846 (Jul 2026), which is not in the corpus"),
 ("unanswerable", "What does HTTP status code 422 mean?", "2010-01-01", [], None, None,
  "422 was not defined in any HTTP RFC current in 2010 within the corpus (RFC 2616 has no 422)"),
]

def main():
    cat = ingest.load_catalog()
    valid, _q = ingest.validate(cat)
    text = {f"RFC{r}": " ".join(t.split()).lower() for r, t in zip(valid.rfc, valid.text)}
    rows, problems, counters = [], [], {}
    for t, q, a, gold, key, stale, note in Q:
        counters[t] = counters.get(t, 0) + 1
        prefix = {"answerable": "A", "stale_trap": "S", "unanswerable": "U"}[t]
        qid = f"Q3-{prefix}{counters[t]:02d}"
        cur = ingest.current_docs(cat, a)
        for g in gold:
            if g not in cur:
                problems.append(f"{qid}: gold {g} not current on {a}")
            if key and key.lower() not in text.get(g, ""):
                problems.append(f"{qid}: key '{key}' not in {g}")
        rows.append({"qid": qid, "question": q, "as_of": a, "type": t, "gold_doc_ids": gold,
                     "answer_key": key, "stale_key": stale, "note": note})
    with open(os.path.join(os.path.dirname(__file__), "golden.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(len(rows), "questions |", counters, "| label problems:", problems or "none")

if __name__ == "__main__":
    main()
