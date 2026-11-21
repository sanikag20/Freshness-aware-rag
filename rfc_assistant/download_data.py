"""Download the Q3 corpus: a topical subset of IETF RFCs from rfc-editor.org.

Source   : https://www.rfc-editor.org (RFC Editor, public archive)
License  : RFCs may be freely reproduced and distributed (IETF Trust Legal Provisions).
What     : rfc-index.txt (the full catalogue) + the plain-text of ~250 selected RFCs
           about everyday internet protocols, including superseded versions and the
           April-1st joke RFCs that act as an untrusted source tier.
Usage    : python download_data.py            (writes into rfc_assistant/data/raw, ~15 MB)
The retrieval date and the selected list are recorded in rfc_assistant/data/catalog.json.
"""
import json
import os
import re
import time
import urllib.request
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
BASE = "https://www.rfc-editor.org/rfc"

TOPICS = re.compile(
    r"\b(HTTP|SMTP|Mail|DNS|Domain|TLS|Transport Layer Security|TCP|UDP|Datagram|"
    r"URI|URL|URN|JSON|WebSocket|FTP|File Transfer|IMAP|POP3|SSH|NTP|Network Time|"
    r"DHCP|IPv6|IPv4|Internet Protocol|OAuth|Cookie|MIME|Telnet|ICMP|ARP|CBOR|"
    r"QUIC|Internet Message|Hypertext)\b", re.I)
ALWAYS = [768, 791, 793, 821, 822, 959, 1034, 1035, 1149, 1945, 2068, 2324, 2396,
          2616, 2821, 2822, 3986, 5321, 5322, 6455, 6265, 7230, 7231, 7540, 8259,
          8446, 9110, 9111, 9112, 9113, 9114, 9293, 1738, 1866, 2818, 4251, 5905,
          2460, 8200, 826, 792, 7159, 4627, 6749, 5246, 4346, 2246]
MAX_DOCS = 250


def fetch(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "rfc-assistant-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", errors="replace")


def parse_index(text):
    entries = {}
    for block in re.split(r"\n\n+", text):
        block = " ".join(ln.strip() for ln in block.splitlines())
        m = re.match(r"^(\d{1,4}) (.*)", block)
        if not m:
            continue
        num = int(m.group(1))
        body = m.group(2)
        title = body.split(". ")[0]
        if title.strip().lower() == "not issued":
            continue
        dm = re.search(r"\b((?:January|February|March|April|May|June|July|August|"
                       r"September|October|November|December) \d{4})\b", body)
        entries[num] = {
            "rfc": num, "title": title, "date": dm.group(1) if dm else None,
            "status": (re.search(r"\(Status: ([^)]+)\)", body) or [None, "UNKNOWN"])[1],
            "obsoletes": [int(x) for x in re.findall(r"RFC(\d{4})",
                          (re.search(r"\(Obsoletes ([^)]+)\)", body) or [None, ""])[1])],
            "obsoleted_by": [int(x) for x in re.findall(r"RFC(\d{4})",
                             (re.search(r"\(Obsoleted by ([^)]+)\)", body) or [None, ""])[1])],
            "updated_by": [int(x) for x in re.findall(r"RFC(\d{4})",
                           (re.search(r"\(Updated by ([^)]+)\)", body) or [None, ""])[1])],
        }
    return entries


def main():
    os.makedirs(RAW, exist_ok=True)
    print("fetching rfc-index.txt ...")
    index_text = fetch(f"{BASE.rsplit('/', 1)[0]}/rfc-index.txt")
    with open(os.path.join(RAW, "rfc-index.txt"), "w", encoding="utf-8") as fh:
        fh.write(index_text)
    entries = parse_index(index_text)
    print(f"index entries: {len(entries)}")

    selected = set(ALWAYS)
    for num, e in sorted(entries.items()):
        if len(selected) >= MAX_DOCS:
            break
        if num in selected or not TOPICS.search(e["title"]):
            continue
        selected.add(num)
        selected.update(x for x in e["obsoleted_by"] if x in entries)
    selected = sorted(x for x in selected if x in entries)[:MAX_DOCS]

    ok, failed = [], []
    for i, num in enumerate(selected):
        path = os.path.join(RAW, f"rfc{num}.txt")
        if not os.path.exists(path):
            try:
                text = fetch(f"{BASE}/rfc{num}.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                time.sleep(0.2)
            except Exception as exc:
                failed.append({"rfc": num, "error": str(exc)})
                continue
        ok.append(num)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(selected)} downloaded")

    catalog = {"source": BASE, "retrieved": date.today().isoformat(),
               "n_selected": len(selected), "n_downloaded": len(ok), "failed": failed,
               "rfcs": {str(n): entries[n] for n in ok}}
    with open(os.path.join(HERE, "data", "catalog.json"), "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=1)
    size = sum(os.path.getsize(os.path.join(RAW, f)) for f in os.listdir(RAW))
    print(f"done: {len(ok)} RFCs, {size/1e6:.1f} MB in {RAW}")


if __name__ == "__main__":
    main()
