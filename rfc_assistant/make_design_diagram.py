"""Draws rfc_assistant/outputs/q3_design.png: the high-level design of the RFC assistant (Q3.2.1)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BOX = {"ingest": "#dbe9f6", "store": "#fdf1d6", "query": "#e3f2e1", "eval": "#f4e1ef"}
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")

def box(x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=BOX[kind], ec="#555", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)

def arrow(x1, y1, x2, y2, label=None, t=0.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#333", lw=1.1))
    if label:
        ax.text(x1 + (x2 - x1) * t + 0.08, y1 + (y2 - y1) * t, label, fontsize=7.4, ha="left", color="#333",
                bbox=dict(fc="white", ec="none", pad=0.5))

ax.text(0.2, 6.9, "Ingestion path (batch, daily)", fontsize=10, weight="bold", color="#2b5d8a")
box(0.2, 5.7, 1.9, 0.9, "rfc-editor.org\nrfc-index.txt +\nRFC plain texts", "ingest")
box(2.5, 5.7, 1.9, 0.9, "Validate\n(contract rules)\n-> quarantine", "ingest")
box(4.8, 5.7, 1.9, 0.9, "Clean\n(page headers,\nfooters, ToC)", "ingest")
box(7.1, 5.7, 1.9, 0.9, "Chunk by section\n(<=180 words)\n+ content hash", "ingest")
box(9.4, 5.7, 1.9, 0.9, "Embed only\nnew hashes\n(LSA / MiniLM)", "ingest")
for x in (2.1, 4.4, 6.7, 9.0):
    arrow(x, 6.15, x + 0.4, 6.15)

ax.text(0.2, 4.95, "Data stores", fontsize=10, weight="bold", color="#8a6d2b")
box(0.2, 3.75, 2.3, 0.95, "rfc_catalog + index_dates\n(status, tier, published,\nobsoleted_by graph)", "store")
box(2.8, 3.75, 2.3, 0.95, "knowledge_chunks\n(Parquet: chunk_id, rfc,\nsection, tier, text, hash)", "store")
box(5.4, 3.75, 2.0, 0.95, "BM25 index +\nvector matrix\n(by content hash)", "store")
box(7.7, 3.75, 1.8, 0.95, "quarantine\n(rule broken)", "store")
box(9.8, 3.75, 1.9, 0.95, "answer cache\n(exact + semantic,\nre-check on read)", "store")
arrow(3.45, 5.7, 1.35, 4.7, "catalog")
arrow(8.05, 5.7, 3.95, 4.7, "chunks", t=0.25)
arrow(10.35, 5.7, 6.4, 4.7, "vectors")
arrow(3.45, 5.7, 8.6, 4.7, "violations", t=0.78)

ax.text(0.2, 3.05, "Query path (online)", fontsize=10, weight="bold", color="#2e7d32")
box(0.2, 1.85, 1.5, 0.95, "POST /answer\n(question,\nas_of)", "query")
box(2.0, 1.85, 1.5, 0.95, "cache\nlookup", "query")
box(3.8, 1.85, 2.0, 0.95, "hybrid retrieval\nBM25 + dense, RRF\n+ synonym expansion", "query")
box(6.1, 1.85, 2.0, 0.95, "freshness filter\n(current on as_of,\nfull-index successors)", "query")
box(8.4, 1.85, 1.9, 0.95, "guardrails: scope,\ntrust tier, coverage\n-> abstain", "query")
box(10.6, 1.85, 2.2, 0.95, "extractive answer\n+ [RFC, section]\ncitation", "query")
for x1, x2 in ((1.7, 2.0), (3.5, 3.8), (5.8, 6.1), (8.1, 8.4), (10.3, 10.6)):
    arrow(x1, 2.32, x2, 2.32)
arrow(4.8, 3.75, 4.8, 2.8, "")
arrow(1.35, 3.75, 7.1, 2.8, "")
arrow(11.2, 2.8, 11.2, 3.75, "put")

ax.text(0.2, 1.15, "Evaluation and monitoring loop", fontsize=10, weight="bold", color="#8a2b6d")
box(0.2, 0.1, 3.1, 0.85, "golden.jsonl (34) + held-out test (14)\nharness harness, nightly", "eval")
box(3.6, 0.1, 3.1, 0.85, "metrics: accuracy, stale-citation rate,\nabstention P/R, p95, cache wrong-hits", "eval")
box(7.0, 0.1, 2.6, 0.85, "alerts (5 thresholds)\n-> on-call / ingest fix", "eval")
box(9.9, 0.1, 2.9, 0.85, "sampled live answers re-checked\nagainst the live obsolescence graph", "eval")
arrow(3.3, 0.52, 3.6, 0.52); arrow(6.7, 0.52, 7.0, 0.52); arrow(9.9, 0.52, 9.6, 0.52)
arrow(12.3, 1.85, 12.3, 0.95, "logs")
plt.tight_layout()
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "q3_design.png"), dpi=150)
print("wrote outputs/q3_design.png")
