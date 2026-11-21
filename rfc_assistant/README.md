# RFC Assistant: reproducing the results

The full design write-up is in `DESIGN.md`. This file explains how to reproduce every number.

## Files
| File | What it is |
|---|---|
| `download_data.py` | Downloads `rfc-index.txt` and the selected RFC texts into `data/raw/` and writes `data/catalog.json`. |
| `ingest.py`, `system.py` | The first version of the system (v1), kept for comparison. |
| `ingest_new.py` | Validation → quarantine → cleaning → section chunking → freshness rule. Writes `data/knowledge_chunks_new.parquet`, `data/quarantine_new.csv`, `data/index_dates.json`. |
| `system_new.py` | Query path: `build()`, `retrieve(question, as_of, k)`, `answer(question, as_of)`, and `AnswerCache`. `CONFIG` holds the ablation switches; the defaults are the shipped system. |
| `golden.jsonl` / `make_golden.py` | 34 labelled questions (20 answerable, 4 stale traps, 10 unanswerable), plus the script that writes and checks them. |
| `golden_test.jsonl` / `make_test.py` | 14 held-out questions, frozen before the v2 changes (sha1 `3a0c9d3888406afabdd8f1c5e4c954b0eab9d12a`). |
| `evaluate.py` / `evaluate.ipynb` | Baseline vs. v1 vs. v2, ablations, held-out test, latency and the cache experiment. Writes `outputs/`. |
| `make_design_diagram.py` | Draws `outputs/q3_design.png`. |

Every line that differs from the original `ingest.py` / `system.py` sits between `#-----` markers in the `_new` files.

## Reproduce (from the repo root; Python 3.10+)
```bash
pip install -r requirements.txt          # pandas, numpy, scikit-learn, pyarrow, matplotlib
python rfc_assistant/download_data.py               # ~12 MB from rfc-editor.org (the texts are not shipped in the ZIP)
python rfc_assistant/ingest_new.py                      # 240 valid RFCs, 9 quarantined, 9,377 chunks (< 1 s)
python rfc_assistant/evaluate.py                    # all tables in rfc_assistant/outputs/ (~25 s on 2 vCPUs)
python rfc_assistant/make_design_diagram.py         # optional: redraw the diagram
```
`system_new.build()` rebuilds the chunks from `data/raw/` when the texts are present and otherwise reads
`data/knowledge_chunks_new.parquet`. The freshness rule reads successor dates from `data/raw/rfc-index.txt`,
or from `data/index_dates.json` when `raw/` is absent. No API key and no network access are needed after the download.

**Embedding model:** TF-IDF (unigrams, min_df 2) + TruncatedSVD (256 dims), i.e. LSA, fitted on the chunks with
scikit-learn (`random_state=0`). No model weights are downloaded.

## Evaluation protocol (honesty notes)
1. `golden.jsonl` was written from the RFC texts before the system was run on it, and its labels were never changed.
2. The v1 → v2 changes (scope guard, trusted evidence, sentence scoring, full-index successors) were made after
   reading v1's golden failures, so the v2 golden numbers are optimistic.
3. `golden_test.jsonl` was written and hashed before those changes and evaluated once. Report it as the unbiased number.
4. Timings depend on the machine. Re-run `evaluate.py` and quote your own hardware (printed on the first line).
