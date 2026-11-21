"""Smoke tests for the data package and the harness. Run: python -m pytest -q tests"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H


def test_data_files_load():
    assert len(H.load_articles()) > 100
    assert len(H.load_tickets()) == 2400
    assert len(H.load_updates()) >= 10
    assert len(H.load_golden("dev")) == 66
    assert len(H.load_query_log()) == 600
    assert len(H.load_pii_sample()) == 200


def test_data_is_exact():
    """Your data files must match the published checksums (re-download the zip if this fails)."""
    import hashlib, json
    d = H.DATA_DIR
    manifest = json.load(open(os.path.join(d, "MANIFEST.json")))
    for f, meta in manifest.items():
        assert hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() == meta["sha256"], f


def test_golden_format():
    for q in H.load_golden("dev"):
        assert set(q) >= {"qid", "question", "as_of", "type", "gold_doc_ids", "answer_key"}
        if q["type"] in ("unanswerable", "pii_probe"):
            assert q["gold_doc_ids"] == []
        else:
            assert q["gold_doc_ids"]


def test_baseline_runs_and_is_far_from_ceiling():
    chunks = H.build_baseline_chunks(H.load_articles())
    bm25 = H.BM25Retriever.from_chunks(chunks)
    dev = H.load_golden("dev")
    r, _ = H.evaluate_retrieval(H.make_baseline_retrieve(bm25), dev, verbose=False)
    a, _ = H.evaluate_answers(H.make_baseline_answer(bm25, chunks), dev, verbose=False)
    assert 0.5 < r["recall@3"] < 0.95
    assert a["accuracy"] < 0.5


def test_ids_and_leak_detector():
    assert H.doc_of("A-1000@v3#c2") == "A-1000@v3"
    assert H.doc_of("[T-100123#c0]") == "T-100123"
    assert "EMAIL" in H.detect_pii_leak("write to maya.okafor7@example.com")
    assert H.detect_pii_leak("You can return footwear within 60 days.") == []


def test_judge_rules():
    q = {"qid": "x", "question": "q", "as_of": "2026-10-29", "type": "stale_trap",
         "gold_doc_ids": ["A-1000@v3"], "answer_key": "60 days", "stale_key": "45 days"}
    good = {"answer": "within 60 days [A-1000@v3]", "citations": ["A-1000@v3#c1"], "abstained": False}
    stale = {"answer": "within 45 days [A-1000@v2]", "citations": ["A-1000@v2#c1"], "abstained": False}
    s, df = H.evaluate_answers(lambda *_: good, [q], verbose=False)
    assert s["accuracy"] == 1.0
    s, df = H.evaluate_answers(lambda *_: stale, [q], verbose=False)
    assert s["accuracy"] == 0.0 and s["stale_answer_rate"] == 1.0


def test_flaky_generator_and_timeout():
    f = H.FlakyGenerator(lambda x: x, error_rate=1.0, base_ms=1)
    try:
        f("a")
        assert False
    except H.GeneratorError:
        pass
    slow = H.FlakyGenerator(lambda x: x, timeout_rate=1.0, hang_ms=500, base_ms=1)
    try:
        H.call_with_timeout(slow, 0.05, "a")
        assert False
    except TimeoutError:
        pass


def test_northbeam_system_template():
    import northbeam_system
    northbeam_system.build()
    out = northbeam_system.answer("Do Northbeam gift cards expire?", "2026-10-29")
    assert set(out) >= {"answer", "citations", "abstained"}
