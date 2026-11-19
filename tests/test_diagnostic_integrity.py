from scripts.summarize_revision_evidence import summarize
from scripts.freeze_canonical_evidence_20260819 import normalized_text_sha256
import pytest


def test_population_standard_error_coverage():
    rows = summarize()["population_se_coverage"]
    assert [r["n"] for r in rows] == [160, 640, 2560]
    assert [r["coverage"] for r in rows] == pytest.approx([.9675, .959, .958])


def test_text_hash_independent_of_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.tsv", tmp_path / "crlf.tsv"
    lf.write_bytes(b"n\tvalue\n160\t0.9675\n")
    crlf.write_bytes(b"n\tvalue\r\n160\t0.9675\r\n")
    assert normalized_text_sha256(lf) == normalized_text_sha256(crlf)


def test_diagnostic_comparison_rejects_changed_counts_hashes_and_values():
    import runpy
    from pathlib import Path
    compare = runpy.run_path(str(Path(__file__).with_name("test_revision_evidence.py")))["assert_diagnostics_match"]
    compare({"correlation": .9 + 2e-16, "count": 1000, "hash": "abc"},
            {"correlation": .9, "count": 1000, "hash": "abc"})
    for changed in ({"count": 999}, {"hash": "def"}, {"correlation": .900001}):
        expected = {"correlation": .9, "count": 1000, "hash": "abc"}
        with pytest.raises(AssertionError):
            compare({**expected, **changed}, expected)
