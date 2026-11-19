"""Table-body checks against a LaTeX manuscript, with negative controls that need no manuscript."""
import os
from pathlib import Path
import pytest
from scripts.audit_revision_manuscript import validate
from scripts.reproduce_arxiv_tables import tables, latex_rows


def synthetic_tables():
    return "".join(f"% BEGIN GENERATED {key}\n{latex_rows(rows)}% END GENERATED {key}\n"
                   for key, rows in tables().items())


def test_changed_number_is_rejected():
    text = synthetic_tables()
    assert validate(text)["numerical_rows"] == 78
    altered = text.replace("0.9270", "0.9500", 1)
    assert altered != text
    with pytest.raises(ValueError, match="numerical table"):
        validate(altered)


def test_missing_and_duplicate_tables_are_rejected():
    text = synthetic_tables()
    with pytest.raises(ValueError, match="inventory"):
        validate("")
    with pytest.raises(ValueError, match="duplicated"):
        validate(text + text)


def test_supplied_manuscript_tables():
    name = os.environ.get("RHOP_REVISION_TEX")
    if not name:
        pytest.skip("set RHOP_REVISION_TEX to the path of a LaTeX manuscript")
    report = validate(Path(name).read_text())
    assert report["tables"] == 9
