"""Check the numerical table bodies of a LaTeX manuscript against the public results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_arxiv_tables import tables, latex_rows


def validate(text):
    expected = tables()
    blocks = re.findall(r"% BEGIN GENERATED (\w+)\n([\s\S]*?)% END GENERATED (\w+)", text)
    found = {}
    for start, body, end in blocks:
        if start != end or start in found:
            raise ValueError("mismatched or duplicated generated table block")
        found[start] = body
    if set(found) != set(expected):
        raise ValueError("generated table inventory differs from public evidence")
    normalize = lambda s: re.sub(r"\s+", " ", s).strip()
    for name, rows in expected.items():
        if normalize(found[name]) != normalize(latex_rows(rows)):
            raise ValueError(f"numerical table differs from source: {name}")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate LaTeX labels")
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text))
    if refs - set(labels):
        raise ValueError("unresolved labels: " + ", ".join(sorted(refs - set(labels))))
    bib = re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", text)
    if len(bib) != len(set(bib)):
        raise ValueError("duplicate bibliography keys")
    cited = set()
    for group in re.findall(r"\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}", text):
        cited.update(k.strip() for k in group.split(","))
    if cited != set(bib):
        raise ValueError("missing or uncited bibliography entries: " + str(cited ^ set(bib)))
    return dict(passed=True, tables=len(found), numerical_rows=sum(map(len, expected.values())),
                labels=len(labels), references=len(bib),
                scope="numerical table reconciliation and LaTeX cross-references; not proof certification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tex", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.tex.read_text())
    report["tex_sha256"] = hashlib.sha256(args.tex.read_bytes()).hexdigest()
    data = json.dumps(report, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("choose a fresh audit output")
        args.output.write_text(data)
    print(data)
