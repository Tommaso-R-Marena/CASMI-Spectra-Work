from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_massbank_postrelease_lockbox import build_audit


def _record(
    accession: str,
    formula: str,
    inchikey: str,
    energy: int,
    *,
    ion_mode: str = "POSITIVE",
    precursor_type: str = "[M+H]+",
) -> str:
    return f"""ACCESSION: {accession}
DATE: 2026.06.16
CH$NAME: synthetic
CH$FORMULA: {formula}
CH$LINK: INCHIKEY {inchikey}
AC$INSTRUMENT: Synthetic QTOF
AC$INSTRUMENT_TYPE: LC-ESI-QTOF
AC$MASS_SPECTROMETRY: MS_TYPE MS2
AC$MASS_SPECTROMETRY: ION_MODE {ion_mode}
AC$MASS_SPECTROMETRY: COLLISION_ENERGY {energy} eV
MS$FOCUSED_ION: PRECURSOR_TYPE {precursor_type}
PK$NUM_PEAK: 2
PK$PEAK: m/z int. rel.int.
  50.0 100 999
  75.0 10 100
//
"""


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _series(root: Path, prefix: str, key: str, *, energies=(10, 20, 40), formula="C8H10O2") -> None:
    for e in energies:
        _write(root, f"Lib/{prefix}{e}.txt", _record(f"{prefix}{e}", formula, key, e))


def test_postrelease_audit_excludes_renames_and_uses_any_scope_for_novelty(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    baseline.mkdir(); current.mkdir()

    a = "AAAAAAAAAAAAAA-1111111111-A"
    b = "BBBBBBBBBBBBBB-1111111111-A"
    d = "DDDDDDDDDDDDDD-1111111111-A"
    r = "RRRRRRRRRRRRRR-1111111111-A"

    # A and R are ordinary baseline structures with complete reference series.
    _series(baseline, "A", a)
    _series(baseline, "R", r)
    _write(baseline, "Old/R30.txt", _record("R30", "C8H10O2", r, 30))

    # D existed before the cutoff, but only in a negative-ion record. It must
    # still count as historically seen and therefore not structurally novel.
    _write(
        baseline,
        "Other/D5neg.txt",
        _record("D5NEG", "C8H10O2", d, 5, ion_mode="NEGATIVE", precursor_type="[M-H]-"),
    )

    shutil.copytree(baseline, current, dirs_exist_ok=True)
    (current / "Old/R30.txt").unlink()
    _write(current, "Renamed/R30.txt", _record("R30", "C8H10O2", r, 30))

    # A gets a genuinely new query spectrum; B is a genuinely new structure;
    # D gets a positive complete series but is historically known via negative ion.
    _write(current, "New/A30.txt", _record("A30NEW", "C8H10O2", a, 30))
    for prefix, key in (("B", b), ("D", d)):
        for e in (10, 20, 30, 40):
            _write(current, f"New/{prefix}{e}.txt", _record(f"{prefix}{e}", "C8H10O2", key, e))

    added = {
        "New/A30.txt",
        *(f"New/B{e}.txt" for e in (10, 20, 30, 40)),
        *(f"New/D{e}.txt" for e in (10, 20, 30, 40)),
        # Deliberately omit Renamed/R30.txt: git rename detection would classify it R, not A.
    }

    summary, manifest = build_audit(
        current,
        baseline,
        added,
        current_sha="current",
        baseline_sha="baseline",
    )

    assert summary["status"] == "MASSBANK_POSTRELEASE_LABEL_FREE_FEASIBILITY_AUDIT"
    assert summary["design"]["ranking_scores_computed"] is False
    assert summary["guardrails"]["renames_and_modifications_excluded_by_git_diff"] is True

    by_key = {row["truth_connectivity_key"]: row for row in manifest}
    assert set(by_key) == {a.split("-")[0], b.split("-")[0], d.split("-")[0]}
    assert r.split("-")[0] not in by_key

    assert by_key[a.split("-")[0]]["structure_novel_at_baseline"] is False
    assert by_key[b.split("-")[0]]["structure_novel_at_baseline"] is True
    assert by_key[d.split("-")[0]]["structure_novel_at_baseline"] is False
    assert all(int(row["candidate_count"]) == 4 for row in manifest)
    assert summary["coverage"]["eligible_novel_structure"]["queries"] == 1
    assert summary["coverage"]["eligible_added_spectrum"]["queries"] == 3

    forbidden = {"mrr", "rr", "rank", "score", "top1", "top5", "expert_score", "static_best"}
    assert not (set(summary) & forbidden)
    assert not (set(manifest[0]) & forbidden)


def test_postrelease_audit_rejects_added_paths_missing_from_current_tree(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"; current = tmp_path / "current"
    baseline.mkdir(); current.mkdir()
    with pytest.raises(ValueError, match="absent from current tree"):
        build_audit(
            current,
            baseline,
            {"Ghost/missing.txt"},
            current_sha="current",
            baseline_sha="baseline",
        )
