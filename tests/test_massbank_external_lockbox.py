from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_massbank_external_lockbox import build_external_audit


def _record(accession: str, formula: str, inchikey: str, energy: int, contributor: str) -> str:
    return f"""ACCESSION: {accession}
RECORD_TITLE: synthetic
DATE: 2026.01.01
CH$NAME: synthetic
CH$FORMULA: {formula}
CH$EXACT_MASS: 100.0
CH$LINK: INCHIKEY {inchikey}
AC$INSTRUMENT: {contributor} QTOF
AC$INSTRUMENT_TYPE: LC-ESI-QTOF
AC$MASS_SPECTROMETRY: MS_TYPE MS2
AC$MASS_SPECTROMETRY: ION_MODE POSITIVE
AC$MASS_SPECTROMETRY: COLLISION_ENERGY {energy} eV
MS$FOCUSED_ION: PRECURSOR_M/Z 101.0
MS$FOCUSED_ION: PRECURSOR_TYPE [M+H]+
PK$NUM_PEAK: 2
PK$PEAK: m/z int. rel.int.
  50.0 100 999
  75.0 10 100
//
"""


def _write(root: Path, contributor: str, accession: str, text: str) -> None:
    d = root / contributor
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{accession}.txt").write_text(text, encoding="utf-8")


def test_external_audit_is_label_free_and_uses_same_formula_complete_refs(tmp_path: Path) -> None:
    a = "AAAAAAAAAAAAAA-1111111111-A"
    b = "BBBBBBBBBBBBBB-1111111111-A"
    c = "CCCCCCCCCCCCCC-1111111111-A"

    # A and B are distinct structures with one formula and complete 10/20/40 refs.
    # Their 30 eV spectra become held-out queries. Put them in different contributors
    # to exercise contributor accounting.
    for e in (10, 20, 30, 40):
        _write(tmp_path, "LabA", f"A{e}", _record(f"A{e}", "C8H10O2", a, e, "LabA"))
        _write(tmp_path, "LabB", f"B{e}", _record(f"B{e}", "C8H10O2", b, e, "LabB"))

    # C shares the formula but lacks 40 eV, so it must be absent from pools.
    for e in (10, 20, 30):
        _write(tmp_path, "LabC", f"C{e}", _record(f"C{e}", "C8H10O2", c, e, "LabC"))

    summary, manifest = build_external_audit(tmp_path, dataset_sha="deadbeef")
    assert summary["status"] == "MASSBANK_WHOLE_LIBRARY_LABEL_FREE_EXTERNAL_FEASIBILITY_AUDIT"
    assert summary["design"]["ranking_scores_computed"] is False
    assert summary["design"]["performance_unopened"] is True
    assert summary["coverage"]["queries"] == 2
    assert summary["coverage"]["formula_blocks"] == 1
    assert summary["coverage"]["contributors"] == 2
    assert summary["coverage"]["pool_size"]["min"] == 2
    assert summary["coverage"]["pool_size"]["max"] == 2

    assert len(manifest) == 2
    for row in manifest:
        assert row["candidate_count"] == 2
        assert set(row["candidate_connectivity_keys"].split(";")) == {
            "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"
        }
        assert "CCCCCCCCCCCCCC" not in row["candidate_connectivity_keys"]
        # The held-out 30 eV query path is distinct from every 10/20/40 truth ref.
        assert row["query_path"] not in {
            row["truth_ref_10_path"], row["truth_ref_20_path"], row["truth_ref_40_path"]
        }

    # Feasibility output must contain no performance fields.
    text = str(summary).lower()
    for forbidden in ("mrr", "top1", "top5", "expert_score", "static_best"):
        assert forbidden not in text
