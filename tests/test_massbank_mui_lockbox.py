from __future__ import annotations

import sys
from pathlib import Path

# CI runs with PYTHONPATH=src, so import the standalone audit script explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_massbank_mui_lockbox import build_audit


def _record(
    accession: str,
    formula: str,
    inchikey: str,
    energy: int,
    *,
    date: str = "2026.06.16",
    ion_mode: str = "POSITIVE",
    precursor_type: str = "[M+H]+",
) -> str:
    return f"""ACCESSION: {accession}
RECORD_TITLE: synthetic; LC-ESI-QTOF; MS2; CE {energy} eV; {precursor_type}
DATE: {date}
CH$NAME: synthetic
CH$FORMULA: {formula}
CH$EXACT_MASS: 100.0
CH$SMILES: C
CH$IUPAC: InChI=1S/CH4/h1H4
CH$LINK: INCHIKEY {inchikey}
AC$INSTRUMENT: Synthetic QTOF
AC$INSTRUMENT_TYPE: LC-ESI-QTOF
AC$MASS_SPECTROMETRY: MS_TYPE MS2
AC$MASS_SPECTROMETRY: ION_MODE {ion_mode}
AC$MASS_SPECTROMETRY: COLLISION_ENERGY {energy} eV
AC$MASS_SPECTROMETRY: IONIZATION ESI
MS$FOCUSED_ION: PRECURSOR_M/Z 101.0
MS$FOCUSED_ION: PRECURSOR_TYPE {precursor_type}
PK$NUM_PEAK: 2
PK$PEAK: m/z int. rel.int.
  50.0 100 999
  75.0 10 100
//
"""


def _write(root: Path, accession: str, text: str) -> None:
    (root / f"{accession}.txt").write_text(text, encoding="utf-8")


def test_massbank_mui_feasibility_is_label_free_and_tie_safe(tmp_path: Path) -> None:
    # Structure A has two stereochemical InChIKeys sharing the same connectivity block.
    # They must collapse to one 2D candidate identity.
    a1 = "AAAAAAAAAAAAAA-1111111111-A"
    a2 = "AAAAAAAAAAAAAA-2222222222-B"
    b = "BBBBBBBBBBBBBB-1111111111-A"
    c = "CCCCCCCCCCCCCC-1111111111-A"

    # A: complete 10/20/40 references plus a separate held-out 30 eV query.
    for e, key in [(10, a1), (20, a1), (30, a2), (40, a1)]:
        _write(tmp_path, f"A{e}", _record(f"A{e}", "C8H10O2", key, e))

    # B: a distinct same-formula structure with complete references and query.
    for e in (10, 20, 30, 40):
        _write(tmp_path, f"B{e}", _record(f"B{e}", "C8H10O2", b, e))

    # C: same formula but incomplete reference trajectory (missing 40 eV).
    # It must not enter the candidate pool.
    for e in (10, 20, 30):
        _write(tmp_path, f"C{e}", _record(f"C{e}", "C8H10O2", c, e))

    # Off-scope negative-ion record must be ignored.
    _write(tmp_path, "NEG30", _record("NEG30", "C8H10O2", "DDDDDDDDDDDDDD-1-A", 30, ion_mode="NEGATIVE"))

    summary, manifest = build_audit(
        tmp_path,
        dataset_commit="deadbeef",
        baseline_tag="2026.03",
    )

    assert summary["status"] == "MASSBANK_MUI_LABEL_FREE_FEASIBILITY_AUDIT"
    assert summary["predeclared_design"]["no_ranking_scores_computed"] is True
    assert summary["coverage"]["structures_with_complete_10_20_40_refs"] == 2
    assert summary["coverage"]["eligible_ambiguous_queries"] == 2
    assert summary["coverage"]["eligible_formula_groups"] == 1
    assert summary["coverage"]["pool_size"]["min"] == 2
    assert summary["coverage"]["pool_size"]["max"] == 2

    assert len(manifest) == 2
    by_truth = {row["truth_connectivity_key"]: row for row in manifest}
    assert set(by_truth) == {"AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"}

    arow = by_truth["AAAAAAAAAAAAAA"]
    assert arow["query_accession"] == "A30"
    assert arow["truth_ref_10_accession"] == "A10"
    assert arow["truth_ref_20_accession"] == "A20"
    assert arow["truth_ref_40_accession"] == "A40"
    assert arow["candidate_count"] == 2
    assert arow["candidate_connectivity_keys"].split(";") == ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]
    assert "CCCCCCCCCCCCCC" not in arow["candidate_connectivity_keys"]

    # Feasibility outputs must not expose ranking/performance fields.
    serialized_keys = set(summary.keys()) | set(arow.keys())
    forbidden = {"mrr", "rr", "rank", "score", "top1", "top5", "expert_score", "static_best"}
    assert not (serialized_keys & forbidden)
