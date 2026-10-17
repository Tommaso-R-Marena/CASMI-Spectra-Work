"""Metadata-only SPLISUM gate validation: NEVER inspect fragment peaks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_splisum_header_gate import explicit_identity,headers_only,candidate_pool


def test_gate_ignores_unparseable_target_peaks(tmp_path):
    p=tmp_path/"mock.mgf"
    p.write_text(
        "BEGIN IONS\nPEPMASS=300.21\nINCHIKEY=UHOVQNZJYSORNB-UHFFFAOYSA-N\nIONMODE=Positive\n"
        "not even a numeric peak line\nEND IONS\n")
    rows,stats=headers_only(p,"positive")
    assert rows==[{"index":0,"label":"inchikey:UHOVQNZJYSORNB-UHFFFAOYSA-N",
                  "precursor_mz":300.21,"mode":"positive"}]
    assert stats["with_molecule_identity"]==1


def test_unknown_identity_never_falls_back_to_title():
    assert explicit_identity({"NAME":"compound 1", "TITLE":"compound 1"}) is None
    assert explicit_identity({"INCHIKEY":"UHOVQNZJYSORNB-UHFFFAOYSA-N"})=="inchikey:UHOVQNZJYSORNB-UHFFFAOYSA-N"
    assert explicit_identity({"SMILES":"C(C)O"})=="literal_smiles:C(C)O"


def test_fixed_precursor_gate_keeps_multiple_competitors():
    rows=[{"precursor_mz":299.8,"label":"1"},
          {"precursor_mz":300.10,"label":"2"},
          {"precursor_mz":300.11,"label":"3"},
          {"precursor_mz":303.0,"label":"4"}]
    got=candidate_pool(rows,{"precursor_mz":300.10})
    assert {x["label"] for x in got}=={"2","3"}


def test_mode_mismatch_is_fatal(tmp_path):
    p=tmp_path/"bad.mgf"
    p.write_text("BEGIN IONS\nPEPMASS=30\nINCHIKEY=UHOVQNZJYSORNB-UHFFFAOYSA-N\nIONMODE=negative\nEND IONS")
    import pytest
    with pytest.raises(RuntimeError):
        headers_only(p,"positive")
