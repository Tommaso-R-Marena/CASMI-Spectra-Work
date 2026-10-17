"""MassBank native SMILES feasibility must never parse target MS/MS peaks."""
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_massbank_graph_native_headers import reference_header,chemical_candidate


def test_reference_header_reads_only_header_until_peak_delimiter(tmp_path):
    path=tmp_path/"small.zip"
    with ZipFile(path,"w") as z:
        z.writestr("candidate.txt",
            "CH$FORMULA: C3H8O\n"
            "CH$SMILES: CCOC\n"
            "CH$LINK: INCHIKEY ABC-XYZ\n"
            "PK$PEAK: m/z int. rel.int.\n"
            "100.000 20 30\n"
            "CH$SMILES: FORBIDDEN_AFTER_SPECTRAL_PEAK\n")
    with ZipFile(path) as z:
        h=reference_header(z,"candidate.txt")
    assert h["SMILES"]=="CCOC"
    assert h["FORMULA"]=="C3H8O"
    assert h["INCHIKEY"]=="ABC-XYZ"


def test_usable_reference_graph_does_not_need_target_spectrum():
    c={"formula":"C3H8O","inchikey":"ABC-XYZ"}
    h={"FORMULA":"C3H8O","INCHIKEY":"ABC-XYZ","SMILES":"CCOC"}
    got=chemical_candidate(c,h,h)
    assert got["status"]=="OK"
    assert got["n_fragment_masses"]>0
    assert got["n_eligible_breaks"]>=1


def test_missing_graph_candidate_retained_as_zero():
    c={"formula":"C3H8O","inchikey":"ABC-XYZ"}
    h={"FORMULA":"C3H8O","INCHIKEY":"ABC-XYZ"}
    out=chemical_candidate(c,h,h)
    assert out["status"]=="MISSING_NATIVE_REFERENCE_SMILES"
    assert out["n_fragment_masses"]==0
    assert out["smiles"]==""


def test_contradictory_reference_chemistry_fails_closed():
    c={"formula":"C3H8O","inchikey":"ABC-XYZ"}
    h={"FORMULA":"C3H8O","INCHIKEY":"ABC-XYZ","SMILES":"CCOC"}
    bad={"FORMULA":"C4H10","INCHIKEY":"ABC-XYZ","SMILES":"CCOC"}
    with pytest.raises(RuntimeError):
        chemical_candidate(c,h,bad)


def test_mismatched_smiles_do_not_change_candidate_selection():
    c={"formula":"C3H8O","inchikey":"ABC-XYZ"}
    h={"FORMULA":"C3H8O","INCHIKEY":"ABC-XYZ","SMILES":"CCOC"}
    alt={**h,"SMILES":"CCCO"}
    out=chemical_candidate(c,h,alt)
    assert out["status"]=="CONTRADICTORY_NATIVE_REFERENCE_SMILES"
    assert out["n_fragment_masses"]==0


def test_candidate_native_smiles_formula_mismatch_is_invalid():
    c={"formula":"C3H8O","inchikey":"ABC-XYZ"}
    h={"FORMULA":"C3H8O","INCHIKEY":"ABC-XYZ","SMILES":"CCCC"}
    out=chemical_candidate(c,h,h)
    assert out["status"]=="NATIVE_SMILES_FORMULA_MISMATCH"
    assert out["n_fragment_masses"]==0
