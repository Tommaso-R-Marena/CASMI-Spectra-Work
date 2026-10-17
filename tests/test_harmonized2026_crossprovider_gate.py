"""Test metadata-only 2026 GNPS vs MassSpecGym formula hard negatives."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_harmonized2026_crossprovider_gate import parse_head,freeze,stablehash


def item(source,identity,index,mz,formula="C8H10O2",mode="positive"):
    return {"source":source,"inchikey":identity,"formula":formula,
            "mode":mode,"precursor_mz":mz,"index":index,
            "splash":f"unique{index}","instrument":"TOF"}


def test_source_metadata_only_requires_explicit_full_key():
    h={"SOURCE_DATASET":"GNPS","INCHIKEY":"UHOVQNZJYSORNB-UHFFFAOYSA-N",
       "FORMULA":"C8H10O2","IONMODE":"Positive","PEPMASS":"301.123"}
    assert parse_head(h,10)["source"]=="GNPS"
    assert parse_head({**h,"INCHIKEY":"short"},10) is None
    assert parse_head({**h,"SOURCE_DATASET":"unknown"},10) is None
    assert parse_head({**h,"PEPMASS":"nan"},10) is None
    assert parse_head({**h,"IONMODE":""},10) is None


def test_one_query_per_identity_and_candidates_from_reference_only():
    A="AAAAAAAAAAAAAA-AAAAAAAAAA-A"
    B="BBBBBBBBBBBBBB-BBBBBBBBBB-B"
    rows=[item("GNPS",A,0,301.123),
          item("GNPS",B,1,301.124),
          item("MASSSPECGYM",A,2,301.1231),
          item("MASSSPECGYM",A,3,301.1232)]
    refs,qs,coverage,gate=freeze(rows)
    assert len(refs)==2
    assert len(qs)==1
    assert qs[0]["candidate_structures"]==2
    assert coverage["cross_provider_structures"]==1
    assert coverage["hard_queries"]==1
    assert gate is False
    # Input order must NEVER change a selected structure/scan.
    refs2,qs2,_,_=freeze(list(reversed(rows)))
    assert qs2==qs
    assert refs2==refs


def test_formula_or_polarity_restriction_keeps_wrong_candidates_out():
    A="AAAAAAAAAAAAAA-AAAAAAAAAA-A"
    B="BBBBBBBBBBBBBB-BBBBBBBBBB-B"
    rows=[item("GNPS",A,0,301.123),
          item("GNPS",B,1,301.125,formula="C8H10N2"),
          item("MASSSPECGYM",A,2,301.1231)]
    _,qs,cov,_=freeze(rows)
    assert qs==[]
    assert cov["hard_queries"]==0


def test_precursor_outside_fixed_mass_window_cannot_be_cherrypicked():
    A="AAAAAAAAAAAAAA-AAAAAAAAAA-A"
    B="BBBBBBBBBBBBBB-BBBBBBBBBB-B"
    rows=[item("GNPS",A,0,301.5),item("GNPS",B,1,301.123),
          item("MASSSPECGYM",A,2,301.123)]
    _,qs,cov,_=freeze(rows)
    assert qs==[]
    assert cov["cross_provider_structures"]==0
