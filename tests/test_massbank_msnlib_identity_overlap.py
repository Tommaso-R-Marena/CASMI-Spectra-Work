import importlib.util
from pathlib import Path
import pytest

FILE=Path(__file__).resolve().parents[1]/"scripts"/"audit_massbank_msnlib_identity_overlap.py"
spec=importlib.util.spec_from_file_location("audit_massbank_msnlib_identity_overlap",FILE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def records():
    return [
        {"query_path":"AA/1.txt","contributor":"AA","instrument_type":"QTOF","inchikey":"A"},
        {"query_path":"AA/2.txt","contributor":"AA","instrument_type":"QTOF","inchikey":"A"},
        {"query_path":"BB/3.txt","contributor":"BB","instrument_type":"ITFT","inchikey":"B"},
        {"query_path":"BB/4.txt","contributor":"BB","instrument_type":"ITFT","inchikey":"C"},
    ]


def libraries():
    return {
        "mcebio_neg":{"A","Z"}, "mcescaf_pos":{"B"},
        "enammol_pos":{"Z"}, "enamdisc_neg":set(),
    }


def test_exact_full_key_overlap_and_contributor_counts():
    got=mod.audit_overlap(records(),libraries(),discovery_keys={"D"})
    assert got["n_query_triplets"]==4
    assert got["n_structures"]==3
    assert got["n_overlapping_prior_msnlib_structures"]==2
    assert got["n_nonoverlapping_prior_msnlib_structures"]==1
    assert got["n_overlapping_prior_msnlib_queries"]==3
    assert got["n_nonoverlapping_prior_msnlib_queries"]==1
    assert got["by_previous_evaluated_cohort"]["mcebio_neg"]["shared_massbank_query_triplets"]==2
    assert got["by_massbank_contributor_instrument"][1]["previous_msnlib_nonoverlap_queries"]==1
    assert got["retrospective_filtering_of_primary_scores"] is False


def test_fail_closed_on_leak_missing_cohort_and_duplicate_accession():
    with pytest.raises(ValueError,match="discovery source leaked"):
        mod.audit_overlap(records(),libraries(),discovery_keys={"A"})
    with pytest.raises(ValueError,match="all four"):
        mod.audit_overlap(records(),{"mcebio_neg":{"A"}},discovery_keys=set())
    rows=records()
    rows[1]["query_path"]=rows[0]["query_path"]
    with pytest.raises(ValueError,match="duplicate MassBank query accession"):
        mod.audit_overlap(rows,libraries(),discovery_keys=set())
