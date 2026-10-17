"""Metadata-only invariants for reverse-provider fixed cohort selection."""
import copy
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_reverse_gnps_msg_graph_gate import freeze_reverse


def row(i,source,key,formula,mass,mode="positive"):
    return {"index":i,"source":source,"inchikey":key,
            "formula":formula,"precursor_mz":mass,
            "mode":mode,"splash":"","instrument":""}


def fixture():
    return [
        row(0,"MASSSPECGYM","AAA","C6H6",79.0),
        row(1,"MASSSPECGYM","BBB","C6H6",79.004),
        row(2,"MASSSPECGYM","CCC","C6H6",79.005),
        row(3,"GNPS","AAA","C6H6",79.001),
        row(4,"GNPS","AAA","C6H6",79.001),
        row(5,"GNPS","ZZZ","C6H6",79.002),
    ]


def test_fixed_native_structure_competitors_and_query_truth():
    refs,queries,coverage,passed=freeze_reverse(fixture())
    assert len(queries)==1
    assert queries[0]["inchikey"]=="AAA"
    assert queries[0]["candidate_structures"]==3
    assert len(queries[0]["selected_reference_indices"])==3
    assert not passed
    assert coverage["hard_query_positive"]==1
    assert coverage["hard_query_negative"]==0


def test_no_target_signal_selection_dependency():
    rows=fixture()
    a=freeze_reverse(rows)
    for row in rows:
        row["fake_query_peak_intensities"]=[42.5,1,9987]
        row["fake_query_smiles"]="forbidden"
    b=freeze_reverse(rows)
    # Synthetic forbidden fields must never enter metadata eligibility or
    # hashing and must never change which index became a query/reference.
    assert [x["index"] for x in a[0]]==[x["index"] for x in b[0]]
    assert [x["index"] for x in a[1]]==[x["index"] for x in b[1]]


def test_formula_polarity_and_precursor_constraints_are_strict():
    rows=fixture()
    rows.append(row(6,"MASSSPECGYM","DDD","C7H8",79.001))
    rows.append(row(7,"MASSSPECGYM","EEE","C6H6",80.1))
    rows.append(row(8,"MASSSPECGYM","FFF","C6H6",79.001,"negative"))
    refs,queries,coverage,_=freeze_reverse(rows)
    assert len(queries)==1
    assert queries[0]["candidate_structures"]==3
    assert len(queries[0]["selected_reference_indices"])==3


def test_prohibit_full_key_collision_as_truth_fallback():
    rows=fixture()
    rows[0]["inchikey"]="DIFFERENT"
    refs,queries,coverage,_=freeze_reverse(rows)
    assert queries==[]
    assert coverage["cross_provider_exact_structure_query_overlap"]==0


def test_hash_selection_is_source_order_independent():
    first=freeze_reverse(fixture())
    second=freeze_reverse(list(reversed(fixture())))
    assert first==second


def test_negative_ion_cohort_stays_separate():
    rows=fixture()+[
        row(10,"MASSSPECGYM","NNN","C6H6",79.0,"negative"),
        row(11,"MASSSPECGYM","MMM","C6H6",79.001,"negative"),
        row(12,"GNPS","NNN","C6H6",79.002,"negative"),
    ]
    refs,queries,coverage,passed=freeze_reverse(rows)
    assert len(queries)==2
    assert coverage["hard_query_negative"]==1
    assert {x["mode"] for x in queries}=={"positive","negative"}
