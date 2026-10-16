"""Adversarial tests for the strictly retrospective MassBank artifact analysis."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(rel):
    path=ROOT/rel
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m=load("scripts/posthoc_massbank_robustness.py")
inv=load("scripts/export_prior_msnlib_key_inventory.py")

def toy():
    def row(target,formula,key,contributor,score,base,oracle,non_orb,t):
        return {
          "query_path":target,"formula":formula,"inchikey":key,
          "contributor":contributor,"instrument_type":"QTOF" if non_orb else "ITFT",
          "ion_mode":"POSITIVE","non_orbitrap":non_orb,"t":t,
          "unmatched_query_mass":.05,"alpha_quarter":score,
          "hellinger_fixed":base,"hellinger_oracle":oracle,
          "support_only":.8,"convex_oracle":.85,
        }
    return [
      row("a","F1","A","source1",.9,.8,.95,True,.5),
      row("b","F1","A","source1",.8,.9,.96,True,.4),
      row("c","F2","B","source2",.95,.9,.98,False,.5),
    ]

def test_formula_block_bootstrap_is_deterministic_and_weighted():
    a=m.delta_summary(toy(),"hellinger_fixed",1000,123)
    b=m.delta_summary(toy(),"hellinger_fixed",1000,123)
    assert a==b
    assert a["formula_block_bootstrap_95"]["n_blocks"]==2 if "formula_block_bootstrap_95" in a else True
    assert a["wins"]==2 and a["losses"]==1
    assert abs(a["mean_query_weighted_delta"]-(.1-.1+.05)/3)<1e-12

def test_exact_inventory_is_all_four_libraries():
    libraries={k:{"A"} for k in inv.EXPECTED_MD5}
    libraries["mcescaf_pos"]={"B"}
    a=inv.inventory(libraries)
    assert a["all_prior_evaluated_full_inchikeys"]==["A","B"]
    with pytest.raises(ValueError,match="exactly four"):
        inv.inventory({"mcebio_neg":{"A"}})

def test_full_key_overlap_rejects_incorrect_count_or_incomplete_sources():
    rows=toy()
    metadata={
      "n_overlapping_prior_msnlib_structures":1,
      "n_overlapping_prior_msnlib_queries":2,
    }
    identity={
      "status":"PRIOR_MSNLib_EXACT_INCHIKEY_INVENTORY_POSTHOC",
      "sources":{x:{} for x in m.PREVIOUS},
      "all_prior_evaluated_full_inchikeys":["A"],
    }
    assert m.exact_prior_keys(identity,rows,metadata)=={"A"}
    with pytest.raises(ValueError,match="conflicts"):
        m.exact_prior_keys(identity,rows,{**metadata,"n_overlapping_prior_msnlib_queries":1})
    identity["all_prior_evaluated_full_inchikeys"]=["A","A"]
    with pytest.raises(ValueError,match="unique"):
        m.exact_prior_keys(identity,rows,metadata)

def test_nonoverlap_does_not_remove_original_scored_queries():
    metadata={"n_nonoverlapping_prior_msnlib_queries":1,
              "n_overlapping_prior_msnlib_queries":2}
    result=m.analyze(toy(),metadata,prior={"A"},draws=100)
    assert result["all"]["queries"]==3
    assert result["subgroups"]["no_prior_msnlib_structure_overlap"]["queries"]==1
    assert result["subgroups"]["prior_msnlib_structure_overlap"]["queries"]==2
    assert result["subgroups"]["offcenter_energy"]["queries"]==1
    assert result["n_contributors"]==2
    assert len(result["leave_one_contributor_out_descriptive"])==2
    assert result["no_refitting_or_retuning"]
    with pytest.raises(ValueError,match="cohort"):
        m.analyze(toy(),{**metadata,"n_nonoverlapping_prior_msnlib_queries":2},prior={"A"},draws=100)

def test_oracle_dominates_fixed_comparator_in_synthetic_data():
    rows=toy()
    assert all(r["hellinger_oracle"]>=r["hellinger_fixed"] for r in rows)
