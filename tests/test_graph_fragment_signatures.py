"""Chemistry-conditioned fragment graph correctness/falsification safeguards."""
import numpy as np
import pytest
pytest.importorskip("rdkit")
from qfd.graph_fragment_signatures import (
    signatures,candidate_match_vector,discriminative_graph_scores)


def test_constitutional_isomers_same_formula_yield_distinct_signatures():
    a,stat_a,cuta=signatures("CCCOC","C4H10O")
    b,stat_b,cutb=signatures("CCOCC","C4H10O")
    assert stat_a=="OK" and stat_b=="OK"
    assert a and b and a!=b
    assert 0<cuta<=24 and 0<cutb<=24
    assert all(v>0 for v in a+b)


def test_unusable_native_chemical_annotation_must_fail_without_target_selection():
    assert signatures("CCCOC","C5H12O")[1]=="NATIVE_SMILES_FORMULA_MISMATCH"
    assert signatures("not a smile","C4H10O")[1]=="INVALID_NATIVE_SMILES"
    assert signatures("","C4H10O")[1]=="MISSING_NATIVE_STRUCTURE_OR_FORMULA"
    assert signatures("CC","C2H6")[1]=="UNSUPPORTED_HEAVY_ATOM_COUNT"


def test_graph_scores_identity_blind_and_reward_candidate_specific_features():
    query=(np.array([100.,120.,140.]),np.array([1.,8.,1.]))
    a=(100.,120.)
    b=(100.,130.)
    val,diag=discriminative_graph_scores(query,{"A":a,"B":b})
    assert val["A"]==pytest.approx(.4)
    assert val["B"]==pytest.approx(0.)
    assert diag["n_query_peaks_with_any_structure_explanation"]==2
    again,_=discriminative_graph_scores(query,{"B":b,"A":a})
    assert again==val


def test_mass_window_and_unmatched_peaks_never_renormalized():
    x=candidate_match_vector([100,120,140],[100.005,140.011],.01)
    assert x.tolist()==[True,False,False]
    q=(np.array([100.,120.]),np.array([1.,9.]))
    result,_=discriminative_graph_scores(q,{"A":(100.,),"B":(100.25,)})
    assert result["A"]==pytest.approx(.05)


def test_universal_formula_peaks_have_zero_discriminative_power():
    q=(np.array([100.,200.]),np.array([10.,20.]))
    scores,_=discriminative_graph_scores(q,{"A":(100.,200.),
                                            "B":(100.,200.)})
    assert scores=={"A":0.,"B":0.}


def test_constant_graph_fallback_preserves_historical_spectrum_only_ranking():
    q=(np.array([100.]),np.array([1.]))
    f,_=discriminative_graph_scores(q,{"A":(),"B":()})
    assert f=={"A":0.,"B":0.}
    h={"A":.7,"B":.5}
    final={k:.9*h[k]+.1*f[k] for k in h}
    assert max(final,key=final.get)==max(h,key=h.get)


def test_empty_candidate_pool_rejected():
    q=(np.array([100.]),np.array([1.]))
    with pytest.raises(ValueError):
        discriminative_graph_scores(q,{"A":(100.,)})


def test_fixed_candidate_mass_is_not_artificially_renormalized():
    q=(np.array([100.,120.,140.]),np.array([1.,1.,8.]))
    scores,_=discriminative_graph_scores(q,{
        "A":(100.,),"B":(120.,)})
    assert scores["A"]==pytest.approx(.05)
    assert scores["B"]==pytest.approx(.05)
