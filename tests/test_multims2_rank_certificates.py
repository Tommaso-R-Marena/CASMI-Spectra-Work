"""Pure reference-only rank certificate implementation tests."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_multims2_rank_certificates import replay_reference_predictions
from qfd.target_blind_geometry import affinity,query_mass_on_reference_grid
from qfd.support_certificates import certify_top1_unchanged


def spec(mz,v):
    return SimpleNamespace(mzs=np.asarray(mz,float),intensities=np.asarray(v,float))


def test_rank_audit_predictions_unchanged_when_query_changes():
    lo=spec([100,110,130],[8,1,1])
    hi=spec([100.002,140,160],[3,5,2])
    grid,preds,bounds=replay_reference_predictions(lo,hi)
    original={name:row.copy() for name,row in preds.items()}
    query1=spec([100.001,170],[5,5])
    query2=spec([100.003,180],[5,5])
    for query in (query1,query2):
        q,unmatched=query_mass_on_reference_grid(query,grid)
        assert abs(float(q.sum())+unmatched-1)<1e-10
        for name,pred in preds.items():
            assert np.array_equal(pred,original[name])
            assert 0<=affinity(q,pred)<=1
    assert 0<=bounds["h_to_q"]<=1
    assert 0<=bounds["hybrid_to_q"]<=1
    assert 0<=bounds["hybrid_to_support"]<=1


def test_identical_references_prove_all_rank_changes_impossible():
    a=spec([100,120],[5,5])
    grid,preds,bounds=replay_reference_predictions(a,a)
    assert bounds["h_to_q"]==0
    assert bounds["hybrid_to_q"]==0
    assert bounds["hybrid_to_support"]==0
    check=certify_top1_unchanged({"A":1.,"B":.5},
                                 {"A":bounds["h_to_q"],"B":bounds["h_to_q"]})
    assert check["certified_unchanged"]
