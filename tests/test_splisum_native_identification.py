"""Robust fixed 20-eV candidate-ranking scorer tests."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_splisum_native_identification import (
    scores_ion_spectra,tie_aware_rank,peaks_for_selected)


def spec(masses,intensities):
    masses=np.asarray(masses,float);intensities=np.asarray(intensities,float)
    i=np.argsort(masses)
    return masses[i],intensities[i]


def test_identity_score_is_one():
    a=spec([100,120,140],[5,3,2])
    v=scores_ion_spectra(a,a)
    for x in v.values():assert x==pytest.approx(1.)


def test_zero_mz_overlap_yields_zero_scores():
    a=spec([100,120],[4,6]);b=spec([101,121],[4,6])
    vals=scores_ion_spectra(a,b)
    assert all(x==0 for x in vals.values())


def test_unmatched_mass_must_not_be_renormalized():
    a=spec([100,120],[50,50])
    b=spec([100,140],[50,50])
    score=scores_ion_spectra(a,b)
    assert score["hellinger"]==pytest.approx(.5)
    assert score["fragment_overlap"]==pytest.approx(.5)
    assert score["cosine"]==pytest.approx(.5)
    assert score["support_composite"]==pytest.approx(.5)


def test_extra_wrong_fragment_strictly_lowers_similarity():
    a=spec([100,120],[5,5])
    b=spec([100,120],[5,5])
    c=spec([100,120,140],[5,5,5])
    s0=scores_ion_spectra(a,b)
    s1=scores_ion_spectra(a,c)
    assert s1["hellinger"]<s0["hellinger"]
    assert s1["fragment_overlap"]<s0["fragment_overlap"]
    assert s1["cosine"]<s0["cosine"]


def test_tie_aware_top1_and_mrr():
    tied=tie_aware_rank({"X":.8,"Y":.8,"Z":.3},"X")
    assert tied["mrr"]==pytest.approx(2/3)
    assert tied["top1_fractional"]==.5
    nonwinner=tie_aware_rank({"X":.8,"Y":.2},"Y")
    assert nonwinner["mrr"]==.5
    assert nonwinner["top1_fractional"]==0


def test_source_peak_parser_fails_empty_selected_record(tmp_path):
    p=tmp_path/"source.mgf"
    p.write_text("BEGIN IONS\nPEPMASS=200\nEND IONS\n")
    with pytest.raises(RuntimeError):
        peaks_for_selected(p,{0},[{"index":0}])


def test_source_peak_parser_ignores_unselected_malformed_peak(tmp_path):
    p=tmp_path/"source.mgf"
    p.write_text(
        "BEGIN IONS\nPEPMASS=200\nthis unselected peak is invalid\nEND IONS\n"
        "BEGIN IONS\nPEPMASS=201\n100.00 50\n110.00 50\nEND IONS\n")
    s=peaks_for_selected(p,{1},[{"index":0},{"index":1}])
    assert set(s)=={1}
    assert len(s[1][0])==2


def test_bad_selected_peak_aborts(tmp_path):
    p=tmp_path/"source.mgf"
    p.write_text("BEGIN IONS\nPEPMASS=200\n100 not_a_number\nEND IONS\n")
    with pytest.raises(RuntimeError):
        peaks_for_selected(p,{0},[{"index":0}])
