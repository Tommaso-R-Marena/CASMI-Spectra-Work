"""No query-structure access, duplicate-copy checks and formula block inference."""
import sys
from pathlib import Path

import numpy as np
import pytest
import zstandard

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_reverse_gnps_msg_graph_transfer import (
    reference_smiles, query_reference_signal_equal, block_inference)


def spectra(m,z):
    return np.asarray(m,float),np.asarray(z,float)


def test_raw_reference_structure_reads_only_explicit_msg_indices(tmp_path):
    # Missing GNPS query smiles may contain a forbidden answer; the
    # reader should never surface it as a candidate structure annotation.
    content=(
        "BEGIN IONS\nSOURCE_DATASET=GNPS\nINCHIKEY=SECRET\n"
        "FORMULA=C6H6\nSMILES=FORBIDDEN_QUERY_ANSWER\n100.0 50\nEND IONS\n"
        "BEGIN IONS\nSOURCE_DATASET=MassSpecGym\nINCHIKEY=REF\n"
        "FORMULA=C2H6\nSMILES=CC\n100.0 40\nEND IONS\n")
    path=tmp_path/"fixture.mgf.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(content.encode()))
    values=reference_smiles(path,{1},{1:{
        "inchikey":"REF","formula":"C2H6"}})
    assert values=={1:"CC"}
    with pytest.raises(RuntimeError):
        reference_smiles(path,{0},{0:{
            "inchikey":"SECRET","formula":"C6H6"}})


def test_exact_spectrum_duplicate_uses_both_mass_and_normalized_intensity():
    x=spectra([100.,120.],[1.,2.])
    y=spectra([100.,120.],[2.,4.])
    z=spectra([100.,121.],[1.,2.])
    assert query_reference_signal_equal(x,y)
    assert not query_reference_signal_equal(x,z)


def test_molecular_formula_is_cluster_unit():
    def row(formula,d):
        return {"formula":formula,
                "graph_mrr":.5+d,"hellinger_mrr":.5}
    o=block_inference([
        row("C2H6",.2),row("C2H6",.2),
        row("C3H8",-.1),row("C4H10",.1)])
    assert o["mean_mrr_delta"]==pytest.approx(.1)
    assert o["formula_block_bootstrap_95"]["n_blocks"]==3
    assert o["formula_block_signflip"]["n_blocks"]==3
