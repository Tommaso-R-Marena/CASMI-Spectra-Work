from qfd.pilot_eval import PilotRecord, split_query_reference_replicates
from qfd.trajectory import SpectrumView


def rec(key, sid, e):
    return PilotRecord(key, "C6H6", sid, "[M+H]+", float(e), SpectrumView.make([50, 100], [1, .5], [e], adduct="[M+H]+"))


def test_replicate_split_uses_distinct_rows_same_energies():
    records = []
    for e in [20, 40, 60]:
        records += [rec("A", f"a{e}x", e), rec("A", f"a{e}y", e)]
    g = split_query_reference_replicates(records, min_energies=2)["A"]
    qe = [s.collision_energy_ev[0] for s in g["query"]]
    re = [s.collision_energy_ev[0] for s in g["reference"]]
    assert qe == re == [20.0, 40.0, 60.0]
    assert all(q is not r for q, r in zip(g["query"], g["reference"]))
