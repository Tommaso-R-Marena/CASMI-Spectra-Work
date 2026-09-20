from qfd.pilot_eval import PilotRecord, evaluate_formula_candidates, split_query_reference, summarize_results
from qfd.trajectory import SpectrumView


def rec(key, sid, e, inten, formula="C6H6"):
    v = SpectrumView.make([50, 100, 150], inten, [e], adduct="[M+H]+")
    return PilotRecord(key, formula, sid, "[M+H]+", float(e), v)


def test_split_is_energy_disjoint():
    records = []
    for i, e in enumerate([10, 20, 30, 40]):
        records.append(rec("A", f"a{i}", e, [1, 0.5, 0.25]))
    groups = split_query_reference(records, min_energies_each=2)
    q = {v.collision_energy_ev[0] for v in groups["A"]["query"]}
    r = {v.collision_energy_ev[0] for v in groups["A"]["reference"]}
    assert q.isdisjoint(r)
    assert q | r == {10, 20, 30, 40}


def test_formula_candidate_eval_returns_all_modes():
    records = []
    for i, (e, a, b) in enumerate([
        (10, [1, .5, .25], [1, .5, .25]),
        (20, [.8, .7, .25], [1, .5, .25]),
        (30, [.5, 1, .25], [1, .5, .25]),
        (40, [.3, 1, .25], [1, .5, .25]),
    ]):
        records += [rec("A", f"a{i}", e, a), rec("B", f"b{i}", e, b)]
    groups = split_query_reference(records, min_energies_each=2)
    rows = evaluate_formula_candidates(groups, max_energy_gap_ev=15)
    summary = summarize_results(rows)
    assert summary["n_queries"] == 2
    for mode in ("static_best", "joint_energy", "two_energy_qfd"):
        assert f"mrr25_{mode}" in summary
