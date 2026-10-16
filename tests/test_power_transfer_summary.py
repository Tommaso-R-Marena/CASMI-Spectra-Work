import json

from scripts.summarize_power_transfer import COHORTS, PREFIX, holm_adjust, markdown, summarize


def _write(root, dataset, *, gate=True, scores=True, good=True):
    target = root / (PREFIX + dataset)
    target.mkdir()
    (target / "feasibility.json").write_text(json.dumps({
        "dataset": dataset, "gate_pass": gate, "coverage": {"query_triplets": 1000},
    }))
    if not gate or not scores:
        return
    (target / "provenance.json").write_text(json.dumps({
        "dataset": dataset,
        "frozen_transfer_sha": "be7813155a5dada31c8c557952a060b664a1e3ac",
    }))
    delta = 0.01 if good else -0.01
    (target / "result.json").write_text(json.dumps({
        "dataset": dataset, "coverage": {"query_triplets": 1000},
        "frozen_parameters": {"alpha_quarter": 0.25},
        "methods": {"alpha_quarter": {"mean": 0.90}, "hellinger_chord": {"mean": 0.89}},
        "comparisons": {
            "alpha_quarter_vs_hellinger": {
                "mean_delta": delta, "n_queries": 1000, "n_formula_blocks": 600,
                "formula_block_bootstrap_95": {
                    "ci_low": delta - 0.001, "ci_high": delta + 0.001
                },
                "formula_block_signflip_two_sided": {"p_value_two_sided": 0.001},
            },
            "alpha_quarter_vs_oracle_probability_mixture": {"mean_delta": 0.02}
        },
    }))


def test_holm_is_conservative_for_unopened_cohorts():
    assert holm_adjust({"a": 0.001, "b": 0.01}, 8) == {"a": 0.008, "b": 0.07}


def test_incomplete_is_not_silent_success(tmp_path):
    _write(tmp_path, COHORTS[0], gate=True)
    _write(tmp_path, COHORTS[1], gate=False)
    report = summarize(tmp_path)
    statuses = [x["status"] for x in report["cohorts"]]
    assert statuses[:3] == ["SCORED", "GATE_FAILED_UNOPENED", "MISSING_FEASIBILITY_ARTIFACT"]
    assert report["status"] == "INCOMPLETE_OR_INVALID"
    assert report["cohorts"][0]["holm_supported"]
    assert "MISSING_FEASIBILITY_ARTIFACT" in markdown(report)


def test_complete_when_all_scored_or_failed_gate(tmp_path):
    for i, dataset in enumerate(COHORTS):
        _write(tmp_path, dataset, gate=(i < 4), good=(i != 3))
    report = summarize(tmp_path)
    assert report["status"] == "COMPLETE"
    assert report["counts"]["SCORED"] == 4
    assert report["counts"]["GATE_FAILED_UNOPENED"] == 4
    assert report["cohorts"][3]["positive_predeclared"] is False


def test_failed_gate_cannot_open_scores(tmp_path):
    dataset = COHORTS[0]
    _write(tmp_path, dataset, gate=False)
    (tmp_path / (PREFIX + dataset) / "result.json").write_text("{}")
    assert summarize(tmp_path)["cohorts"][0]["status"] == "PROTOCOL_BREACH_SCORED_FAILED_GATE"


def test_provenance_mismatch_is_not_scientific_failure(tmp_path):
    dataset = COHORTS[0]
    _write(tmp_path, dataset)
    p = tmp_path / (PREFIX + dataset) / "provenance.json"
    x = json.loads(p.read_text())
    x["frozen_transfer_sha"] = "not-frozen"
    p.write_text(json.dumps(x))
    assert summarize(tmp_path)["cohorts"][0]["status"] == "FROZEN_PARAMETER_MISMATCH"
