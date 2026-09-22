from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit_massbank_postrelease_lockbox import REFERENCE_ENERGIES, _in_scope, _is_energy, parse_record
from qfd.consensus_ranker import CONSENSUS_FEATURE_NAMES, consensus_feature_matrix
from qfd.metrics import tie_aware_truth_metrics
from qfd.protected_tail_v53 import protected_tail_v53_scores
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie
from qfd.spectra import matched_cosine
from run_casmi2016_cfm_transfer import TOLERANCE_DA, _clean_raw, _sqrt_view
from run_casmi2016_trajectory_dev import _path_score


N_BOOT = 50_000
N_SIGNFLIP = 200_000
BOOT_SEED = 20260922
SIGNFLIP_SEED = 20260923
CONTROL_SEED = 20260924
V53 = {
    "overlap_threshold": 0.8,
    "protected_k": 1,
    "overlap_k": 10,
    "head_survival_slack": 1,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _value(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _parse_massbank_spectrum(path: Path) -> tuple[list[tuple[float, float]], float | None]:
    """Parse MassBank peaks using the second PK$PEAK column (absolute intensity).

    Cosine/path preprocessing is scale invariant after normalization. The absolute
    intensity column is prospectively fixed because it preserves more precision than
    the rounded relative-intensity column while retaining the same peak locations.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    precursor = None
    raw_prec = _value(lines, "MS$FOCUSED_ION: PRECURSOR_M/Z")
    if raw_prec:
        m = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", raw_prec)
        if m:
            precursor = float(m.group(0))

    start = next((i + 1 for i, line in enumerate(lines) if line.startswith("PK$PEAK:")), None)
    if start is None:
        raise ValueError(f"no PK$PEAK block in {path}")
    peaks: list[tuple[float, float]] = []
    number = re.compile(r"^\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s+([-+]?[0-9]+(?:\.[0-9]+)?)")
    for line in lines[start:]:
        if line.strip() == "//":
            break
        m = number.match(line)
        if not m:
            continue
        mz, intensity = float(m.group(1)), float(m.group(2))
        if math.isfinite(mz) and math.isfinite(intensity) and intensity >= 0:
            peaks.append((mz, intensity))
    if not peaks:
        raise ValueError(f"no usable peaks in {path}")
    return peaks, precursor


def _metric(ids: list[str], scores: np.ndarray, truth: str) -> dict[str, float]:
    vals = [float(x) if math.isfinite(float(x)) else -math.inf for x in scores]
    m = tie_aware_truth_metrics(ids, vals, truth, k=25)
    rev = tie_aware_truth_metrics(list(reversed(ids)), list(reversed(vals)), truth, k=25)
    if m != rev:
        raise RuntimeError("tie-aware metric changed under candidate-order reversal")
    return {
        "rr": float(m["rr_at_k"]),
        "top1": float(m["top1_credit"]),
        "top5": float(m["top5_credit"]),
        "expected_rank": float(m["expected_rank"]),
    }


def _top1_vector(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    top = s == np.max(s)
    out = np.zeros(len(s), dtype=float)
    out[top] = 1.0 / float(np.sum(top))
    return out


def _seed(text: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}\x1f{text}".encode()).digest()[:8], "big")


def _apply_v53(static: np.ndarray, expert: np.ndarray):
    return protected_tail_v53_scores(static, expert, **V53)


def _inference(rows: list[dict], method: str, baseline: str = "static", seed_offset: int = 0) -> dict:
    a = [r[f"rr_{method}"] for r in rows]
    b = [r[f"rr_{baseline}"] for r in rows]
    blocks = [r["formula"] for r in rows]
    return {
        "bootstrap": paired_block_bootstrap_delta(
            a, b, blocks, n_boot=N_BOOT, seed=BOOT_SEED + seed_offset
        ),
        "signflip": paired_block_signflip_test(
            a, b, blocks, n_permutations=N_SIGNFLIP,
            seed=SIGNFLIP_SEED + seed_offset, exact_max_blocks=20,
        ),
        "win_loss_tie": paired_win_loss_tie(a, b, atol=1e-15),
    }


def _summarize(rows: list[dict], method: str) -> dict:
    if not rows:
        return {"n_queries": 0, "n_formula_blocks": 0}
    return {
        "n_queries": len(rows),
        "n_formula_blocks": len({r["formula"] for r in rows}),
        "mrr25": float(np.mean([r[f"rr_{method}"] for r in rows])),
        "top1": float(np.mean([r[f"top1_{method}"] for r in rows])),
        "top5": float(np.mean([r[f"top5_{method}"] for r in rows])),
        "median_expected_rank": float(np.median([r[f"expected_rank_{method}"] for r in rows])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="One-shot frozen v5.3 whole-MassBank external lockbox")
    p.add_argument("massbank_root", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("frozen_model", type=Path)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--dataset-sha", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    got_manifest_sha = _sha256(args.manifest)
    if got_manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(f"manifest identity mismatch: {got_manifest_sha}")

    model = json.loads(args.frozen_model.read_text(encoding="utf-8"))
    if tuple(model.get("feature_names", ())) != CONSENSUS_FEATURE_NAMES:
        raise RuntimeError("frozen feature schema mismatch")
    if not bool(model.get("freeze_gate_passed", False)):
        raise RuntimeError("frozen expert did not pass its original freeze gate")
    weights = np.asarray(model.get("weights", ()), dtype=float)
    if weights.shape != (3,) or not np.all(np.isfinite(weights)):
        raise RuntimeError("invalid frozen consensus weights")

    manifest = pd.read_csv(args.manifest)
    required = {
        "formula", "truth_connectivity_key", "query_path", "query_contributor",
        "query_instrument_type", "candidate_count", "candidate_connectivity_keys",
    }
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"manifest missing {sorted(required - set(manifest.columns))}")
    if manifest["query_path"].duplicated().any():
        raise RuntimeError("duplicate query path")

    # Reconstruct the exact deterministic 10/20/40 reference map used by feasibility.
    paths = sorted(args.massbank_root.rglob("*.txt"))
    records = [r for path in paths if (r := parse_record(path, args.massbank_root)) is not None and _in_scope(r)]
    by_structure = defaultdict(list)
    for r in records:
        by_structure[r.connectivity_key].append(r)
    refs_by_structure = {}
    for key, rows in by_structure.items():
        refs = {}
        for energy in REFERENCE_ENERGIES:
            cands = sorted(
                (r for r in rows if _is_energy(r.collision_energy_ev, energy)),
                key=lambda r: (r.accession, r.path),
            )
            if cands:
                refs[energy] = cands[0].path
        if len(refs) == len(REFERENCE_ENERGIES):
            refs_by_structure[key] = refs

    base = []
    candidate_rows = []
    for qi, row in enumerate(manifest.itertuples(index=False), start=1):
        truth = str(row.truth_connectivity_key)
        ids = str(row.candidate_connectivity_keys).split(";")
        if truth not in ids or len(ids) != int(row.candidate_count):
            raise RuntimeError(f"candidate/truth mismatch for {row.query_path}")
        if any(cid not in refs_by_structure for cid in ids):
            raise RuntimeError(f"missing deterministic reference for {row.query_path}")

        q_peaks, q_prec = _parse_massbank_spectrum(args.massbank_root / str(row.query_path))
        q_raw = _clean_raw(q_peaks, q_prec, 0.0)
        q_sqrt = _sqrt_view(q_raw)

        cosines, path_scores = [], []
        for cid in ids:
            refs_raw = []
            cos = []
            for energy in REFERENCE_ENERGIES:
                rel = refs_by_structure[cid][energy]
                peaks, prec = _parse_massbank_spectrum(args.massbank_root / rel)
                raw = _clean_raw(peaks, prec, energy)
                refs_raw.append(raw)
                sr = _sqrt_view(raw)
                cos.append(float(matched_cosine(
                    q_sqrt.mzs, q_sqrt.intensities, sr.mzs, sr.intensities,
                    tolerance_da=TOLERANCE_DA,
                )))
            path_score, _, _ = _path_score(q_raw, refs_raw)
            if not math.isfinite(path_score):
                raise RuntimeError(f"non-finite path score for {row.query_path} / {cid}")
            cosines.append(cos)
            path_scores.append(float(path_score))

        C = np.asarray(cosines, dtype=float)
        static = np.max(C, axis=1)
        X = consensus_feature_matrix(C[:, 0], C[:, 1], C[:, 2], path_scores)
        expert = X @ weights
        chosen, overlap, used, protected, head_survives, max_rank = _apply_v53(static, expert)
        if not np.array_equal(_top1_vector(static), _top1_vector(chosen)):
            raise RuntimeError(f"candidatewise Top-1 invariant failed for {row.query_path}")

        rng = np.random.default_rng(_seed(str(row.query_path), "massbank-v53-shuffle"))
        shuffled_expert = expert[rng.permutation(len(expert))]
        inverted_expert = -expert
        shuffled, *_ = _apply_v53(static, shuffled_expert)
        inverted, *_ = _apply_v53(static, inverted_expert)

        base.append({
            "formula": str(row.formula),
            "query_path": str(row.query_path),
            "contributor": str(row.query_contributor),
            "instrument_type": str(row.query_instrument_type),
            "truth": truth,
            "ids": ids,
            "candidate_count": len(ids),
            "static": static,
            "expert": expert,
            "v53": chosen,
            "shuffle": shuffled,
            "inverted": inverted,
            "overlap": float(overlap),
            "used_v53": bool(used),
            "protected_count": int(np.sum(protected)),
            "head_survives": bool(head_survives),
            "max_protected_expert_rank": int(max_rank),
            "C": C,
            "path_scores": np.asarray(path_scores, dtype=float),
        })
        print(f"[{qi}/{len(manifest)}] {row.query_path} candidates={len(ids)} used={used}", flush=True)

    # Negative control 3: cyclically reassign the whole expert vector among queries
    # sharing a formula. This preserves candidate identities and formula-level score
    # structure while breaking the query-specific trajectory correspondence. Formula
    # singletons fall back to static and are excluded from this control's inference.
    by_formula = defaultdict(list)
    for i, item in enumerate(base):
        by_formula[item["formula"]].append(i)
    formula_perm_eligible = np.zeros(len(base), dtype=bool)
    formula_perm_scores = [item["static"].copy() for item in base]
    for formula, idxs in by_formula.items():
        idxs = sorted(idxs, key=lambda i: base[i]["query_path"])
        if len(idxs) < 2:
            continue
        for pos, i in enumerate(idxs):
            donor = idxs[(pos + 1) % len(idxs)]
            if base[i]["ids"] != base[donor]["ids"]:
                raise RuntimeError(f"formula candidate order mismatch: {formula}")
            score, *_ = _apply_v53(base[i]["static"], base[donor]["expert"])
            formula_perm_scores[i] = score
            formula_perm_eligible[i] = True

    rows = []
    for i, item in enumerate(base):
        score_sets = {
            "static": item["static"],
            "v53": item["v53"],
            "shuffle_control": item["shuffle"],
            "inverted_control": item["inverted"],
            "formula_perm_control": formula_perm_scores[i],
        }
        out = {
            "formula": item["formula"],
            "query_path": item["query_path"],
            "contributor": item["contributor"],
            "instrument_type": item["instrument_type"],
            "candidate_count": item["candidate_count"],
            "used_v53": item["used_v53"],
            "gate_overlap": item["overlap"],
            "protected_count": item["protected_count"],
            "head_survives": item["head_survives"],
            "formula_perm_control_eligible": bool(formula_perm_eligible[i]),
        }
        for method, scores in score_sets.items():
            m = _metric(item["ids"], np.asarray(scores, float), item["truth"])
            for k, v in m.items():
                out[f"{k}_{method}"] = v
        if out["top1_v53"] != out["top1_static"]:
            raise RuntimeError(f"Top-1 changed for {item['query_path']}")
        rows.append(out)

        for j, cid in enumerate(item["ids"]):
            candidate_rows.append({
                "formula": item["formula"],
                "query_path": item["query_path"],
                "candidate_connectivity_key": cid,
                "is_truth": int(cid == item["truth"]),
                "cos10": float(item["C"][j, 0]),
                "cos20": float(item["C"][j, 1]),
                "cos40": float(item["C"][j, 2]),
                "static_best": float(item["static"][j]),
                "ordered_path_score": float(item["path_scores"][j]),
                "expert_score": float(item["expert"][j]),
                "v53_score": float(item["v53"][j]),
            })

    methods = ("static", "v53", "shuffle_control", "inverted_control", "formula_perm_control")
    summaries = {m: _summarize(rows, m) for m in methods}
    primary_inf = _inference(rows, "v53", seed_offset=0)
    shuffle_inf = _inference(rows, "shuffle_control", seed_offset=1000)
    inverted_inf = _inference(rows, "inverted_control", seed_offset=2000)
    formula_rows = [r for r in rows if r["formula_perm_control_eligible"]]
    formula_perm_inf = _inference(formula_rows, "formula_perm_control", seed_offset=3000)

    # Direct control contrasts are stronger than merely asking whether controls are null.
    contrasts = {
        "v53_vs_shuffle_control": _inference(rows, "v53", "shuffle_control", 4000),
        "v53_vs_inverted_control": _inference(rows, "v53", "inverted_control", 5000),
        "v53_vs_formula_perm_control": _inference(formula_rows, "v53", "formula_perm_control", 6000),
    }

    pool5 = [r for r in rows if r["candidate_count"] >= 5]
    by_contributor = {}
    for contributor in sorted({r["contributor"] for r in rows}):
        subset = [r for r in rows if r["contributor"] == contributor]
        if len(subset) >= 20:
            by_contributor[contributor] = {
                "static": _summarize(subset, "static"),
                "v53": _summarize(subset, "v53"),
                "delta_mrr": float(np.mean([r["rr_v53"] - r["rr_static"] for r in subset])),
            }

    bootstrap_ok = float(primary_inf["bootstrap"]["ci_low"]) > 0.0
    signflip_ok = float(primary_inf["signflip"]["p_value_two_sided"]) < 0.05
    top1_ok = summaries["v53"]["top1"] == summaries["static"]["top1"]
    control_contrasts_ok = all(
        float(v["bootstrap"]["ci_low"]) > 0.0 for v in contrasts.values()
    )
    success = bool(bootstrap_ok and signflip_ok and top1_ok and control_contrasts_ok)

    result = {
        "status": "MASSBANK_WHOLE_LIBRARY_V53_ONE_SHOT_LOCKBOX",
        "evaluation_role": "prospectively_predeclared_external_performance_lockbox",
        "dataset": {
            "sha": args.dataset_sha,
            "manifest_sha256": got_manifest_sha,
            "queries": len(rows),
            "formula_blocks": len({r["formula"] for r in rows}),
        },
        "frozen_components": {
            "v53_parent_sha": "0006c7f81c7dfac53da48690ca0fe627108b9767",
            "v53": V53,
            "expert_weights": weights.tolist(),
            "expert_feature_names": list(CONSENSUS_FEATURE_NAMES),
            "massbank_peak_intensity_column": "PK$PEAK second column (absolute intensity)",
            "query_energy_ev": 30.0,
            "reference_energies_ev": list(REFERENCE_ENERGIES),
            "tolerance_da": TOLERANCE_DA,
        },
        "summaries": summaries,
        "primary_v53_vs_static": primary_inf,
        "negative_controls_vs_static": {
            "within_query_shuffle": shuffle_inf,
            "expert_inversion": inverted_inf,
            "within_formula_query_vector_permutation": formula_perm_inf,
        },
        "v53_vs_negative_controls": contrasts,
        "robustness": {
            "pool_ge_5": {
                "static": _summarize(pool5, "static"),
                "v53": _summarize(pool5, "v53"),
                "inference": _inference(pool5, "v53", seed_offset=7000),
            },
            "contributors_n_ge_20": by_contributor,
            "used_v53_queries": int(sum(r["used_v53"] for r in rows)),
            "used_v53_fraction": float(np.mean([r["used_v53"] for r in rows])),
        },
        "invariants": {
            "candidatewise_top1_violations": 0,
            "aggregate_top1_exactly_equal": bool(top1_ok),
        },
        "predeclared_success_gate": {
            "formula_block_bootstrap_ci_low_gt_0": bool(bootstrap_ok),
            "formula_block_signflip_p_lt_0_05": bool(signflip_ok),
            "top1_exactly_invariant": bool(top1_ok),
            "v53_beats_all_three_controls_bootstrap_ci_low_gt_0": bool(control_contrasts_ok),
            "PASS": bool(success),
        },
        "guardrail": (
            "This is an external measured-spectrum benchmark for the frozen CASMI-trained expert and NIST-developed v5.3 selector. "
            "It tests transfer across MassBank contributors/instruments under same-formula measured-reference identification. "
            "It does not establish universal metabolite identification or performance on candidate pools larger than those present here."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_query.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(args.out_dir / "candidate_scores.csv.gz", index=False, compression="gzip")
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
