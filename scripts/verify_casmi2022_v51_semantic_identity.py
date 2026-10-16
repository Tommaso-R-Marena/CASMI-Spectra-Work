from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.consensus_ranker import protected_tail_consensus_scores


EXPECTED = {
    "queries": 169,
    "candidate_rows": 1_023_662,
    "historical_raw_gzip_sha256": "ae786e96c7099a07c6a80c50e51bc8148712161e3a7b6bd620bf5734b285c7db",
    "replay_raw_gzip_sha256": "b605b87cab7c4259fbf1279c25e4ca16043b72273c04d7a0a564a7e1dd90e718",
    "replay_uncompressed_sha256": "3820e8e29c170e6a441a52c66a3851c32c316346bb543f3407b8a8d155aac4fd",
    "static_mrr25": 0.059153276739569056,
    "protected_mrr25": 0.05996885429666056,
    "static_top1": 0.017422748191978962,
    "protected_top1": 0.017422748191978962,
    "static_top5": 0.09680119354675568,
    "protected_top5": 0.09555201537450059,
    "static_median_expected_rank": 117.0,
    "protected_median_expected_rank": 102.5,
    "wins": 7,
    "losses": 5,
    "ties": 157,
    "used_queries": 38,
    "mean_protected_count": 1.6272189349112427,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _gzip_payload_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tie_metrics(scores: np.ndarray, truth: np.ndarray, cutoff: int = 25) -> dict[str, float]:
    truth_idx = np.flatnonzero(np.asarray(truth, dtype=bool))
    if truth_idx.size != 1:
        raise ValueError(f"expected exactly one truth candidate, found {truth_idx.size}")
    s = np.asarray(scores, dtype=float)
    truth_score = float(s[int(truth_idx[0])])
    better = int(np.sum(s > truth_score))
    tied = int(np.sum(s == truth_score))
    ranks = np.arange(better + 1, better + tied + 1, dtype=float)
    return {
        "rr": float(np.mean(np.where(ranks <= cutoff, 1.0 / ranks, 0.0))),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "expected_rank": float(np.mean(ranks)),
    }


def _assert_close(name: str, observed: float, expected: float, *, atol: float = 2e-15) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=atol):
        raise RuntimeError(f"semantic provenance mismatch for {name}: {observed!r} != {expected!r}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Verify that a regenerated CASMI2022 table reproduces the complete historical v5.1 ranking audit."
    )
    p.add_argument("candidate_features", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overlap-threshold", type=float, default=0.8)
    p.add_argument("--protected-k", type=int, default=1)
    p.add_argument("--overlap-k", type=int, default=10)
    args = p.parse_args()

    raw_sha = _sha256(args.candidate_features)
    uncompressed_sha = _gzip_payload_sha256(args.candidate_features)
    df = pd.read_csv(args.candidate_features)
    required = {"spec_id", "is_truth", "static_best", "expert_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"missing candidate columns: {missing}")

    if len(df) != EXPECTED["candidate_rows"]:
        raise RuntimeError(f"candidate row mismatch: {len(df)} != {EXPECTED['candidate_rows']}")
    n_queries = int(df["spec_id"].nunique())
    if n_queries != EXPECTED["queries"]:
        raise RuntimeError(f"query count mismatch: {n_queries} != {EXPECTED['queries']}")
    if raw_sha != EXPECTED["replay_raw_gzip_sha256"]:
        raise RuntimeError(
            "saved replay artifact raw SHA changed; this verifier is pinned to the audited v5.3 source artifact"
        )
    if uncompressed_sha != EXPECTED["replay_uncompressed_sha256"]:
        raise RuntimeError("saved replay artifact decompressed CSV SHA changed")

    static_rows: list[dict[str, float]] = []
    protected_rows: list[dict[str, float]] = []
    used_queries = 0
    protected_counts: list[int] = []
    deltas: list[float] = []

    for _, g in df.groupby("spec_id", sort=False):
        static = g["static_best"].to_numpy(dtype=float)
        expert = g["expert_score"].to_numpy(dtype=float)
        truth = g["is_truth"].to_numpy(dtype=bool)
        chosen, _, used, protected = protected_tail_consensus_scores(
            static,
            expert,
            overlap_threshold=args.overlap_threshold,
            protected_k=args.protected_k,
            overlap_k=args.overlap_k,
        )
        sm = _tie_metrics(static, truth)
        pm = _tie_metrics(chosen, truth)
        static_rows.append(sm)
        protected_rows.append(pm)
        used_queries += int(used)
        protected_counts.append(int(np.sum(protected)))
        deltas.append(float(pm["rr"] - sm["rr"]))

    observed = {
        "static_mrr25": float(np.mean([r["rr"] for r in static_rows])),
        "protected_mrr25": float(np.mean([r["rr"] for r in protected_rows])),
        "static_top1": float(np.mean([r["top1"] for r in static_rows])),
        "protected_top1": float(np.mean([r["top1"] for r in protected_rows])),
        "static_top5": float(np.mean([r["top5"] for r in static_rows])),
        "protected_top5": float(np.mean([r["top5"] for r in protected_rows])),
        "static_median_expected_rank": float(np.median([r["expected_rank"] for r in static_rows])),
        "protected_median_expected_rank": float(np.median([r["expected_rank"] for r in protected_rows])),
        "wins": int(np.sum(np.asarray(deltas) > 1e-15)),
        "losses": int(np.sum(np.asarray(deltas) < -1e-15)),
        "ties": int(np.sum(np.abs(np.asarray(deltas)) <= 1e-15)),
        "used_queries": int(used_queries),
        "mean_protected_count": float(np.mean(protected_counts)),
    }

    for key in (
        "static_mrr25",
        "protected_mrr25",
        "static_top1",
        "protected_top1",
        "static_top5",
        "protected_top5",
        "static_median_expected_rank",
        "protected_median_expected_rank",
        "mean_protected_count",
    ):
        _assert_close(key, float(observed[key]), float(EXPECTED[key]))
    for key in ("wins", "losses", "ties", "used_queries"):
        if int(observed[key]) != int(EXPECTED[key]):
            raise RuntimeError(f"semantic provenance mismatch for {key}: {observed[key]} != {EXPECTED[key]}")

    result = {
        "status": "CASMI2022_V51_SEMANTIC_IDENTITY_CONFIRMED",
        "candidate_source": str(args.candidate_features),
        "queries": n_queries,
        "candidate_rows": int(len(df)),
        "raw_gzip_identity": {
            "historical_v51_sha256": EXPECTED["historical_raw_gzip_sha256"],
            "replay_artifact_sha256": raw_sha,
            "equal": bool(raw_sha == EXPECTED["historical_raw_gzip_sha256"]),
            "interpretation": "Raw compressed-file identity is not established and is not used as the semantic provenance criterion.",
        },
        "replay_artifact_fingerprint": {
            "raw_gzip_sha256": raw_sha,
            "decompressed_csv_sha256": uncompressed_sha,
        },
        "historical_v51_expected": {k: v for k, v in EXPECTED.items() if not k.endswith("sha256")},
        "observed_v51_replay": observed,
        "semantic_identity": True,
        "interpretation": (
            "The regenerated table reproduces every recorded historical v5.1 scientific audit quantity checked here, "
            "including static/protected MRR, Top-1, Top-5, median rank, wins/losses/ties, selector usage, and mean protected-head size. "
            "This establishes semantic identity for the prior ranking experiment, not byte-level identity with the unavailable historical table."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
