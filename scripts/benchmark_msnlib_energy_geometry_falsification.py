from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qfd.energy_geometry import energy_prediction_scores_strict
from qfd.energy_geometry_controls import geometry_control_scores
from qfd.energy_interpolation import _common_grid_strict, _l1_columns
from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _comparison(rows, a: str, b: str, *, seed: int) -> dict[str, object]:
    av = np.asarray([float(r[a]) for r in rows], dtype=float)
    bv = np.asarray([float(r[b]) for r in rows], dtype=float)
    blocks = [str(r['formula']) for r in rows]
    return {
        'a': a,
        'b': b,
        'n_queries': len(rows),
        'n_formula_blocks': len(set(blocks)),
        'mean_delta': float(np.mean(av - bv)),
        'median_delta': float(np.median(av - bv)),
        'win_loss_tie': paired_win_loss_tie(av, bv, atol=1e-15),
        'formula_block_bootstrap_95': paired_block_bootstrap_delta(av, bv, blocks, n_boot=10_000, seed=seed),
        'formula_block_signflip_two_sided': paired_block_signflip_test(av, bv, blocks, seed=seed, n_permutations=200_000),
    }


def _method_summary(rows, key: str) -> dict[str, float]:
    x = np.asarray([float(r[key]) for r in rows], dtype=float)
    return {
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'q10': float(np.quantile(x, 0.10)),
        'q90': float(np.quantile(x, 0.90)),
    }


def _run_comparisons(rows, *, seed: int) -> dict[str, object]:
    specs = [
        ('geodesic_vs_hellinger_chord', 'hellinger_geodesic', 'hellinger_chord'),
        ('geodesic_vs_matched_coeff_mixture', 'hellinger_geodesic', 'matched_coeff_mixture'),
        ('geodesic_vs_orthogonal_warp_mixture', 'hellinger_geodesic', 'orthogonal_warp_mixture'),
        ('geodesic_vs_oracle_probability_mixture', 'hellinger_geodesic', 'oracle_probability_mixture'),
        ('oracle_geodesic_vs_fixed_geodesic', 'oracle_geodesic', 'hellinger_geodesic'),
        ('hellinger_chord_vs_oracle_probability_mixture', 'hellinger_chord', 'oracle_probability_mixture'),
        ('matched_coeff_mixture_vs_linear', 'matched_coeff_mixture', 'linear'),
    ]
    return {
        name: _comparison(rows, a, b, seed=seed + i)
        for i, (name, a, b) in enumerate(specs)
    }


def main() -> None:
    p = argparse.ArgumentParser(description='Post-hoc falsification/decomposition of the frozen MCEBIO energy-geometry result.')
    p.add_argument('jsonl', type=Path)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260922)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)

    rows: list[dict[str, object]] = []
    for skey, g in sorted(groups.items()):
        refs = sorted(float(e) for e in g['same_energy'])
        if len(refs) < 2:
            continue
        for eq in sorted(float(e) for e in g['single_best']):
            lower = [e for e in refs if e < eq]
            higher = [e for e in refs if e > eq]
            if not lower or not higher:
                continue
            elo = max(lower)
            ehi = min(higher)
            query = _clean_view(g['single_best'][eq], transform='none')
            low = _clean_view(g['same_energy'][elo], transform='none')
            high = _clean_view(g['same_energy'][ehi], transform='none')

            base = energy_prediction_scores_strict(query, low, high)
            X = _l1_columns(_common_grid_strict([query, low, high], 0.01))
            if X is None:
                raise ValueError('strict grid failed to produce normalized triplet')
            q, p0, p1 = X[:, 0], X[:, 1], X[:, 2]
            ctl = geometry_control_scores(q, p0, p1, base.t)

            rows.append({
                'structure_adduct': f'{skey[0]}|{skey[1]}',
                'formula': str(g['formula']),
                'query_energy_ev': base.query_energy_ev,
                'low_energy_ev': base.low_energy_ev,
                'high_energy_ev': base.high_energy_ev,
                't': base.t,
                'best_endpoint': base.best_endpoint,
                'linear': base.linear,
                'hellinger_geodesic': base.hellinger_geodesic,
                'hellinger_chord': ctl.hellinger_chord,
                'matched_coeff_mixture': ctl.matched_coeff_mixture,
                'orthogonal_warp_mixture': ctl.orthogonal_warp_mixture,
                'oracle_probability_mixture': ctl.oracle_probability_mixture,
                'oracle_probability_weight': ctl.oracle_probability_weight,
                'oracle_geodesic': ctl.oracle_geodesic,
                'oracle_geodesic_coordinate': ctl.oracle_geodesic_coordinate,
                'oracle_geodesic_coordinate_error': abs(ctl.oracle_geodesic_coordinate - base.t),
            })

    if not rows:
        raise RuntimeError('no eligible LOEO triplets')

    methods = [
        'best_endpoint', 'linear', 'hellinger_geodesic', 'hellinger_chord',
        'matched_coeff_mixture', 'orthogonal_warp_mixture',
        'oracle_probability_mixture', 'oracle_geodesic',
    ]
    by_triplet: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        key = f"{float(r['low_energy_ev']):g}->{float(r['query_energy_ev']):g}->{float(r['high_energy_ev']):g}"
        by_triplet[key].append(r)

    t = np.asarray([float(r['t']) for r in rows])
    sstar = np.asarray([float(r['oracle_geodesic_coordinate']) for r in rows])
    wstar = np.asarray([float(r['oracle_probability_weight']) for r in rows])
    coord_error = np.abs(sstar - t)

    triplet_results = {}
    for j, (key, rr) in enumerate(sorted(by_triplet.items())):
        triplet_results[key] = {
            'n_queries': len(rr),
            'n_formula_blocks': len({str(r['formula']) for r in rr}),
            't': float(rr[0]['t']),
            'methods': {m: _method_summary(rr, m) for m in methods},
            'comparisons': _run_comparisons(rr, seed=args.seed + 100 + 20 * j),
            'coordinate_error_mean': float(np.mean([float(r['oracle_geodesic_coordinate_error']) for r in rr])),
            'coordinate_error_median': float(np.median([float(r['oracle_geodesic_coordinate_error']) for r in rr])),
        }

    result = {
        'status': 'MSNLIB_ENERGY_GEOMETRY_POSTHOC_FALSIFICATION',
        'dataset': args.dataset,
        'input': {'path': str(args.jsonl), 'sha256': _sha256(args.jsonl), 'records': len(records)},
        'protocol': {
            'source_plan': 'docs/ENERGY_GEOMETRY_POSTHOC_FALSIFICATION.md',
            'cohort': 'identical generalized LOEO selection rule as frozen prospective benchmark',
            'mz_alignment': 'strict diameter-bounded 0.01 Da clusters',
            'note': 'post-hoc mechanism/robustness audit; not an independent confirmatory lockbox',
        },
        'coverage': {
            'query_triplets': len(rows),
            'formula_blocks': len({str(r['formula']) for r in rows}),
            'structure_adduct_groups': len({str(r['structure_adduct']) for r in rows}),
            'energy_triplet_histogram': {k: len(v) for k, v in sorted(by_triplet.items())},
        },
        'methods_all': {m: _method_summary(rows, m) for m in methods},
        'comparisons_all': _run_comparisons(rows, seed=args.seed),
        'energy_coordinate_diagnostics': {
            'mean_abs_sstar_minus_t': float(np.mean(coord_error)),
            'median_abs_sstar_minus_t': float(np.median(coord_error)),
            'q90_abs_sstar_minus_t': float(np.quantile(coord_error, 0.90)),
            'pearson_t_sstar': float(np.corrcoef(t, sstar)[0, 1]) if np.std(t) > 0 and np.std(sstar) > 0 else None,
            'mean_oracle_probability_weight': float(np.mean(wstar)),
            'median_oracle_probability_weight': float(np.median(wstar)),
        },
        'by_energy_triplet': triplet_results,
        'query_diagnostics': rows,
        'guardrail': 'Controls were committed in the post-hoc falsification plan before this script was run on MCEBIO.',
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'query_diagnostics', 'by_energy_triplet'}}, indent=2))


if __name__ == '__main__':
    main()
