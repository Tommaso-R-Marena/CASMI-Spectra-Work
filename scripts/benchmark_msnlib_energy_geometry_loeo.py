from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qfd.energy_geometry import energy_prediction_scores_strict
from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _clean_view, _unambiguous_groups, index_records
from qfd.stats import paired_block_bootstrap_delta, paired_block_signflip_test, paired_win_loss_tie


OFFCENTER_ATOL = 1e-12


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
        'win_loss_tie': paired_win_loss_tie(av, bv, atol=1e-15),
        'formula_block_bootstrap_95': paired_block_bootstrap_delta(
            av, bv, blocks, n_boot=10_000, seed=seed,
        ),
        'formula_block_signflip_two_sided': paired_block_signflip_test(
            av, bv, blocks, seed=seed, n_permutations=200_000,
        ),
    }


def _method_summary(rows, key: str) -> dict[str, float]:
    v = np.asarray([float(r[key]) for r in rows], dtype=float)
    return {
        'mean_affinity': float(np.mean(v)),
        'median_affinity': float(np.median(v)),
        'q10_affinity': float(np.quantile(v, 0.10)),
        'q90_affinity': float(np.quantile(v, 0.90)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description='Prospective generalized collision-energy LOEO prediction benchmark.')
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
        ref_energies = sorted(float(e) for e in g['same_energy'])
        if len(ref_energies) < 2:
            continue
        for eq in sorted(float(e) for e in g['single_best']):
            lower = [e for e in ref_energies if e < eq]
            higher = [e for e in ref_energies if e > eq]
            if not lower or not higher:
                continue
            elo = max(lower)
            ehi = min(higher)
            query = _clean_view(g['single_best'][eq], transform='none')
            low = _clean_view(g['same_energy'][elo], transform='none')
            high = _clean_view(g['same_energy'][ehi], transform='none')
            s = energy_prediction_scores_strict(query, low, high)
            rows.append({
                'structure_adduct': f'{skey[0]}|{skey[1]}',
                'full_inchikey': str(skey[0]),
                'adduct': str(skey[1]),
                'formula': str(g['formula']),
                'query_energy_ev': s.query_energy_ev,
                'low_energy_ev': s.low_energy_ev,
                'high_energy_ev': s.high_energy_ev,
                't': s.t,
                'offcenter': abs(s.t - 0.5) > OFFCENTER_ATOL,
                'best_endpoint': s.best_endpoint,
                'low_endpoint': s.low_endpoint,
                'high_endpoint': s.high_endpoint,
                'linear': s.linear,
                'equal_mix': s.equal_mix,
                'reversed_energy': s.reversed_energy,
                'hellinger_geodesic': s.hellinger_geodesic,
            })

    if not rows:
        raise RuntimeError('no eligible generalized LOEO query triplets')
    offcenter = [r for r in rows if bool(r['offcenter'])]

    methods = ['best_endpoint', 'linear', 'equal_mix', 'reversed_energy', 'hellinger_geodesic']
    metrics = {m: _method_summary(rows, m) for m in methods}
    metrics_offcenter = {m: _method_summary(offcenter, m) for m in methods} if offcenter else {}

    comps = {}
    specs = [
        ('linear_vs_best_endpoint', rows, 'linear', 'best_endpoint'),
        ('hellinger_geodesic_vs_best_endpoint', rows, 'hellinger_geodesic', 'best_endpoint'),
        ('hellinger_geodesic_vs_linear', rows, 'hellinger_geodesic', 'linear'),
        ('linear_vs_equal_mix_offcenter', offcenter, 'linear', 'equal_mix'),
        ('linear_vs_reversed_energy_offcenter', offcenter, 'linear', 'reversed_energy'),
        ('hellinger_geodesic_vs_equal_mix_offcenter', offcenter, 'hellinger_geodesic', 'equal_mix'),
        ('hellinger_geodesic_vs_reversed_energy_offcenter', offcenter, 'hellinger_geodesic', 'reversed_energy'),
    ]
    for i, (name, subset, a, b) in enumerate(specs):
        if not subset:
            comps[name] = {'status': 'NO_ELIGIBLE_QUERIES'}
        else:
            comps[name] = _comparison(subset, a, b, seed=args.seed + i)

    energy_triplet_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        key = f"{float(r['low_energy_ev']):g}->{float(r['query_energy_ev']):g}->{float(r['high_energy_ev']):g}"
        energy_triplet_counts[key] += 1

    result = {
        'status': 'MSNLIB_ENERGY_GEOMETRY_LOEO_BENCHMARK',
        'dataset': args.dataset,
        'input': {'path': str(args.jsonl), 'sha256': _sha256(args.jsonl), 'records': len(records)},
        'protocol': {
            'selection': 'each SINGLE_BEST_SCAN internal energy; nearest lower/higher SAME_ENERGY references; exact query energy withheld',
            'mz_alignment': 'strict diameter-bounded 0.01 Da clusters; no transitive chaining',
            'normalization': 'L1 probability distribution after alignment',
            'similarity': 'Bhattacharyya affinity',
            'linear': '(1-t)p_low + t p_high, t determined only by collision energies',
            'equal_mix': '0.5 p_low + 0.5 p_high',
            'reversed_energy': 't p_low + (1-t) p_high',
            'hellinger_geodesic': 'great-circle interpolation after square-root probability embedding',
            'inference_unit': 'molecular formula block',
        },
        'coverage': {
            'query_triplets': len(rows),
            'structure_adduct_groups': len({str(r['structure_adduct']) for r in rows}),
            'formula_blocks': len({str(r['formula']) for r in rows}),
            'offcenter_triplets': len(offcenter),
            'offcenter_formula_blocks': len({str(r['formula']) for r in offcenter}),
            'energy_triplet_histogram': dict(sorted(energy_triplet_counts.items())),
        },
        'metrics_all': metrics,
        'metrics_offcenter': metrics_offcenter,
        'inference': comps,
        'query_diagnostics': rows,
        'guardrail': (
            'All predictors, energy-selection rules, controls, strict m/z alignment, and inferential criteria were frozen '
            'before this dataset was scored under the generalized LOEO protocol.'
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'query_diagnostics'}, indent=2))


if __name__ == '__main__':
    main()
