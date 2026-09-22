from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from qfd.msnlib import load_msnlib_jsonl
from qfd.msnlib_pilot import _unambiguous_groups, index_records


OFFCENTER_ATOL = 1e-12


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _eligible_triplets(groups):
    rows = []
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
            t = (eq - elo) / (ehi - elo)
            rows.append({
                'structure_adduct': f'{skey[0]}|{skey[1]}',
                'formula': str(g['formula']),
                'query_energy_ev': float(eq),
                'low_energy_ev': float(elo),
                'high_energy_ev': float(ehi),
                't': float(t),
                'offcenter': abs(float(t) - 0.5) > OFFCENTER_ATOL,
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description='Label-free feasibility audit for generalized collision-energy LOEO prediction.')
    p.add_argument('jsonl', type=Path)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    records = load_msnlib_jsonl(args.jsonl)
    indexed = index_records(records)
    groups = _unambiguous_groups(indexed)
    rows = _eligible_triplets(groups)

    structures = {r['structure_adduct'] for r in rows}
    formulas = {r['formula'] for r in rows}
    offcenter = [r for r in rows if r['offcenter']]
    triplets = Counter(
        f"{r['low_energy_ev']:g}->{r['query_energy_ev']:g}->{r['high_energy_ev']:g}"
        for r in rows
    )
    t_hist = Counter(f"{r['t']:.12g}" for r in rows)

    gate = {
        'query_triplets_min': 1000,
        'structure_adduct_groups_min': 500,
        'formula_blocks_min': 300,
        'offcenter_triplets_min': 500,
    }
    coverage = {
        'query_triplets': len(rows),
        'structure_adduct_groups': len(structures),
        'formula_blocks': len(formulas),
        'offcenter_triplets': len(offcenter),
        'offcenter_fraction': (len(offcenter) / len(rows)) if rows else 0.0,
        'energy_triplet_histogram': dict(sorted(triplets.items())),
        't_histogram': dict(sorted(t_hist.items())),
    }
    gate_pass = bool(
        coverage['query_triplets'] >= gate['query_triplets_min']
        and coverage['structure_adduct_groups'] >= gate['structure_adduct_groups_min']
        and coverage['formula_blocks'] >= gate['formula_blocks_min']
        and coverage['offcenter_triplets'] >= gate['offcenter_triplets_min']
    )

    out = {
        'status': 'MSNLIB_ENERGY_GEOMETRY_LOEO_LABEL_FREE_FEASIBILITY',
        'dataset': args.dataset,
        'input': {'path': str(args.jsonl), 'sha256': _sha256(args.jsonl), 'records': len(records)},
        'identity': 'full InChIKey + adduct',
        'selection_rule': (
            'For each SINGLE_BEST_SCAN query energy, choose nearest lower and nearest higher SAME_ENERGY references; '
            'withhold the query energy itself. No spectrum similarity or outcome is computed.'
        ),
        'coverage': coverage,
        'predeclared_gate': gate,
        'gate_pass': gate_pass,
        'guardrail': 'This audit imports no spectral prediction/scoring function and computes no similarity or efficacy outcome.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
