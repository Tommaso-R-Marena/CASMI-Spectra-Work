from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .casmi_schema import normalize_energy_list
from .spectra import matched_cosine


@dataclass(frozen=True)
class SpectrumView:
    mzs: np.ndarray
    intensities: np.ndarray
    collision_energy_ev: tuple[float, ...]
    adduct: str | None = None
    base_peak_intensity: float | None = None

    @staticmethod
    def make(mzs, intensities, collision_energy_ev, adduct=None, base_peak_intensity=None):
        return SpectrumView(
            np.asarray(mzs, dtype=float),
            np.asarray(intensities, dtype=float),
            normalize_energy_list(collision_energy_ev),
            adduct,
            None if base_peak_intensity is None else float(base_peak_intensity),
        )


def _energy_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != 1 or len(b) != 1:
        return math.inf
    return abs(a[0] - b[0])


def joint_energy_score(
    query: list[SpectrumView],
    reference: list[SpectrumView],
    *,
    tolerance_da: float = 0.01,
    max_energy_gap_ev: float = 5.0,
    require_same_adduct: bool = True,
    use_base_peak_weights: bool = False,
) -> float:
    """Aggregate matched spectral evidence while preserving conditions.

    Single-energy query spectra are paired to the closest compatible reference
    spectra. Merged-energy spectra are deliberately excluded and must be scored
    with the cone/convex-hull machinery.
    """
    vals: list[float] = []
    weights: list[float] = []
    for q in query:
        if len(q.collision_energy_ev) != 1:
            continue
        choices = []
        for r in reference:
            if len(r.collision_energy_ev) != 1:
                continue
            if require_same_adduct and q.adduct != r.adduct:
                continue
            gap = _energy_distance(q.collision_energy_ev, r.collision_energy_ev)
            if gap <= max_energy_gap_ev:
                choices.append((gap, r))
        if not choices:
            continue
        _, r = min(choices, key=lambda x: x[0])
        vals.append(matched_cosine(q.mzs, q.intensities, r.mzs, r.intensities, tolerance_da=tolerance_da))
        if use_base_peak_weights and q.base_peak_intensity is not None and q.base_peak_intensity > 0:
            weights.append(math.sqrt(q.base_peak_intensity))
        else:
            weights.append(1.0)
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=np.asarray(weights)))
