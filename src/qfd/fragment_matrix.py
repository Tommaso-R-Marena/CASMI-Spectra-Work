from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .trajectory import SpectrumView


@dataclass(frozen=True)
class FragmentEnergyMatrix:
    fragment_mz: np.ndarray
    energies_ev: np.ndarray
    intensities: np.ndarray  # fragments x energies; NaN = not observed

    @property
    def mask(self):
        return np.isfinite(self.intensities)


def align_fragment_matrix(
    spectra: list[SpectrumView],
    *,
    tolerance_da: float = 0.01,
    min_presence: int = 2,
) -> FragmentEnergyMatrix:
    """Align single-energy spectra into a sparse fragment x energy matrix.

    Missing peaks remain NaN. Inputs are sorted by collision energy so output is
    deterministic. Multiple peaks from one spectrum are never allowed to occupy
    the same aligned fragment row.
    """
    single = [s for s in spectra if len(s.collision_energy_ev) == 1]
    if not single:
        return FragmentEnergyMatrix(np.empty(0), np.empty(0), np.empty((0, 0)))
    single = sorted(single, key=lambda s: s.collision_energy_ev[0])
    energies = np.array([s.collision_energy_ev[0] for s in single], dtype=float)

    peaks: list[tuple[float, int, float]] = []
    for j, s in enumerate(single):
        order = np.argsort(s.mzs)
        for k in order:
            mz = float(s.mzs[k])
            inten = float(s.intensities[k])
            if np.isfinite(mz) and np.isfinite(inten) and inten >= 0:
                peaks.append((mz, j, inten))
    peaks.sort(key=lambda x: x[0])

    clusters: list[dict] = []
    for mz, j, inten in peaks:
        if not clusters or abs(mz - clusters[-1]["centroid"]) > tolerance_da:
            clusters.append({"centroid": mz, "mzs": [mz], "values": {j: (mz, inten)}})
            continue
        c = clusters[-1]
        prev = c["values"].get(j)
        if prev is None or abs(mz - c["centroid"]) < abs(prev[0] - c["centroid"]):
            c["values"][j] = (mz, inten)
        c["mzs"].append(mz)
        c["centroid"] = float(np.mean(c["mzs"]))

    kept = [c for c in clusters if len(c["values"]) >= min_presence]
    X = np.full((len(kept), len(single)), np.nan, dtype=float)
    mzs = np.empty(len(kept), dtype=float)
    for i, c in enumerate(kept):
        mzs[i] = c["centroid"]
        for j, (_, inten) in c["values"].items():
            X[i, j] = inten
    return FragmentEnergyMatrix(mzs, energies, X)


def centered_log_fold_change(matrix: FragmentEnergyMatrix, *, eps: float = 1e-12):
    """Two-energy quotient signature on fragments present in both spectra."""
    X = matrix.intensities
    if X.shape[1] != 2:
        raise ValueError("centered_log_fold_change requires exactly two energies")
    common = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & (X[:, 0] > 0) & (X[:, 1] > 0)
    if not np.any(common):
        return np.empty(0), np.empty(0)
    r = np.log(np.maximum(X[common, 1], eps)) - np.log(np.maximum(X[common, 0], eps))
    r = r - r.mean()
    return matrix.fragment_mz[common], r
