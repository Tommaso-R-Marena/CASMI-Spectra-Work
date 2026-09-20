from __future__ import annotations

import numpy as np


def preprocess_spectrum(
    mzs,
    intensities,
    *,
    precursor_mz: float | None = None,
    intensity_floor: float = 0.0,
    max_peaks: int | None = 128,
    precursor_margin: float = 2.0,
    intensity_transform: str = "sqrt",
):
    mz = np.asarray(mzs, dtype=float)
    it = np.asarray(intensities, dtype=float)
    if mz.ndim != 1 or it.ndim != 1 or len(mz) != len(it):
        raise ValueError("aligned one-dimensional peak arrays required")
    good = np.isfinite(mz) & np.isfinite(it) & (it >= 0)
    if precursor_mz is not None and np.isfinite(float(precursor_mz)):
        good &= mz <= float(precursor_mz) + float(precursor_margin)
    mz, it = mz[good], it[good]
    if len(it) == 0:
        return mz, it
    max_i = float(it.max())
    if max_i > 0 and intensity_floor > 0:
        keep = it >= intensity_floor * max_i
        mz, it = mz[keep], it[keep]
    if max_peaks is not None and len(it) > max_peaks:
        idx = np.argpartition(it, -max_peaks)[-max_peaks:]
        mz, it = mz[idx], it[idx]
    order = np.argsort(mz)
    mz, it = mz[order], it[order]
    if intensity_transform == "sqrt":
        it = np.sqrt(it)
    elif intensity_transform == "log1p":
        it = np.log1p(it)
    elif intensity_transform == "none":
        pass
    else:
        raise ValueError("intensity_transform must be sqrt, log1p, or none")
    norm = np.linalg.norm(it)
    if norm > 0:
        it = it / norm
    return mz, it


def match_peaks(mz_a, mz_b, *, tolerance_da: float = 0.01):
    """Greedy one-to-one matching of sorted peak arrays within absolute tolerance."""
    a = np.asarray(mz_a, dtype=float)
    b = np.asarray(mz_b, dtype=float)
    i = j = 0
    pairs: list[tuple[int, int]] = []
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tolerance_da:
            best_j = j
            if j + 1 < len(b) and abs(a[i] - b[j + 1]) < abs(d) and abs(a[i] - b[j + 1]) <= tolerance_da:
                best_j = j + 1
            pairs.append((i, best_j))
            i += 1
            j = best_j + 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return pairs


def matched_cosine(mz_a, int_a, mz_b, int_b, *, tolerance_da: float = 0.01) -> float:
    ia = np.asarray(int_a, dtype=float)
    ib = np.asarray(int_b, dtype=float)
    na = np.linalg.norm(ia)
    nb = np.linalg.norm(ib)
    if na == 0 or nb == 0:
        return 0.0
    dot = sum(float(ia[i] * ib[j]) for i, j in match_peaks(mz_a, mz_b, tolerance_da=tolerance_da))
    return float(dot / (na * nb))
