from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np

SHARED_COLUMNS = (
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
    "precursor_mz", "collision_energy_ev", "collision_energy_orig",
    "collision_energy_orig_units",
)
TRAIN_ONLY_COLUMNS = (
    "normalized_smiles", "inchikey", "inchikey14", "molecular_formula",
    "ingest_lib", "adduct_orig", "precursor_error_ppm", "num_peaks",
)
TRAIN_COLUMNS = SHARED_COLUMNS + TRAIN_ONLY_COLUMNS


def normalize_energy_list(value) -> tuple[float, ...]:
    """Return collision energy as a validated tuple of finite floats."""
    if value is None:
        return ()
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ()
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value]
    out: list[float] = []
    for x in values:
        if x is None:
            continue
        y = float(x)
        if math.isfinite(y):
            out.append(y)
    return tuple(out)


def is_merged_energy(value) -> bool:
    return len(normalize_energy_list(value)) > 1


def reconstruct_raw_intensities(normalized, base_peak_intensity):
    """Undo max-normalization when the raw base-peak count is available."""
    x = np.asarray(normalized, dtype=float)
    if base_peak_intensity is None:
        return None
    b = float(base_peak_intensity)
    if not math.isfinite(b) or b < 0:
        return None
    return x * b


def validate_peak_arrays(mzs, intensities) -> None:
    mz = np.asarray(mzs, dtype=float)
    it = np.asarray(intensities, dtype=float)
    if mz.ndim != 1 or it.ndim != 1:
        raise ValueError("peak arrays must be one-dimensional")
    if len(mz) != len(it):
        raise ValueError("m/z and intensity arrays must have equal length")
    if np.any(~np.isfinite(mz)) or np.any(~np.isfinite(it)):
        raise ValueError("peak arrays must be finite")
    if np.any(it < 0):
        raise ValueError("intensities must be nonnegative")
