from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator
import numpy as np

from .trajectory import SpectrumView


@dataclass(frozen=True)
class MSnLibRecord:
    inchikey14: str
    full_inchikey: str
    molecular_formula: str
    compound_name: str
    spectrum_id: str
    adduct: str
    collision_energies: tuple[float, ...]
    merge_type: str
    precursor_mz: float | None
    view: SpectrumView

    @property
    def is_merged_energy(self) -> bool:
        return len(self.collision_energies) > 1

    @property
    def is_single_energy(self) -> bool:
        return len(self.collision_energies) == 1


def _energy_tuple(value) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, list):
        return tuple(float(x) for x in value if x is not None)
    return ()


def record_from_json(obj: dict) -> MSnLibRecord | None:
    """Convert one released MSnLib JSONL object into the QFD neutral representation."""
    full_key = str(obj.get("inchikey") or "").strip()
    formula = str(obj.get("formula") or "").strip()
    adduct = str(obj.get("adduct") or "").strip()
    energies = _energy_tuple(obj.get("collision_energy"))
    peaks = obj.get("peaks") or []
    if not full_key or not formula or not energies or not peaks:
        return None
    mzs = []
    intensities = []
    for p in peaks:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        mz, intensity = float(p[0]), float(p[1])
        if np.isfinite(mz) and np.isfinite(intensity) and intensity >= 0:
            mzs.append(mz)
            intensities.append(intensity)
    if not mzs:
        return None
    sid = str(obj.get("usi") or obj.get("scan_number") or obj.get("feature_id") or "")
    precursor = obj.get("precursor_mz")
    precursor = None if precursor is None else float(precursor)
    view = SpectrumView.make(
        mzs, intensities, energies, adduct=adduct,
        base_peak_intensity=float(max(intensities)),
    )
    return MSnLibRecord(
        inchikey14=full_key[:14],
        full_inchikey=full_key,
        molecular_formula=formula,
        compound_name=str(obj.get("compound_name") or ""),
        spectrum_id=sid,
        adduct=adduct,
        collision_energies=energies,
        merge_type=str(obj.get("merge_type") or ""),
        precursor_mz=precursor,
        view=view,
    )


def iter_msnlib_jsonl(path: str | Path) -> Iterator[MSnLibRecord]:
    """Stream an MSnLib release file without loading it wholly into memory."""
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                continue
            rec = record_from_json(obj)
            if rec is not None:
                yield rec


def load_msnlib_jsonl(path: str | Path) -> list[MSnLibRecord]:
    return list(iter_msnlib_jsonl(path))
