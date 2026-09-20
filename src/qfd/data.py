from __future__ import annotations

from pathlib import Path

from .casmi_schema import TRAIN_COLUMNS


def _polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("CASMI parquet support requires the optional 'data' dependencies (polars, pyarrow)") from exc
    return pl


def scan_train(path: str | Path, columns=None, *, ingest_lib: str | None = None):
    """Memory-safe lazy scan of train.parquet."""
    pl = _polars()
    selected = list(columns or TRAIN_COLUMNS)
    lf = pl.scan_parquet(str(path)).select(selected)
    if ingest_lib is not None:
        if "ingest_lib" not in selected:
            raise ValueError("ingest_lib must be selected when filtering by ingest_lib")
        lf = lf.filter(pl.col("ingest_lib") == ingest_lib)
    return lf


def metadata_audit(path: str | Path) -> dict:
    """Compute a compact audit without collecting peak arrays into memory."""
    pl = _polars()
    cols = [
        "spectrum_id", "inchikey14", "molecular_formula", "ingest_lib",
        "instrument_type", "adduct", "collision_energy_ev", "num_peaks",
    ]
    lf = pl.scan_parquet(str(path)).select(cols)
    summary = lf.select(
        pl.len().alias("rows"),
        pl.col("inchikey14").n_unique().alias("unique_structures"),
        pl.col("molecular_formula").n_unique().alias("unique_formulas"),
        pl.col("ingest_lib").n_unique().alias("libraries"),
    ).collect().row(0, named=True)
    return {k: int(v) for k, v in summary.items()}
