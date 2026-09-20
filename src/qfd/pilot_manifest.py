from __future__ import annotations

from pathlib import Path


def _polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("pilot manifest support requires polars") from exc
    return pl


def build_pilot_manifest(
    train_parquet: str | Path,
    *,
    ingest_lib: str = "enveda-180",
    min_spectra: int = 3,
    min_energies: int = 2,
    max_structures: int = 5000,
    min_formula_structures: int = 2,
):
    """Select a deterministic, instrument-consistent multi-energy pilot cohort."""
    pl = _polars()
    lf = (
        pl.scan_parquet(str(train_parquet))
        .select([
            "inchikey14", "molecular_formula", "ingest_lib", "instrument_type",
            "adduct", "ionization_mode", "collision_energy_ev", "spectrum_id",
        ])
        .filter(pl.col("ingest_lib") == ingest_lib)
        .filter(pl.col("inchikey14").is_not_null())
        .filter(pl.col("collision_energy_ev").is_not_null())
        .filter(pl.col("collision_energy_ev").list.len() == 1)
        .with_columns(pl.col("collision_energy_ev").list.first().cast(pl.Float64).alias("energy_ev"))
    )
    groups = (
        lf.group_by(["inchikey14", "molecular_formula", "adduct", "ionization_mode"])
        .agg([
            pl.len().alias("n_spectra"),
            pl.col("energy_ev").n_unique().alias("n_energies"),
            pl.col("energy_ev").min().alias("energy_min_ev"),
            pl.col("energy_ev").max().alias("energy_max_ev"),
        ])
        .filter((pl.col("n_spectra") >= min_spectra) & (pl.col("n_energies") >= min_energies))
        .with_columns(pl.col("inchikey14").n_unique().over("molecular_formula").alias("formula_structure_count"))
        .filter(pl.col("formula_structure_count") >= min_formula_structures)
        .sort(["molecular_formula", "inchikey14", "adduct", "ionization_mode"])
        .head(max_structures)
    )
    return groups.collect()


def write_pilot_manifest(train_parquet: str | Path, out_path: str | Path, **kwargs):
    manifest = build_pilot_manifest(train_parquet, **kwargs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        manifest.write_csv(out)
    else:
        manifest.write_parquet(out)
    return manifest


def build_replicate_pilot_manifest(
    train_parquet: str | Path,
    *,
    ingest_lib: str = "enveda-180",
    min_replicates_per_energy: int = 2,
    min_energies: int = 2,
    max_structures: int = 5000,
    min_formula_structures: int = 2,
):
    """Select same-formula structures with replicated spectra at multiple energies."""
    pl = _polars()
    lf = (
        pl.scan_parquet(str(train_parquet))
        .select([
            "inchikey14", "molecular_formula", "ingest_lib", "adduct",
            "ionization_mode", "collision_energy_ev", "spectrum_id",
        ])
        .filter(pl.col("ingest_lib") == ingest_lib)
        .filter(pl.col("inchikey14").is_not_null())
        .filter(pl.col("molecular_formula").is_not_null())
        .filter(pl.col("collision_energy_ev").is_not_null())
        .filter(pl.col("collision_energy_ev").list.len() == 1)
        .with_columns(pl.col("collision_energy_ev").list.first().cast(pl.Float64).alias("energy_ev"))
    )
    replicated = (
        lf.group_by(["inchikey14", "molecular_formula", "adduct", "ionization_mode", "energy_ev"])
        .agg(pl.len().alias("replicate_count"))
        .filter(pl.col("replicate_count") >= min_replicates_per_energy)
    )
    groups = (
        replicated.group_by(["inchikey14", "molecular_formula", "adduct", "ionization_mode"])
        .agg([
            pl.len().alias("n_replicated_energies"),
            pl.col("replicate_count").sum().alias("n_replicated_spectra"),
            pl.col("energy_ev").min().alias("energy_min_ev"),
            pl.col("energy_ev").max().alias("energy_max_ev"),
        ])
        .filter(pl.col("n_replicated_energies") >= min_energies)
        .with_columns(pl.col("inchikey14").n_unique().over("molecular_formula").alias("formula_structure_count"))
        .filter(pl.col("formula_structure_count") >= min_formula_structures)
        .sort(["molecular_formula", "inchikey14", "adduct", "ionization_mode"])
        .head(max_structures)
    )
    return groups.collect()
