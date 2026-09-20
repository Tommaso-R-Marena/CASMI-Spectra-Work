# Build status

## Implemented

- Official CASMI shared/train schema constants and collision-energy list handling.
- Raw-intensity reconstruction from normalized peak intensities and `base_peak_intensity`.
- Exact MRR@25 implementation and ranking helper.
- Deterministic `inchikey14` structure holdout with explicit leakage assertion.
- Sparse spectrum preprocessing, peak matching, and matched cosine baseline.
- Collision-energy/adduct-preserving joint trajectory score.
- Static-best versus joint-energy candidate-library ranker.
- Lazy Polars Parquet scanning and compact metadata audit.
- CLI for memory-safe `train.parquet` audit.
- Regression tests for competition-specific edge cases.

## Not yet claimed

- No real CASMI validation result has been produced by this repository yet.
- No leaderboard score is used as scientific validation.
- No de novo generation model is implemented yet.

## Immediate next experiments

1. Run the metadata audit on the official `train.parquet`.
2. Build a compact `enveda-180` multi-energy pilot manifest.
3. Establish static-best MRR@25.
4. Compare joint-energy MRR@25 on identical candidates.
5. Construct same-formula/high-static-similarity hard negatives.
6. Add quotient/FECI and merged-energy cone ablations.
