# CASMI benchmark protocol

This project treats the visible Kaggle `test.parquet` as a submission-format/inference-path check only, never as a validation set. Kaggle documents that the visible test file is comprised of examples from training and is replaced by the hidden test during re-run.

## Primary evaluation target

- Metric: MRR@25 on `inchikey14` / canonical 2D connectivity.
- Unit of evaluation: molecule, not individual spectrum.
- Candidate sets must be identical across static and dynamic ablations.

## Track A: Class-1-style library retrieval

Question: when reference spectra for the true structure exist, does preserving collision-energy/adduct trajectories improve ranking over best single-spectrum similarity?

Rules:

1. No query spectrum may be compared with itself or an exact duplicate row.
2. Query and reference spectra are partitioned deterministically by `spectrum_id`.
3. Static-best and joint-energy methods receive the exact same candidate structures.
4. Report all molecules and a predeclared hard subset: same-formula candidates with high static similarity.
5. Merged-energy spectra are excluded from ordinary energy matching and scored separately with cone/convex-hull methods.

## Track B: structure-held-out generalization

Question: can a model predict enough of the energy-conditioned trajectory to rank a structure whose `inchikey14` had no spectra in model training?

Rules:

1. Split by `inchikey14`; assert the train/validation intersection is empty.
2. Fit all preprocessing, calibration, and learned parameters on training structures only.
3. Keep candidate generation separate from candidate reranking.
4. Report same-formula and high-static-similarity subsets separately.

## Initial source priority

1. `enveda-180`: timsTOF family and eV collision energies, closest to hidden-test instrumentation.
2. `enveda-np-examples`: timsTOF examples, useful for instrument-matched checks but small.
3. `pluskal_ms2` / MSnLib-derived data: consistent multi-energy protocol, useful for trajectory experiments; instrument transfer must be reported explicitly.

## Preprocessing grid

Predeclare and ablate rather than silently tune:

- intensity floor: 0, 0.001, 0.01
- transform: none, sqrt, log1p
- max peaks: 64, 128
- m/z tolerance: begin at 0.01 Da and separately test ppm-aware matching
- drop peaks above precursor + 2 Da by default to mirror hidden-test cleaning

## Statistical reporting

For every headline comparison save:

- molecule count and spectrum count
- candidate-set construction
- seed
- MRR@25, top-1, top-5, top-10
- paired per-molecule reciprocal-rank difference
- bootstrap 95% confidence interval
- wins/losses/ties versus the static baseline

A dynamic method is not considered successful because a synthetic theorem works. It must improve leakage-controlled real-data rankings, especially on hard structural alternatives.
