# Target-blind collision-energy geometry falsification v1

**Protocol authored October 16, 2026, after reviewing the positive four-cohort alpha-quarter transfer result.**

## Status and non-confirmatory limitation

This is a **post-hoc robustness / mechanism falsification of a known positive result**, NOT an independent prospective confirmation, a unique geometry law, or a held-out exponent selection study. The four eligible datasets and earlier results were already observed when this protocol was designed. The unchanged alpha-quarter predictor, original feasibility gate, and all eight prespecified cohort names remain fixed.

## Primary threat: target-dependent m/z grid

The previous `_common_grid_strict([query, low, high], .01)` used **the withheld target's m/z values when constructing the spectral prediction basis**. Even though target intensities were not used in the mixing coefficients, target peak locations could affect the reference bin assignments and merging; strict diameter bounding fixes transitive chaining but does not fix this target influence.

The new implementation:

1. Builds a 0.01 Da **complete-link** m/z grid solely from low/high reference peaks (and sums all positive peaks within each reference bin).
2. Normalizes each reference spectrum independently by its full positive mass.
3. Computes each predictor on the reference-only grid, **before reading the query**.
4. For scoring, matches observed query peaks to reference-grid centers within <=0.01 Da by deterministic smallest-error one-to-one matching. Query intensity is normalized by **all** positive observed peaks. Unmatched query mass scores zero rather than disappearing or being renormalized.
5. Checks permutation and target independence, one-to-one matching, reference mass conservation, handling of absent/shared peaks, and oracle correctness by unit tests.

This replaces—not silently patches—the previous evaluation metric. Absolute affinities across old and new alignment are not directly comparable. Old and new **within-metric paired deltas** are reported explicitly and independently.

## Frozen model comparisons (no tuning)

- ordinary probability-linear path r=1
- Hellinger/square-root chord r=1/2
- power path r=1/4 **unchanged** from MCEBIO-positive discovery
- support-only suppression, simple joint/endpoint-specific rule: on bins present at both energies use ordinary probability mixture; on bins unique to low/high use (1-t)^4 low or t^4 high, then normalize
- answer-aware convex probability-mixture oracle (nondeployable upper bound on *convex mixing only*)
- best measured endpoint
- r=1/8 **exploratory boundary sensitivity**, not preselected and not eligible for a prospective superiority claim

All scores use the same target-blind binning and one-to-one matching.

## Data, gate, and inference

Use the exact eight pinned 2024-10-03 MSnLib positive/negative files with recorded MD5 and the original label-free LOEO gate: >=1,000 query triplets, >=500 structure/adduct groups, >=300 formula blocks, >=500 off-center triplets. Do not compute outcomes for any gate failure; preserve only feasibility/provenance.

For each eligible dataset, save per-query paired outcomes, query unmatched-mass fraction, source-overlap statistics (computed from reference spectra only), and hashes. Produce formula-block bootstrap 95% CIs (3,000 replicates) and two-sided block sign-flip p-values (50,000 draws). The **first robustness contrast** is r=1/4 minus Hellinger under target-blind alignment. The primary geometry-specific falsification contrast is r=1/4 minus the **support-only** control. Report r=1/4 minus oracle-mixture separately. For comparisons across multiple eligible datasets, correct for all eight cohort tests or explicitly label descriptive. Do not pool formulas as independent across libraries. The new results are post-hoc by construction; p-values quantify conditional evidence, not independent prospective discovery.

## Scientific decision conditions

- If the r=1/4 Hellinger advantage disappears with target-blind alignment, the existing geometry claim must be downgraded materially until replicated independently.
- If support-only suppression matches or beats r=1/4, the *distinctive curved power path* is not established; shared/exclusive-peak heuristics may suffice.
- If target-blind r=1/4 beats both in all eligible cohorts, this is a strong robustness result, **not yet external or physics-general evidence**.
- Large unmatched query mass is a warning that m/z reference-grid construction/measurement differences dominate the comparison. Report it regardless of sign.
- A claim of cross-instrument physical law requires a new source and audit of collision-energy units, instrument acquisition, exact-structure overlap, fragmentation preprocessing, and source duplicate leakage.

## Link to prior data

Earlier frozen transfer and run: `docs/ENERGY_GEOMETRY_POWER_TRANSFER_REPORT.md`; Actions run 38845952532. **Do not modify these files or conflate old and new score semantics.**
