# MassBank 2026.03 post-release robustness and structure nonoverlap

**Classification: POST-HOC EXPLORATORY SENSITIVITY — not a new held-out independent validation.**

Original frozen MassBank efficacy: [run 38918658343](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38918658343), 2,138 scored triplets / 715 structures / 613 molecular formulas. Independent metadata-only overlap: [run 38920155068](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38920155068), 34 overlapping previously evaluated structures and 106 associated MassBank triplets.

## Immutable provenance and safeguards

1. Download the exact two completed-run artifacts. SHA-256-check `rows.jsonl`, `result.json`, `source_overlap.json` and both original query manifests against pinned values. Require the same 2,138 query accessions, full identity fields, unchanged original method means, all affinity values finite within [0,1], and valid fixed-Hellinger oracle dominance.
2. Obtain the exact four previously **evaluated** MSnLib source libraries with the archived MD5 digests: MCEBIO-negative, MCESCAF-positive, ENAMMOL-positive and ENAMDISC-negative. Make a sorted union of full InChIKeys (not just first-block keys). Check the resulting MassBank overlap is exactly **34 structures / 106 query triplets**, matching the independent pre-existing label-free identity audit; fail rather than guess if mismatched. Original MCEBIO-positive discovery structures were already excluded by the frozen eligibility protocol.
3. Report *without changing the original 2,138-row primary denominator*: the 2,032-query no-identified-prior-evaluated-MSnLib-exact-structure-overlap sensitivity, the 106-query overlap complement, and the non-Orbitrap/nonoverlap intersection.
4. Report each of the four frozen comparisons against fixed Hellinger, support-only shared-fragment suppression, the nondeployable answer-aware globally optimal Hellinger chord oracle, and the answer-aware convex mixture oracle. No exponent, intensity normalization, m/z matching tolerance, or geometry is re-fitted.
5. Provide query-weighted and formula-equal-weight means, wins/losses/ties, and formula-cluster bootstrap intervals for major subsets, with seeded full-formula resampling. Retrospective bootstrap CIs are **descriptive**, not a new confirmatory test or multiplicity-controlled trial.
6. Report every instrument/contributor stratum, ion polarity, off-center vs midpoint energies, leave-one-contributor-out results, and contributor-equal-weight mean. Do not hide negative or small strata.

## Interpretation boundary

The originally frozen positive result is preserved intact. A nonoverlap sensitivity still does not rule out independently sourced copies of the same acquisition, shared laboratories, near-identical scaffolds, or measurement-protocol dependence. Instrument and contributor are partially confounded. Only a **prospectively registered, separately acquired source** with untouched scoring and sufficient instrument-level replication can establish broad generality.

## Next scientifically decisive experiments

- Acquire instrument/vendor-specific raw spectra with verified eV units or physically calibrated collision-energy metadata, separate from MassBank and MSnLib redistribution. Register coverage/leakage/energy-unit checks *before* opening outcomes.
- Freeze **all** candidate predictors and controls, especially scalar-energy calibration and support-only heuristics, from the *existing discovery corpus only*. Do not optimize an exponent against this MassBank result.
- Compare fixed α=0.25 with Hellinger, support-only, prospective energy-warp controls, and an instrument-family-aware model evaluated on **held-out instruments and compounds**, not new rows from the same source.
- Report instrument-level replication, chemical-structure/scaffold cluster intervals, failures and effect sizes. Demonstrate a downstream advantage on structure-identification/ranking before claiming an improvement to CASMI identification.
