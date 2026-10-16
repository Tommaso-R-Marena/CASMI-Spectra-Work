# Prospective third-source MultiMS2 CID 20/40/60-eV geometry transfer, v1

**Freeze date: 2026-10-16, BEFORE reading any MultiMS2 query-spectral intensities, model scores, or accuracy.** This source has not been evaluated in the CASMI geometry program. Prior MSnLib/MassBank gains and ExpoLib negative results are public prior observations motivating this experiment.

## Source pin and measurement regime

- Dataset: [MultiMS2](https://github.com/zamboni-lab/MultiMS2), Zamboni Lab / ETH Zürich, ZenoTOF 7600; published open-access repository **MIT code / CC0 spectral data**. Primary upstream manuscript: doi:10.1093/gigascience/giag069; Zenodo doi:10.5281/zenodo.17250693. README explicitly identifies CID at **20, 40, 60 eV**, in positive and negative ion modes, using the Sciex ZenoTOF instrument. Unlike ExpoLib, the *published experimental protocol* supplies these units, although the exported per-scan MGF metadata may provide only numeric CE; do not interpret eV as transferable collision-gas center-of-mass energy.
- Exactly pin upstream Git commit `659bd9b608408a75ecbb4b03295ed4c03ba52539` and file `data/multims2_spectra.mgf`, **Git blob SHA-1 `5ca1b01d1c6c5b2930658a6cd32ee2659070b49d`**, length **77,183,510 bytes**. Download through a commit-pinned raw URL; calculate and publish the full byte SHA-256 before scoring. Never replace it with a newer version even if a newer release appears.
- Mass spectra are from a *new independently acquired dataset* relative to prior cohorts, but exact chemical identities can overlap; therefore independently exclude all full InChIKeys in the initial MCEBIO-positive MSnLib discovery source (MD5 `806f33c9c60af74a0aa21fb66aeb84f3`), all four previously evaluated MSnLib cohorts, the original 2,138-triplet eligible MassBank cohort, and ExpoLib's 1,125 selected eligible triplets **before accessing any target intensities**. An overlap-free source remains one laboratory/instrument, not seven independent replications.

## Separate blinded stages

**Stage 1: metadata-only gate (no target peak positions/intensities read).** Stream the exact pinned MGF file, parsing only `KEY=VALUE` headers, never numeric peak intensity lines. Require a complete valid full InChIKey, molecular formula, ion mode, adduct, MS2, CID and an `exact` collision-energy field of **20, 40, or 60**; reject EAD, other energies, ambiguous units/CE strings, malformed/missing identity, and records with fewer than five *declared* peaks (not by peak values). Group by full InChIKey + formula + adduct + polarity + fragmentation family. For duplicate spectrum records at a given energy use the **first file-order entry** (source-position, independent of peaks); report every duplicate discarded. Require exactly one 20/40/60 triple after deterministic de-duplication. Report excluded compound overlap separately for discovery/evaluated-MSnLib/MassBank/ExpoLib. A structure is excluded if it occurs in **any** prior source. Feasibility thresholds: ≥100 query triples, ≥80 full InChIKeys, ≥60 formulas and ≥20 triples in **each** polarity after all exclusions. If gate fails, do **not** open peaks or score: publication of failed feasibility is the entire result.

**Stage 2 (only if gate passes): locked methods, same triplets.** Build the 0.01 Da reference-only peak grid from the 20 and 60 eV spectra, normalize positive reference intensities, match 40 eV query peaks only at scoring (one-to-one fixed nearest m/z, with unmatched intensity retained). Use physical coordinate **t=0.5** for all 20/40/60 eV triples. Fail closed on any query/endpoint identity mismatch, missing file, nonfinite affinity, or energy unexpected. No reduction of the Stage 1 denominator.

Frozen competing predictors:
1. **Quarter-power**: `P_0.25(p20,p60,t=.5)`, chosen on prior MSnLib discovery.
2. **Hellinger**: `P_0.5(p20,p60,t=.5)`.
3. **Untrained adaptive exponent**: set `s=Σ√(p20_i p60_i)` (endpoint-only Bhattacharyya overlap on reference grid); choose `r=0.25 + 0.25*s` in [0.25,0.5], and predict `P_r` at t=.5. This particular monotone geometric rule is *fixed now* before any MultiMS2 outcomes, motivated by prior source heterogeneity. It may lose to both fixed baselines. No model fitting, weights, thresholds or oracle labels allowed.
4. Support-only shared-fragment suppression, frozen power=4.
5. Linear mixture `P_1`, descriptive.
6. Answer-aware optimal Hellinger chord point — **nondeployable falsification oracle**, not a competing usable predictor. It must dominate the fixed Hellinger point on each query.

## Frozen outcomes and interpretation

- **Primary**: mean paired BC quarter vs fixed Hellinger, all eligible 40 eV queries; report n and all failures, 10,000 entire-*molecular-formula*-block bootstrap replicates, and two-sided 50,000 block sign flips, seed `20261016`. Positive evidence requires Δ>0, 95% lower bound>0, p<.05. Negative results require full reporting and explicitly falsify unqualified universal alpha-quarter superiority.
- **Secondary adaptive-exponent primary falsification:** adaptive minus fixed Hellinger and adaptive minus quarter, both on the same queries, with Holm multiplicity adjustment across two comparisons (only if evaluating their statistical claims). Report all three means and count of adaptive wins over each. If adaptive beats neither, report null/negative instead of revising rule. This adaptive rule is informed by previously observed sources; only MultiMS2 is prospectively unopened.
- Specificity: quarter minus support-only, quarter minus answer-aware Hellinger oracle. Also report positive vs negative polarity, source/collection identities if reliably encoded, and near-unmatched target spectral mass.
- No post-hoc gating to rescue the effect; no α/t/mz tolerance tuning, no CE warping, no target-derived predictor features. Group bootstrap by formula and disclose source-level limitations. The result is **third acquisition-source validation**, *not* proof of universal physical-law geometry or CASMI structural identification ranking success.

## Provenance ledger

Keep unchanged upstream file SHA, the entire metadata-only eligibility manifest, all source-key inventory hash digests, full query-level paired scores, methods/configuration hashes, and visible negative outcomes. If metadata format differs unexpectedly, stop as a failed v1 gate and publish exact stagewise attrition. Implement a separate versioned parser protocol before any retry; do not silently relax eligibility during scoring.
