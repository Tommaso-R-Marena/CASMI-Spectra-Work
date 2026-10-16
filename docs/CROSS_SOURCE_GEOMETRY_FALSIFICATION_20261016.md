# CASMI spectral interpolation: cross-source decisive falsification report (2026-10-16)

**Status:** all numerical results are from real completed GitHub Actions runs; provenance and artifact digests are retained in the individual branch workflows. This document is written **after** observing all outcomes and is not a prospective protocol.

## Original discovery and transfer: narrow success

- Frozen r=0.25 power mean selected from MCEBIO+ discovery and tested on four eligible MSnLib internal-energy transfer cohorts, combined **8,296 triplets**. The *target-blind* post-hoc reference-only-grid falsification subsequently found Δ(quarter−Hellinger) between **+0.02067 and +0.03616** by cohort, while a support-only baseline accounted for part of the improvement. This is not 8,296 independent compounds or instruments. [Original transfer run 38845952532](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38845952532), [target-blind follow-up 38847328313](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38847328313).
- Frozen MassBank 2026.03 external release **2,138 triplets**, 715 structures / 613 formula blocks: Δ quarter−Hellinger **+0.0061376**, 95% formula-block CI [+0.0050048,+0.0072962], p at 50,000-flip floor. However, answer-aware global Hellinger oracle was better than quarter (quarter−oracle −0.0032788). [Original scored run 38918658343](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38918658343).
- Post-release exact prior-evaluated-MSnLib structure exclusion left **2,032** MassBank triplets, 681 structures, 589 formula blocks: quarter−Hellinger **+0.006098**, CI [+0.004886,+0.007330]. This survives exact structure overlap removal but is a sensitivity, not an independent new dataset. [Run 38934839645](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38934839645).

## Cross-source independent-acquisition challenge: primary hypothesis falsified

The University of Vienna/Global Exposomics and Biomonitoring Laboratory **ExpoLib v1** source (Zenodo 18186811) was chosen separately from the previously analyzed MSnLib/MassBank sources. An immutable, metadata-only pre-score audit [run 38964190641](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38964190641) passed all four numerical gates with **1,125 triplets, 122 structures, 114 formulas**, after exclusion of all original MCEBIO+ discovery full-InChIKeys. Frozen quarter/Hellinger, support-only, and nondeployable global Hellinger oracle scored all records without refitting, rebinning, or dropping bad outcomes. [Scoring run 38964797817](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38964797817).

| ExpoLib contrast | Paired Δ mean affinity (r=.25−comparator) | Formula-cluster bootstrap 95% CI | Block sign-flip p |
|---|---:|---|---:|
| **Fixed r=.5 Hellinger** | **−0.0022910** | **[−0.0034248,−0.0011878]** | **0.000120** |
| Support-only, power=4 | +0.0014997 | [+0.0011897,+0.0018260] | 0.000020 |
| Answer-aware best Hellinger chord (nondeployable) | −0.0113267 | [−0.01399697,−0.00899173] | 0.000020 |

Mean affinity α=.25 **0.9354632** vs fixed Hellinger **0.9377543**. Quarter won 533, lost 590, tied 2 query-level comparisons. Positive-ion Δ −0.001440, negative-ion Δ −0.003219. **1022 nominal-midpoint energy samples** gave Δ **−0.002636**, CI [−0.003769,−0.001550]; 103 offcenter samples gave +0.001136, CI [−0.004059,+0.006007] (not a supported positive effect).

**Scientific interpretation:** A universal claim that r=.25 beats r=.5 on unseen spectral-energy transfer is **falsified** by the independently acquired ExpoLib source. The negative result was pre-scored under immutable parameters. Crucial limitation: ExpoLib MSP labels nominal `Collision_energy: [40.0]` without explicit calibrated energy units and lacks a fully established provenance independence audit. The experiments support predictive-coordinate conclusions, NOT a physical Amari-geodesic law.

## Independent-source overlap sensitivity also preserves negative result

An additional exact-full-InChIKey artifact-only audit [run 38965358716](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38965358716) found **8 ExpoLib structures** in the four previously evaluated MSnLib cohorts, **25** in the original MassBank eligible pool, **32 in their union** (one common to both), and **90** structure identities without exact overlap with either.

| ExpoLib cohort | Queries | Formulas | Quarter−Hellinger | Formula-cluster 95% CI | Two-sided block p |
|---|---:|---:|---:|---|---:|
| Unchanged full cohort | 1,125 | 114 | −0.002291 | [−0.003425,−0.001188] | 0.000120 |
| No exact prior-MSnLib-evaluated overlap | 1,070 | 108 | −0.002189 | [−0.003359,−0.001050] | 0.000400 |
| No exact prior-MassBank-eligible overlap | 869 | 91 | −0.002517 | [−0.003886,−0.001159] | 0.000380 |
| **No exact overlap with either** | **823** | **85** | **−0.002396** | **[−0.003813,−0.001040]** | **0.001260** |

These subsets are **post-hoc sensitivity tests** of the same source, not an additional independent sample. The negative finding is not explained solely by reusing previously evaluated chemical structures.

## Transfer to actual identification: no observed improvement

A separate frozen, metadata-only isomer challenge grouped same-formula, adduct, ionization mode, instrument, contributor, fragmentation method and target collision energy; all candidates had distinct InChIKey14 connectivity. The fully isolated held-out-target ranking test [run 38963742589](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38963742589) evaluated **252** targets from **44 formulas**, each choosing among 2–10 plausible structures.

- Quarter-power MRR **0.95833**, top-1 **92.06%**.
- Hellinger MRR **0.96032**, top-1 **92.46%**.
- MRR Δ quarter−Hellinger **−0.001984**, 95% formula CI **[−0.007843,+0.004785]**; **3 better, 4 worse, 245 ties**.
- Excluding entire candidate pools touching previously evaluated MSnLib structures preserved the null result on 230 targets.
- Posthoc secondary continuous correct-vs-best-decoy affinity margins favored quarter by approximately **+0.010226**, but did **not** change actual identification ranks materially. **Do not call this an identification improvement.**

This small exact-formula candidate challenge is not official CASMI 2026 candidate generation and suffers from a near-ceiling top-1 result on both methods.

## Research decision and next genuinely decisive experiment

**Retire** the universal-alpha-quarter superiority hypothesis as stated. Retain a narrower, testable fact: nonlinear support-sensitive interpolation can help on some acquisition regimes, but the correct geometry depends on source/experimental conditions, and performance improvements at the correct target's affinity need not affect molecular identification.

Next, independently sample instruments/laboratories and freeze:
1. A physics-consistent collision-energy calibration protocol, including precursor charge, collision gas and instrument activation metadata, with paired energy series measured on the same structures.
2. Mechanism hypotheses based **only on endpoint-available** covariates: shared fragment mass, endpoint support turnover, intensity transfer, precursor-normalized collision energy, and acquisition family. Compare prechosen predictors against a fixed Hellinger baseline and support-only control, using leave-one-laboratory-out evaluation; avoid optimizing on ExpoLib after this falsification.
3. Genuine higher-complexity candidate-ranking evaluation (≥25 same-formula/isobaric candidates where feasible, absent-library compounds, all structure/identity overlap exclusions) with MRR/top-1, source-stratified intervals and full null reporting.
4. Preserve negative results and report at the level of independent compound/formula/source rather than treating multiple energy triplets as independently acquired laboratories.

This is scientifically **substantive falsification**, not yet a new universally better predictive algorithm or physical law.
