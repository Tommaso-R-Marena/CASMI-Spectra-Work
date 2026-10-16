# Frozen MassBank 2026.03 external spectral-energy prediction result

**Completed 2026-10-16 (UTC).** [Scoring workflow 38918658343](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38918658343), successful. [Prior metadata-only audit 38917760935](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38917760935) already passed all six original thresholds without opening spectral results. **This file was written after the scored outcomes became visible and is not a prospective protocol.** Immutable protocols: `docs/MASSBANK_EXTERNAL_ENERGY_GEOMETRY_GATE.md` and `docs/MASSBANK_EXTERNAL_ENERGY_GEOMETRY_SCORING_PROTOCOL.md`.

## Locked provenance and eligibility

- Source: official MassBank-data 2026.03, DOI `10.5281/zenodo.19590880`, MD5 `1de9018361f7ffc689aa327f92e97e0f`, SHA256 `12c6a922929f9bf2484c46586b6101512a415e44680558883b0630c6e47905fb`.
- Full-InChIKey discovery exclusion from MSnLib MCEBIO+ dataset MD5 `806f33c9c60af74a0aa21fb66aeb84f3`, SHA256 `c481de4ad0926891a5a6eeb59d9c74523cc501d45bb2dcc88cb0e14cd027bc17`.
- Frozen corrected metadata parser commit `f027125817542bb24c29192290cda15492c3d3b5`; original prior independent scoring implementation/protocol `e440a0e2ed5e9b79b24986e7da0a49c6b60079e1`. The workflow verified byte-for-byte equivalence of frozen source files.
- Recomputed exactly matching original feasibility coverage before scoring: **2,138** query triplets, **715** structures, **613** formula blocks, **259** off-center triplets, **1,853** non-Orbitrap triplets, **647** non-Orbitrap structures. Every one of the six original gate thresholds passed. No model fitting, exponent search, m/z tolerance retuning or cohort reduction.
- Scoring uses 0.01 Da reference-only grid and one-to-one query peak matching, original raw intensities normalized over all positive query peaks, fixed `r=.25`, frozen fixed Hellinger `r=.5`, support-only, answer-aware Hellinger oracle, answer-aware convex mixture oracle. Bootstrap: 3,000 molecular-formula-block draws; signflip: 50,000 formula-block flips.

## Primary prospective contrast: quarter power vs fixed Hellinger

| Evaluation population | n | Formula blocks | Mean paired affinity gain | 95% formula-block bootstrap CI | Two-sided formula-block sign-flip p | Wins/Losses/Ties |
|---|---:|---:|---:|---|---:|---|
| All eligible MassBank | 2,138 | 613 | **+0.0061376145** | **[+0.0050047865,+0.0072962054]** | **0.0000199996** | 1,360 / 766 / 12 |
| Non-Orbitrap | 1,853 | 568 | **+0.0071605254** | **[+0.0059024235,+0.0084632990]** | **0.0000199996** | 1,206 / 635 / 12 |

**Verdict: PASS** for the predeclared global external primary gate (positive mean; bootstrap lower >0; two-sided p<0.05), with concordant prespecified non-Orbitrap stress. The p-value shown is the minimum obtainable with 50,000 Monte Carlo sign flips plus the add-one correction, not an exact p-value. Mean affinity on all samples: quarter-power **0.93095047**, fixed Hellinger **0.92481286**. Mean unmatched query intensity fraction **0.03916152**.

## Prespecified specificity controls: mixed result

| Comparison (all 2,138 triplets) | Mean delta (quarter minus comparator) | 95% formula-block CI | p |
|---|---:|---|---:|
| Support-only shared-fragment suppression | **+0.00828014** | [+0.00747560,+0.00919394] | 0.0000199996 |
| Answer-aware convex probability mixture | **+0.00852734** | [+0.00590590,+0.01110887] | 0.0000199996 |
| **Answer-aware globally optimal Hellinger chord point** | **−0.00327880** | [−0.00531585,−0.00139834] | 0.00091998 |

The nondeployable Hellinger scalar oracle **significantly beats** quarter power in the global MassBank population. This **falsifies** any statement that quarter power lies closer to the observation than every Hellinger-path reparameterization across external sources, even though it beats the fixed physical-energy Hellinger interpolation and the support-only heuristic. On the non-Orbitrap subset the oracle contrast is **−0.00158441**, CI [−0.00347171,+0.00022916], p=0.09672 (not significant). Earlier positive MSnLib global-oracle results therefore do **not** generalize uniformly. Do not present the source-specific result as a universal power-geodesic law.

## Contributor/instrument heterogeneity (descriptive only)

| Source / instrument type | n | Structures | Quarter minus fixed Hellinger mean |
|---|---:|---:|---:|
| Antwerp_Univ / LC-ESI-QTOF | 618 | 169 | +0.00324258 |
| Athens_Univ / GC-APCI-QTOF | 70 | 32 | +0.00108699 |
| Athens_Univ / LC-ESI-QTOF | 784 | 330 | +0.01085543 |
| Fiocruz / LC-ESI-QTOF | 281 | 96 | +0.00919684 |
| HBM4EU / ESI-QTOF | 57 | 8 | +0.00078170 |
| ISAS_Dortmund / LC-ESI-QTOF | 80 | 52 | +0.00557299 |
| MPI_for_Chemical_Ecology / LC-ESI-ITFT | 234 | 47 | **−0.00162491** |
| MSSJ / LC-ESI-QQ | 14 | 3 | **−0.01163712** |

There are **six positive** and **two negative** instrument/contributor cohorts. Small-source subsets can be noisy; no multiplicity-controlled claims of source-wise effectiveness were predeclared. Differences between datasets can be driven by instruments, compounds, annotation and energy conventions.

## Scientific claim boundary and next falsification

**Established under the frozen score and population:** a reproducible mean advantage for preselected quarter-power vs fixed Hellinger interpolation in this MassBank archive, including its non-Orbitrap subset; also beats support-only and best convex probability mixing on the all-sample mean.

**Not established:** physical Amari geodesic uniqueness; universal optimality of `r=.25`; superiority to an answer-aware Hellinger scalar oracle; no acquisition/provenance overlap with earlier MSnLib; improved molecular structure identification; thermodynamic or biochemical fragmentation mechanisms.

Next: perform *outcome-blind* source provenance and duplicate-spectrum checks using full-InChIKey overlap against all four previously reused MSnLib cohorts (not only MCEBIO+ discovery), same-spectrum duplicate fingerprints across archive records/contributors, and exact-origin audit for the dominant Athens and Antwerp source cohorts. These checks are to **annotate/report** the original frozen result, not to delete unfavorable molecules or retroactively change the primary denominator. Any post-hoc de-duplicated sensitivity should be labeled exploratory.

Artifacts: [original scoring run 38918658343](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38918658343) has original metadata feasibility, `result.json`, per-query `rows.jsonl`, `provenance.json`, and input SHA256 text.
