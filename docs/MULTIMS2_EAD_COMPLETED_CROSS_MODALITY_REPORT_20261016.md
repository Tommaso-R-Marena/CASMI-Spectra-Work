# CID–EAD conditional geometry reversal: completed experimental report

**Date: 2026-10-16 (US Eastern), workflows completed 2026-10-17 UTC.** This report is written AFTER observing results. It is NOT a prospective protocol and does not replace either frozen score or the failed initial feasibility gate.

## Outcomes and provenance

1. EAD V1 strict two-polarity HEADER-ONLY gate: [run 38985385615](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38985385615). Code succeeded but **science gate FAILED**: 695 eligible EAD triplets, 667 full InChIKeys, 594 formula blocks, **694 positive / 1 negative**, compared against predeclared >=5 negative. No target peak intensities opened. Retain failure.
2. V2 positive-only HEADER-ONLY gate: [run 38985767245](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38985767245). New protocol was frozen after knowing V1 metadata but before reading 16-eV targets. Entire 694 positive V1 cohort passed: 666 structures, 593 formulas. No outcome-conditioned exclusions.
3. V2 fully frozen scored experiment: [run 38985827065](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38985827065). The hybrid and all baselines were frozen at commit 4295a1ae7ead5f56a0d5b57a772cd651032d1362 before EAD target scores; source, reference grid and statistic unchanged.
4. After outcomes were known, exact-identity same-source paired comparison: [run 38986025076](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38986025076). This is purely RETROSPECTIVE, no spectra rescored or models fitted.

## EAD positive 12→16→24 eV: null hybrid and strong Hellinger advantage

| Method | Mean BC (n=694) |
|---|---:|
| Hellinger alpha=1/2 | **0.871049** |
| Support-only fourth-power exclusive | 0.853096 |
| New persistent-Hellinger/transient-power hybrid | 0.853092 |
| Quarter-power alpha=1/4 | 0.852993 |

- Hybrid minus quarter: +0.000099, 95% formula bootstrap CI [+0.000001,+0.000204], p=0.063097 before Holm.
- Hybrid minus support-only: -0.000004, CI [-0.000182,+0.000167], p=0.962352.
- Holm p=0.126194 and 0.962352. **The predeclared hybrid-beats-both gate FAILED.**
- Quarter minus fixed Hellinger: **-0.018056**, CI **[-0.019613,-0.016532]**, two-sided block-signflip p≈0.00005. Quarter won 96, lost 598. This confirms predictive regime heterogeneity on newly scored EAD targets, not universal quarter-power superiority.
- No-exact-prior-CID-structure sensitivity n=155, 145 formula blocks: quarter affinity 0.863205, Hellinger 0.879183; sign remains negative. This cohort is within-source, post-hoc descriptive sensitivity, not new independent validation.

## Same-identity CID vs EAD reversal (retrospective)

Matching full InChIKey + formula + adduct + polarity + source dataset (NO structure truncation), 463 paired compound-condition groups and 413 formula blocks:

| Same matched structures | Mean quarter minus Hellinger |
|---|---:|
| CID 20/40/60 eV, t=0.5 | **+0.073754** |
| EAD 12/16/24 eV, t=1/3 | **-0.018601** |
| CID minus EAD within identity | **+0.092356** |

Formula-cluster bootstrap 95% CI of within-identity difference: **[+0.087868,+0.096972]**; formula-block signflip p≈0.000020. The difference was positive for 453 of 463 pairs.

**Crucial confounding:** Mean endpoint BC among these SAME identity pairs was CID **0.189790** vs EAD **0.780956**. Moreover, activation mechanisms, energies, normalized interpolation coordinate and perhaps preprocessing differ. Pairing removes chemical identity/composition as a sufficient explanation; it **does not** establish fragmentation mode as causal, and is not independent-acquisition validation.

## Scientific claim supported, and next falsifiers

Evidence supports **strong acquisition-conditional heterogeneity** of nonlinear spectrum interpolation, including a reversal across fragmentation regimes on matching chemical identities. It **falsifies** any claim that one quarter-power exponent is universally optimal. It does NOT show the newly proposed hybrid is superior; the hybrid failed its declared primary gate. It does NOT improve CASMI molecular structure ranking, prove thermodynamic kinetics, or establish a universal predictive law.

High-value next test: select a **fourth independently acquired laboratory/instrument dataset** with matched CID/EAD (or analogous fragmentation channels), genuine measured collision-energy calibration, sufficient positive AND negative ion coverage where possible, and compound/adduct-matched held-out triplets. Freeze a modality- and endpoint-overlap-aware rule before that source's outcome peaks. Compare against fixed Hellinger, fixed quarter and support-only, with formula/source clustered uncertainty and actual same-formula structure-identification metrics. Do not tune or call confirmatory any threshold fitted on these EAD outcomes.

All original workflow artifacts remain preserved for reproducibility. No main branch modifications or unsupported breakthrough claim.
