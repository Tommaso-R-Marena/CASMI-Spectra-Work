# CASMI QFD — Unified immutable falsification ledger (2026-10-17)

This document separates **theorem/correctness**, **empirical spectral
accuracy**, and **actual chemical-structure identification**, so that
a mathematically improved score is never mistaken for successful
molecule identification. All values below were observed on real data
and are anchored to the specific completed GitHub Actions run.

## Previously completed independent/retrospective experiments

| Program | Cohort | Fixed quantitative result | Scientific verdict | Evidence |
|---|---|---|---|---|
| Matched CID/EAD | 463 exact identity/acquisition matches | CID quarter−Hellinger +0.073754, EAD −0.018601; matched delta +0.092356 | Acquisition-conditioned heterogeneity; noncausal retrospective analysis | [38986025076](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38986025076) |
| Positive-only EAD hybrid | 694 query spectra | Hellinger 0.871049 vs quarter 0.852993; hybrid 0.853092; tiny hybrid-quarter change not Holm-significant | Hybrid failed, all-polarity EAD feasibility failed | [38985827065](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38985827065) |
| PhenolicsDB external Agilent CID | 50 target spectra, 15 unique structures | Quarter−Hellinger +0.030097, molecule-block CI [+0.022155,+0.038721] | Independent-origin CID interpolation improvement; narrow chemistry, NOT independent identification | [39048117682](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39048117682) |
| PhenolicsDB support ablation | Same 50, 15 structure blocks | Quarter−support-only −0.001922 at 0.01 Da; negative at all four tolerances | Shared-fragment quarter curvature not uniquely superior; retrospective | [39048910201](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39048910201) |
| MultiMS2 CID/EAD actual exact-formula isomers | 63 queries, 29 groups | CID Hellinger MRR 0.906878, fused Hellinger MRR 0.901587 | Equal-weight fusion failed to beat CID | [39047245864](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39047245864) |
| MultiMS2 rank certificates | Same 63 | Support MRR 0.914815 vs Hellinger 0.906878; posthoc formula-block signflip p=1.0 | Small observed gain NOT statistically established; mathematically valid stability certificate | [39049646354](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39049646354) |
| SPLISUM new-source intake | Four genuine MGF sources | Source checksums pass but strict native metadata parsing/chemical label gate fails | No target peaks scored; no identification claim | [39054508690](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39054508690) |
| GNPS→MassSpecGym original two-polarity metadata | 443,905 native annotated scans | 6,237 cross-archive native identity overlaps; 4,398 positive-ion hard queries; 0 negative-ion hard queries | Two-polarity gate scientifically failed, despite green CI | [39055589643](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39055589643) |
| GNPS→MassSpecGym separately frozen positive-only V2 | 4,398 full native keys, 2,254 formulas | No native chemical targets opened prior to V2 freeze | Positive-only feasibility gate passed, original two-polarity failure PRESERVED | [39055988705](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39055988705) |
| GNPS→MassSpecGym firstlook structure retrieval | All 4,398 native same-formula hard queries | Hellinger MRR 0.811962, cosine 0.808541, composite 0.807687 | Composite-Hellinger ΔMRR −0.004275, prespecified Holm p=0.027319, negative | [39056094950](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39056094950) |
| Conservative 2,254-formula audit | Same 4,398 frozen queries | Composite−Hellinger formula-cluster CI [−0.007768,−0.000791], exploratory p=0.015780 | Original negative persists after dependent candidate-pool cluster correction | [39056486479](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39056486479) |
| Exact globally optimal peak assignment | Same 4,398, original 0.01 Da windows | Exact Hellinger MRR 0.811409 vs greedy 0.811962; formula-block CI [−0.001407,+0.000276], Holm p=0.352473 | Mathematical per-pair affinity improvement DOES NOT improve molecular ranking | [39057502565](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39057502565) |
| Graph-conditioned native GNPS candidate fragments | Same 4,398 original hard queries, 2,254 native formulas | Fixed 90% Hellinger +10% graph MRR 0.813106 vs Hellinger 0.811962; formula-block CI [+0.000123,+0.002258], retrospective p=0.032019; fractional top1 0.707360 vs 0.707398 | Small posthoc MRR positive signal but NO top-1 gain; multiple earlier ideas tested, NO independent confirmation | [39058512753](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39058512753) |

## General exact mathematical limitations

1. **Power-geometry support factorization:** normalized power
   interpolation `P_alpha(t)` has shared-bin curvature exponent alpha
   and exclusive-support survival exponent `1/alpha`; matching
   exclusive terms across hybrid alternatives isolates shared curvature.
   Code regression establishes identities for finite input arrays; this
   is not a Lean kernel theorem or an experimental physical law.
2. **Disjoint-support energy-gauge nonidentifiability:** for disjoint
   endpoint support, `P_alpha(t)=P_beta(sigmoid((beta/alpha)*logit(t)))`.
   Without independently calibrated energy coordinate or shared-support
   information, the exponent is not uniquely identifiable from this
   spectral path.
3. **Correctness of global assignment:** a globally maximum-weight
   bipartite matching gives affinity >= any feasible one-to-one greedy
   matching for *each candidate*, but relative ranks can shift either
   way since all rival candidates may improve too.
4. **Reference-only spectral rank stability:** if the baseline winning
   candidate beats each competitor by more than the SUM of their
   verified candidate-specific per-score perturbation bounds, the same
   unique top1 is guaranteed under the changed model. Otherwise the
   status is UNKNOWN, not disproved.

## Current experimental continuation — graph-conditioned chemistry

**Status:** full original retrospective benchmark completed and fixed. The
exact model for any genuinely independent replication is anchored at commit
`64aced60540b5d57e191ff0ead73b445c71e1732`; source outcome run
[39058512753](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39058512753).
The nominal positive MRR result cannot be interpreted as a prospective
replication or a top-1 identification gain; see
`docs/PROSPECTIVE_EXTERNAL_LAB_BREAKTHROUGH_VALIDATION_CONTRACT_20261017.md`.


Versioned study in `docs/GRAPH_CONDITIONED_FRAGMENT_RANKING_PROTOCOL_20261017.md`.

- Fixed recipe: 90% historical Hellinger + 10% RDKit graph-cut
  fragment plausibility weighted by rarity among every original
  candidate molecule in the native same-formula pool.
- Native candidate GNPS SMILES + formula only. Never use query SMILES,
  true chemical identity, or target outcome to generate signatures.
- Exact replay of all 4,398 historical Hellinger ranks before claiming
  any new result.
- **This study is RETROSPECTIVE:** the original source has already been
  examined. A positive result would generate a new external-laboratory
  hypothesis, NOT a confirmed breakthrough.

## Non-negotiable requirements for a genuine identification breakthrough

Require a **new independently acquired** source whose target chemical
identities, structure graph, instrument/activation/energy data and
candidate pools are locked BEFORE novel score evaluation. The primary
claim must be actual same-formula structure top-1 and/or MRR improvement
against Hellinger plus a strong contemporary molecule-ranking baseline,
with conservative structure/formula-block uncertainty and an explicit
source-lab holdout. Duplicate physical spectra, source/harmonization
overlap and native full-InChIKey inconsistencies must fail closed.

No result may be promoted simply because a CI workflow passed, a new
mathematical inequality is proven, or affinity improves without
identification. Preserve every falsification, including EAD hybrid failure,
cross-source geometry failures, two-polarity feasibility failures and
the strong composite regression.
