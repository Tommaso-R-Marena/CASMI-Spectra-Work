# Frozen v5.3 MassBank MUI Lockbox Protocol

Status: **predeclared before any MassBank MUI ranking result is computed or inspected**.

## Frozen code and external data

- Frozen selector/model code base: `Tommaso-R-Marena/CASMI-Spectra-Work@0006c7f81c7dfac53da48690ca0fe627108b9767`.
- Experiment branch starts exactly from that commit: `v53-massbank-mui-lockbox`.
- External spectral source: `MassBank/MassBank-data@befc8a1e2f2aef899747797c081a5d80fab12fe7`, directory `MUI/`.
- Historical MassBank tag used only as a temporal/source-separation check: `2026.03`. The feasibility workflow must fail if `MUI/` is present in that tag.
- Frozen expert weights: `benchmark_results/casmi_consensus_frozen_model.json` from the frozen code base. No parameter, weight, threshold, preprocessing rule, or feature definition may be changed after seeing a MassBank MUI ranking result.

## Scientific question

Does a trajectory-aware expert, used only as a guarded reranker of the unprotected static tail, improve same-formula structure identification on a previously unused multi-energy experimental library while exactly preserving the static tie-aware Top-1 head?

This experiment is designed as a one-shot external-domain test. MassBank MUI is not used to select v5.3, its overlap threshold, protected prefix, overlap depth, head-survival slack, feature weights, or minimax tail aggregation.

## Query/reference separation

Only records satisfying all of the following are in scope:

- MS2
- positive ion mode
- precursor type `[M+H]+`
- molecular formula present
- InChIKey present
- collision energy present

A molecular structure is identified by the first 14 characters of its InChIKey (2D connectivity block), so stereochemical variants with identical 2D connectivity are not counted as distinct candidates.

For each structure:

- candidate references are fixed at **10, 20, and 40 eV**;
- the query spectrum is fixed at **30 eV**;
- if multiple records exist at an exact required energy, choose the lexicographically smallest accession deterministically;
- the 30 eV query record is never used as a candidate reference.

A structure can enter a candidate pool only if it has complete 10/20/40 eV references. A query is eligible only if its truth structure also has a 30 eV spectrum.

## Candidate pools

For each eligible query, the candidate pool is **every distinct in-scope MUI 2D structure with the same exact molecular formula that has complete 10/20/40 eV references**.

Primary cohort: all eligible queries with at least two same-formula candidate structures.

Prespecified descriptive difficulty strata:

- candidate count >= 2
- candidate count >= 5
- candidate count >= 10
- candidate count >= 20

No stratum will be selected after outcome inspection as the primary result. The >=2 cohort is primary; larger-pool strata are prespecified robustness/difficulty views.

## Label-free feasibility gate

Before any ranking score is computed, `scripts/audit_massbank_mui_lockbox.py` must report only source identity, energy/structure coverage, formula-pool sizes, dates/instruments, and the frozen design. It must not compute static, expert, v5.3, reciprocal-rank, Top-k, or any other ranking-performance result.

Proceed to ranking if the audit produces at least 30 ambiguous eligible queries and at least 10 formula blocks. If those minima are not met, MassBank MUI remains unscored for this protocol and a different untouched external domain must be chosen rather than relaxing the cohort after inspection.

## Spectrum preprocessing and scores

Use the same frozen CASMI pipeline semantics:

- remove peaks below 0.1% relative intensity;
- retain at most the 128 most intense peaks;
- apply precursor filtering as in the frozen helper;
- mass tolerance: 0.01 Da;
- for cosine, square-root transform intensities followed by L2 normalization;
- static score: maximum matched cosine over the candidate's 10/20/40 eV references;
- ordered-path score: the existing frozen ordered trajectory/path score evaluated on the same query and 10/20/40 references;
- expert features: `consensus_feature_matrix(cos10, cos20, cos40, ordered_path_score)` exactly as frozen;
- expert score: frozen feature vector dot frozen CASMI consensus weights.

## Frozen v5.3 selector

Apply protected-tail v5.3 with no retuning:

- `overlap_threshold = 0.8`
- `protected_k = 1`
- `overlap_k = 10`
- `head_survival_slack = 1`
- protected head is tie-closed
- tail aggregation is the frozen minimax rank rule

The implementation must assert candidate-order equivariance and exact tie-aware Top-1 preservation for every query.

## Outcomes

Primary ranking endpoint: paired delta in tie-aware **MRR@25**, v5.3 minus static, on the full >=2 cohort.

Additional prespecified endpoints:

- tie-aware Top-1
- tie-aware Top-5
- tie-aware Top-25
- median expected truth rank
- wins / losses / ties in reciprocal-rank credit
- selector usage fraction
- protected-head size distribution

Top-1 delta must be exactly zero as an implementation/theory invariant, not merely statistically nonsignificant.

## Dependence-aware inference

Queries sharing a molecular formula use the same candidate pool and are correlated. Therefore primary uncertainty is blocked by **molecular formula**, not by individual query.

Report:

- paired formula-block bootstrap confidence interval for MRR@25 delta;
- paired formula-block sign-flip/randomization test when applicable;
- number of independent formula blocks;
- the same descriptive endpoints in the prespecified >=5, >=10, and >=20 strata without using those strata to retune or select the method.

## Negative controls

Run all controls on exactly the same eligible queries and candidate pools:

1. **Within-query expert permutation:** randomly permute frozen expert scores across candidate identities using a fixed recorded seed.
2. **Expert inversion:** reverse expert preference while leaving static scores unchanged, then pass through the same protected-tail guard/reranker machinery.
3. **Trajectory-feature permutation:** permute the trajectory/path component across candidate identities before constructing the expert score, using a separate fixed seed.

Controls are diagnostic and must not be used to modify v5.3 after the lockbox is opened.

## Interpretation rules fixed before outcomes

Evidence for cross-domain generalization requires all of the following descriptive facts:

- exact Top-1 preservation;
- positive overall MRR@25 delta for frozen v5.3;
- the gain is not reproduced at comparable magnitude by the negative controls;
- no catastrophic protected-head displacement (should be impossible by construction);
- enough independent formula blocks for meaningful paired inference.

Stronger evidence is obtained if the formula-block confidence interval excludes zero and/or the paired randomization test supports a positive effect, and if the direction remains nonnegative in larger candidate-pool strata. Statistical weakness or a negative primary delta is recorded as a failed external lockbox; the MUI data may not then be used to retune v5.3 and relabeled as confirmation.

## Reproducibility guardrails

The workflow must record:

- exact user-repository commit;
- exact MassBank commit;
- a deterministic content hash over the MUI text-file tree;
- feasibility manifest;
- frozen-model file hash;
- ranking/control seeds;
- final candidate-feature artifact hash;
- full JSON results and per-query table.

Any future variant developed after this lockbox is opened must call MassBank MUI development/post-hoc data and seek a different external confirmation set.
