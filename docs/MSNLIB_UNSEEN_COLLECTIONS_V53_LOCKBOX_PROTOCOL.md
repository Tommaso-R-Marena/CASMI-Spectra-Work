# Frozen v5.3 unseen-collection lockbox protocol

## Purpose

Test whether the already-frozen protected-tail v5.3 selector transfers beyond the OTAVAPEP collection used in the earlier MSnLib development/validation work. This protocol is committed before any ranking result from the target collections is computed.

## Frozen method

- Selector parent commit: `0006c7f81c7dfac53da48690ca0fe627108b9767`.
- Expert score: frozen `joint_mean` exact-condition trajectory score.
- Baseline: `static_best` exact-condition score.
- Selector: protected-tail v5.3 minimax tail.
- `overlap_threshold=0.8`, `protected_k=1`, `overlap_k=10`, `head_survival_slack=1`, `rr_cutoff=25`.
- No parameter may be changed after a target-collection ranking result is opened.

## Target lockboxes

Zenodo record `21105617` (MSnLib spectral libraries), positive MS2 collections:

1. `20241003_enamdisc_pos_ms2.json` — ENAMDISC, Discovery Diversity Set-10 (Enamine).
2. `20241003_mcebio_pos_ms2.json` — MCEBIO, Bioactive Compound Library (MedChemExpress).

OTAVAPEP positive MS2 is used only as the prior-development exclusion set. Every target full InChIKey that occurs in OTAVAPEP is removed before feasibility is counted and before ranking. Candidate pools are built entirely within each target collection.

## Candidate protocol

Use the exact-condition protocol already implemented in `export_msnlib_exact_condition_candidates.py`:

- benchmark identity: full InChIKey + adduct;
- candidate pool: same molecular formula and same adduct;
- query: `SINGLE_BEST_SCAN` spectra;
- candidate reference: `SAME_ENERGY` spectra;
- every candidate must support every collision energy required by its query;
- at least two collision energies and at least two candidates are required;
- ambiguous repeated-acquisition identity groups are excluded rather than arbitrarily resolved.

## Label-free feasibility gate

For each target collection independently, before any static/expert/selector rank is computed, require all three:

- at least 150 eligible ambiguous queries;
- at least 50 molecular-formula blocks;
- at least 25 eligible queries with candidate-pool size >=5.

If a collection fails, ranking for that collection remains unopened. The threshold is not relaxed after seeing coverage.

## Primary and safety endpoints

Primary efficacy endpoint: paired change in tie-aware MRR@25, v5.3 minus static baseline.

Safety endpoint: exact tie-aware Top-1 invariance and zero protected-Top-k candidatewise violations. With `protected_k=1`, any Top-1 change is a protocol failure.

Secondary endpoints: Top-5, Top-25, wins/losses/ties, selector-use fraction, and median expected rank.

## Inference

For every collection that passes feasibility:

- report molecular-formula-block bootstrap 95% CI for query-weighted delta MRR@25;
- report a two-sided whole-formula sign-flip/randomization test;
- run the fixed permuted-expert negative control;
- report the same block-bootstrap inference for the negative control.

A strong positive replication requires: (i) safety endpoint passes; (ii) v5.3 delta MRR@25 > 0; (iii) the 95% formula-block bootstrap CI excludes 0 on the positive side; (iv) two-sided formula-block randomization p < 0.05; and (v) the permuted-expert control does not show a corresponding positive effect. Failure of these criteria is reported as a null/negative replication, not used for retuning.

## Interpretation boundary

These collections are chemically separate from OTAVAPEP after explicit full-InChIKey overlap removal, but they were acquired within the broader MSnLib experimental program and therefore do not establish instrument- or laboratory-independent generalization by themselves. A positive result is intended as prospective cross-collection replication of the frozen selector/expert stack, complementary to CASMI2022 and the MassBank lockbox audits.
