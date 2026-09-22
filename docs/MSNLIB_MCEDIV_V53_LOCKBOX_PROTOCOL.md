# Frozen v5.3 MCEDIV lockbox protocol

This protocol is committed before any MCEDIV ranking result is computed and uses the same frozen selector and inferential criteria preregistered for ENAMDISC/MCEBIO.

## Target

- Zenodo record: `21105617`.
- File: `20250828_mcediv_50k_sub_pos_ms2.json`.
- Expected MD5: `c4b7a8e4348b6c4181fc51167cf453a3`.
- Library: MCEDIV_50k_Sub, 20,000-compound subset of the MedChemExpress diversity library.

## Frozen method and exclusions

- Frozen selector parent: `0006c7f81c7dfac53da48690ca0fe627108b9767`.
- Baseline: exact-condition `static_best`.
- Expert: exact-condition `joint_mean` trajectory score.
- v5.3 parameters: overlap threshold 0.8, protected_k 1, overlap_k 10, head-survival slack 1, RR cutoff 25.
- Every full InChIKey appearing in the prior OTAVAPEP positive-MS2 collection is excluded before feasibility counting or ranking.
- Full InChIKey + adduct identity; same-formula/same-adduct candidate pools; >=2 collision energies; SINGLE_BEST_SCAN query versus SAME_ENERGY references; ambiguous repeated-acquisition identities excluded.

## Gate before ranking

Ranking remains unopened unless all are met:

- >=150 eligible ambiguous queries;
- >=50 molecular-formula blocks;
- >=25 eligible queries with candidate-pool size >=5.

These thresholds will not be relaxed after coverage is observed.

## Endpoints and inference

Primary: paired tie-aware delta MRR@25, v5.3 minus static. Safety: exact Top-1 invariance and zero protected-Top-k candidatewise violations. Secondary: Top-5, Top-25, wins/losses/ties, selector use, median expected rank.

Inference is clustered by molecular formula: 20,000-replicate block bootstrap CI and whole-formula two-sided sign-flip/randomization test. The fixed permuted-expert negative control receives the same analysis.

A predeclared strong positive replication requires all of: safety passes; delta MRR@25 > 0; bootstrap 95% CI lower bound > 0; two-sided formula-block randomization p < 0.05; and the permuted-expert control does not show a corresponding positive effect. Any other outcome is reported as null/negative and is not used to retune v5.3.

MCEDIV is a chemically separate collection but remains within the broader MSnLib acquisition program; even a positive result is cross-collection rather than fully laboratory/instrument-independent replication.
