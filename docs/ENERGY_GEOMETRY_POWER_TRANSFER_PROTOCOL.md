# Frozen transfer test for the MCEBIO-selected power geometry

Status: frozen after the MCEBIO formula-disjoint identification result selected the lower-bound candidate `alpha=0.25`, but before `alpha=0.25` is evaluated on the independent MSnLib replication libraries under this protocol.

## Purpose

The MCEBIO discovery half selected `alpha=0.25` from the predeclared power family and the untouched MCEBIO validation half showed a large advantage over the Hellinger chord (`alpha=0.5`). Because `0.25` was the lower boundary of the candidate grid, this does **not** identify an optimal universal exponent. The immediate question is narrower and cleaner:

**Does the already-selected alpha=0.25 path transfer unchanged across independent chemical libraries and ionization polarities?**

No lower alpha is explored until this transfer test is complete.

## Frozen predictors

Use the unchanged generalized LOEO selection rule, strict diameter-bounded 0.01 Da m/z alignment, L1 normalization, and Bhattacharyya affinity from the prior protocol.

Evaluate:

1. `best_endpoint`: best of the measured bracketing endpoint spectra;
2. `linear`: power geometry with `alpha=1`;
3. `hellinger_chord`: power geometry with `alpha=0.5`;
4. `alpha_quarter`: power geometry with **alpha=0.25** fixed from MCEBIO discovery;
5. `warped_hellinger`: Hellinger chord using the MCEBIO discovery-fitted warp fixed at `a=1.1884079253066788`, `b=0.44288663984932064`;
6. `oracle_probability_mixture`: per-query, answer-aware best convex probability mixture, retained only as a nondeployable diagnostic ceiling for scalar endpoint weighting.

No parameter may be refit per transfer dataset.

## Dataset gate

Before any prediction score is computed, each library must independently pass the original generalized LOEO label-free gate:

- >=1000 held-out query triplets;
- >=500 structure+adduct groups;
- >=300 molecular-formula blocks;
- >=500 off-center triplets.

Failed cohorts remain unopened.

## Primary transfer claim

For every independently eligible dataset report `alpha_quarter - hellinger_chord` using molecular-formula-block inference. A dataset supports transfer only if:

- mean paired delta >0;
- 95% formula-block bootstrap lower endpoint >0;
- two-sided formula-block randomization p<0.05.

Also report `alpha_quarter - oracle_probability_mixture`, `alpha_quarter - best_endpoint`, and fixed `warped_hellinger - hellinger_chord`.

## Interpretation

- Repeated alpha-quarter > Hellinger across independent libraries/polarities falsifies a Hellinger-exponent-specific mechanism within the tested family and supports a broader nonlinear power geometry.
- Failure to transfer would indicate MCEBIO-specific exponent selection despite the within-MCEBIO held-out result.
- Even successful transfer does not establish `alpha=0.25` as optimal because it was a boundary candidate. Only after transfer may a lower-alpha refinement be frozen and tested on data not used to choose the refined exponent.
