# Collision-energy geometry identification protocol

Status: mechanism-identification follow-up. This protocol is frozen after the original MCEBIO LOEO and first post-hoc falsification results were observed, but before the analyses defined here are run. It is therefore a held-out within-dataset mechanism test plus a blueprint for unchanged external-library transfer, not a pristine first-look discovery.

## Question

The prior result already excludes a simple explanation in which collision energy merely chooses a scalar convex weight between the two endpoint probability spectra: the Hellinger predictors can exceed even a per-query oracle convex probability mixture. However, the normalized Hellinger chord outperformed Fisher-Rao arc-length slerp on MCEBIO. The remaining question is therefore sharper:

**Is the square-root/Hellinger embedding itself specifically favored, or is the improvement shared by a broader family of nonlinear endpoint transforms and/or a learned scalar reparameterization of collision energy?**

## Fixed cohort and split

Use the identical generalized LOEO cohort construction and strict diameter-bounded 0.01 Da m/z alignment from `ENERGY_GEOMETRY_LOEO_PROTOCOL.md`.

MCEBIO positive mode is split by molecular formula, never by individual query, using:

- validation fraction: 0.5
- split salt: `energy-geometry-identification-v1`

All model selection and parameter fitting use discovery formulas only. Validation formulas may not influence the selected power exponent, energy-warp parameters, thresholds, scoring metric, or inference procedure.

The analysis proceeds only if each split contains at least 800 eligible held-out query triplets and at least 600 molecular-formula blocks.

## 1. Power-mean geometry family

For normalized endpoint distributions `p0,p1`, collision-energy coordinate `t in [0,1]`, and exponent `alpha>0`, define

`P_alpha(t) = normalize(((1-t) p0^alpha + t p1^alpha)^(1/alpha))`.

This nests the two most important hypotheses exactly:

- `alpha=1`: ordinary probability-linear interpolation.
- `alpha=1/2`: normalized linear interpolation in square-root coordinates, i.e. the Hellinger chord.

The fixed candidate grid is

`{0.25, 1/3, 0.40, 0.45, 0.50, 0.55, 0.60, 2/3, 0.75, 1.00, 1.25}`.

Select exactly one `alpha*` by maximum discovery-set mean Bhattacharyya affinity. Ties within `1e-12` are broken by choosing the candidate closest to `0.5`, then the smaller candidate.

### Primary exponent-specificity test

On untouched validation formulas compare:

1. selected `P_alpha*` vs fixed Hellinger chord (`alpha=0.5`);
2. Hellinger chord vs probability-linear (`alpha=1`).

Interpretation is predeclared:

- If `alpha*` is near 0.5 and does not significantly outperform alpha=0.5 on validation, the data support square-root/Hellinger exponent specificity within this family.
- If an alpha materially different from 0.5 significantly outperforms alpha=0.5 on validation, a Hellinger-specific claim is falsified in favor of a broader power geometry.

No significance test is used to choose alpha.

## 2. Collision-energy scalar reparameterization inside Hellinger space

To distinguish path geometry from calibration of the physical energy coordinate, define a monotone two-parameter warp

`w(t;a,b) = sigmoid(a * logit(t) + b)`,

with `a in [0.25,4]`, `b in [-2,2]`, and numerical clipping of `t` only at `1e-8` for the logit. Identity calibration is `(a,b)=(1,0)`.

Fit `(a*,b*)` on discovery formulas only by maximizing mean Bhattacharyya affinity of the Hellinger chord evaluated at `w(t;a,b)`. Optimization is deterministic L-BFGS-B initialized at identity. Validation compares warped Hellinger vs the unwarped Hellinger chord.

Interpretation:

- A reproducible warp gain means raw collision energy is not the optimal affine coordinate along the Hellinger path.
- It does **not** by itself refute Hellinger path geometry.
- Failure of the warp to improve validation strengthens the parameter-free energy-coordinate claim.

## 3. Scalar probability-mixture impossibility boundary

Retain the per-query oracle convex probability-mixture score as a diagnostic ceiling for the entire class

`(1-w)p0 + w p1, 0<=w<=1`.

This oracle is allowed to observe the held-out query and is therefore not deployable. Hellinger chord or selected power geometry exceeding it on average demonstrates that their gain cannot be represented solely by a scalar convex endpoint weight in probability space.

## Inference

Primary sampling unit: molecular-formula block. All fixed validation comparisons report:

- query-weighted paired mean affinity delta;
- win/loss/tie counts;
- 10,000-draw molecular-formula-block bootstrap 95% interval;
- 200,000-draw two-sided molecular-formula-block sign-flip/randomization test.

A validation improvement is called statistically supported only when its mean delta is positive, the bootstrap 95% lower endpoint is >0, and the two-sided block-randomization p-value is <0.05.

## Replication hierarchy

After MCEBIO discovery/validation selection, freeze `alpha*` and `(a*,b*)` exactly. They may then be evaluated unchanged on independent MSnLib cohorts that passed the original LOEO feasibility gate (including MCEBIO negative, ENAMDISC negative, MCESCAF positive, and other eligible replication libraries). No per-library retuning is allowed.

The strongest possible within-MSnLib conclusion would therefore be:

1. convex probability scalar reweighting is insufficient;
2. a particular nonlinear power embedding transfers across libraries/polarities;
3. its exponent is stably close to `1/2`;
4. any energy-coordinate warp either transfers unchanged or provides no material gain.

Even that would identify a reproducible Hellinger-like spectral geometry within this ecosystem, not a universal physical law. Cross-platform/instrument validation remains required for that stronger claim.
