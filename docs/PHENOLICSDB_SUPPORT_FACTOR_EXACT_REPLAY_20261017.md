# Retrospective mechanistic follow-up — after all original PR17 outcomes known

## Objective

Falsify whether the observed quarter-power improvement on the independent
PhenolicsDB source requires **nonlinear redistribution among shared fragments**,
or can be reproduced by simply **suppressing endpoint-exclusive fragments**.

This is NOT a new independent data source or a prospective improvement claim.
It reuses exactly the same 50 outcomes previously scored in workflow 39048117682,
including all 15 exact full InChIKey structures. No row may be dropped by
performance, endpoint similarity, fragment coverage, or apparent success.

## Exact (mathematical) support factorization

For normalized endpoints `a,b`, coordinate `t`, and power exponent `r>0`,
the UNNORMALIZED power path is

`U_r(i) = [(1-t) a_i^r + t b_i^r]^(1/r)`.

On shared bins `a_i,b_i>0`, this is the usual shared-bin power interpolation.
On **low-only** bins, `U_r(i)=(1-t)^(1/r) a_i`. On **high-only** bins,
`U_r(i)=t^(1/r) b_i`. A third kind, bins with both zero, stays zero.

Define the two-parameter factored interpolant `F_(r,g)` with:
- shared bins: `[(1-t) a_i^r+t b_i^r]^(1/r)`
- low-only bins: `(1-t)^g a_i`
- high-only bins: `t^g b_i`
- always one final normalization across all bins.

It follows **algebraically**, with no learned coefficients:

`F_(r,1/r) = P_r` for every nonnegative finite endpoint pair and `r>0`.

Thus for quarter-power `P_0.25 = F_(0.25,4)`, EXACTLY. The following
fixed contrasts hold `g=4` constant, so **exclusive-fragment unnormalized
weights are IDENTICAL**:

1. quarter: `F_(0.25,4)`;
2. fixed previous shared-Hellinger hybrid: `F_(0.5,4)`;
3. fixed previous support-only baseline: `F_(1,4)`.

This makes the specific, testable question: how much does the shared-bin
power law alter final prediction scores **given fixed exclusive survival**?

**Important identification limit:** normalizing once over all bins means
the effect of changing shared-bin curvature can change the normalized
probability on exclusive bins. The additive score contrasts are *not*
a causal decomposition of physical fragmentation or a percentage of
prediction variance explained. If there is NO shared support, all three
paths coincide identically; this is proven by algebra and regression tests.

## Unchanged source, new numerical stress tests

Fetch the original PhenolicsDB 20/40/60/80 eV positive and negative MGF
inputs at the original immutable Git tree commit. Verify their Git blob
SHA-1 and byte sizes. Reuse the **exact original metadata gate and 50-row
original scored artifact** via GitHub Actions run 39048117682. Verify that
the 0.01 Da re-scoring reproduces *every original quarter, Hellinger and
support-only score* to absolute 1e-10, fail closed otherwise.

Perform reference-only grid/matching at **four independently fixed
tolerances**: 0.0025, 0.005, 0.01 and 0.02 Da. At each tolerance:
- retain the SAME original all-positive/all-negative 50 query population;
- do not change query energy coordinate, model coefficient or sample;
- recompute `quarter - Hellinger`, `quarter - support-only`,
  `quarter - shared-Hellinger hybrid`, `hybrid - support-only`;
- record endpoint overlap, number of truly shared reference bins, unmatched
  query intensity and polarity-specific descriptive deltas;
- use exact enumerated 2^15 = 32,768 molecular-identity *block* sign flips
  (no Monte Carlo approximation), plus deterministic 5,000-draw 95%
  chemical-identity-block bootstrap.

All 50 rows come from only 15 distinct chemical identities; positive and
negative polarities and two target energies of the same molecule are ONE
resampling block. This is **posthoc sensitivity**, not a collection of four
new preregistered p-values; no threshold is optimized after scoring.

A sign reversal with smaller tolerance or no quarter advantage over
support-only should be explicitly treated as undermining any
geometry-specific physical claim. Do not run hyperparameter search in
this original dataset to choose a new 'best' exponent.

## Deployment restrictions

Keep original CI, EAD V1 failed gate, EAD V2 failed hybrid, ExpoLib transfer
negative, and the MassBank and MultiMS2 identification-null results
unchanged. Do not silently merge. This work is a mathematical identity and
retrospective sensitivity; a genuine predictive scientific breakthrough
still requires a larger, *newly acquired* independent chemical cohort and
actual prospective molecular-identification gains.
