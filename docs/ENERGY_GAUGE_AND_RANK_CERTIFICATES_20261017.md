# Exact energy nonidentifiability and certified molecular-rank stability

## Mathematical statement 1 — disjoint-support energy-gauge nonidentifiability

For two fully disjoint finite support probability distributions a and b,
the normalized power interpolation with exponent alpha satisfies

`P_alpha(t) = (1-w_alpha(t))a + w_alpha(t)b`, where
`w_alpha(t)=sigmoid(logit(t)/alpha)`.

Hence, for arbitrary positive alpha,beta and t in (0,1), setting

`s=sigmoid((beta/alpha)*logit(t))`

gives **P_alpha(t)=P_beta(s) exactly** for every disjoint-support pair
(a,b). Continuous extension s(0)=0,s(1)=1. For any two exponents the
correspondence is strictly increasing (positive alpha,beta).

**Interpretation:** On disjoint support, a fitted power exponent and an
unrestricted monotone collision-energy recalibration are observationally
nonidentifiable from the spectral path alone. Claiming to have found a
unique physical exponent requires BOTH shared-support evidence and
independently grounded collision-energy calibration. This mathematical
nonidentifiability is not a physical proof that fragmentation actually
follows any of these interpolation paths.

Implementation: `disjoint_support_energy_reparameterization`.
Randomized tests check the equality across thousands of parameter pairs.

## Mathematical statement 2 — support-specific TV and affinity bound

Hold the UNNORMALIZED exclusive-support survival terms E identical, and
consider two shared-support terms S and T. The two normalized predictions
are `p=(S+E)/A`, `q=(T+E)/B`, with positive sums A,B. Define
`d=||S-T||_1`. Then by the triangle inequality and `|A-B|<=d`,

`TV(p,q) <= d / max(A,B)`.

For ANY held-out normalized or subnormalized query mass v, including an
unmatched target-mass fraction,

`|BC(v,p)-BC(v,q)| <= ||sqrt(p)-sqrt(q)||_2`
`<= sqrt(2 TV(p,q)) <= sqrt(2 d/max(A,B))`.

The first (often tighter) bound is computed entirely from prediction
distributions built from candidate reference spectra. No held-out target
spectrum, class label, or identification result is needed to compute it.
It is capped at 1 because affinities lie in [0,1].

Implementation:
`score_perturbation_certificate` for same exclusive terms,
`distribution_affinity_difference_upper` for ANY two normalized spectral
predictions.

## Mathematical statement 3 — top-1 isomer candidate invariance

Suppose a fixed exact-formula candidate set has one candidate's baseline
affinity score `s_w` larger than each competitor `s_j`. For a proposed
spectral model update, suppose each candidate has a certified,
target-independent bound `|s'_k - s_k| <= eps_k`.

**Sufficient (not necessary) guarantee:**

If `s_w - s_j > eps_w + eps_j` for every other candidate j,
the same candidate remains the UNIQUE top-1 under the updated model.

For near-ties or insufficient margins the certificate MUST say
`UNKNOWN_NOT_PROVED`, never that a rank switch necessarily occurs.
The margin itself comes from baseline query scoring; only the perturbation
budget is target-independent. This is a post-scoring rank stability audit,
not a prospective label-free identification predictor.

## Real-data falsification

Recreate **exactly the same 63 held-out query / 29 same-formula candidate
groups** from prior Meta-only CID+EAD isomer-gate and score artifacts of
run 39047245864. SHA-verify the original MultiMS2 raw MGF and preserve all
hard competing full InChIKeys. Use candidate 20/60 eV CID endpoints only
to prepare fixed 40 eV predictions and bounds. Held-out 40 eV targets are
then scored without changing candidate sets or bins.

Replay each originally published CID-Hellinger and CID-quarter
reciprocal-rank/top-1 number *exactly*; abort on even one 1e-12 mismatch.
Compare these parameter-free fixed CID models: Hellinger chord, quarter,
shared-Hellinger/exclusive-fourth hybrid, support-only. Quantify and
export how often the reference-only score-margin bound **certifies**
that top-1 CANNOT change between these predictors, versus observed
number of switches. Certified cases MUST agree with observations; fail
the workflow if contradicted.

No trained model, no new formula holdout, no fourth laboratory. All
quantitative claims here are **retrospective descriptive** since PR16's
real-data molecular-rank outcomes were observed before these questions
were asked. This mathematical bridge can explain *why* molecular
prediction-affinity gains do not necessarily change molecular rankings,
but cannot itself establish improved identification or a novel physical
fragmentation law.

## Next breakthrough criterion

Use this as an auditable planning tool for a **genuinely new laboratory
test set**: locate exact-formula cases with uncertain rankings and
physically interpretable shared-fragment variation. Freeze (1) candidate
library, (2) source acquisition controls, (3) fragmentation model,
(4) recalibration limits, (5) uncertainty/multiple comparisons, and
(6) final CASMI-style reciprocal rank before opening test labels.
Do not confuse a proof that rankings remain unchanged with evidence
that the correct structure's rank improves.
