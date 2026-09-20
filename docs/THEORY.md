# Quotient Fragmentation Dynamics (QFD)

## Exact statements encoded in the current test suite

1. **Normalization gauge.** For `x(E)=exp(EQ)x0`, `Q -> Q+cI` multiplies the trajectory by `exp(cE)` and disappears under homogeneous normalization.
2. **Fragment-energy quotient.** If `X_fe -> a_f b_e X_fe`, then logs change by a row potential plus an energy potential. The nuisance-free subspace is the cycle space of the observed bipartite fragment-energy graph.
3. **Information dimension.** The number of independent nuisance-free log-intensity relations is the cycle rank `|O|-|V|+c`; for a full `n x m` table it is `(n-1)(m-1)`.
4. **Two-energy specialization.** Centered log fold-change cancels fragment-specific response and spectrum-wide gain.
5. **Merged-energy geometry.** Nonnegative raw mixtures lie in a positive cone; after L1 normalization they lie in the convex hull of normalized component spectra.

## Modeling assumptions

The censored quotient likelihood models log-intensities as candidate mean + fragment nuisance + energy nuisance + Gaussian noise, with missing peaks treated as left-censored observations. This is a statistical model to be validated, not an exact law of CID.
