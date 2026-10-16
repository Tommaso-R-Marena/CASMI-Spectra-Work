# Exact support-suppression mechanism of normalized power interpolation

This is an algebraic fact independent of any experimental outcomes. It identifies a concrete mechanism to falsify rather than attributing observed accuracy automatically to a unique statistical-mechanical geometry.

For two discrete probability mass functions `a,b` on a common *reference-only* peak grid and `0<t<1`, define the UNNORMALIZED power interpolant

`U_r(i) = [(1-t) a_i^r + t b_i^r]^(1/r)`, `r>0`.

The deployed predictor is `P_r(i)=U_r(i)/Σ_j U_r(j)`. The normalization depends on every bin, so care is required: a statement about *unnormalized* suppression is not automatically an absolute statement about the normalized probability at a bin.

## Exact edge cases

**Left-exclusive bin** `a_i>0, b_i=0`: `U_r(i)=(1-t)^(1/r) a_i`.

**Right-exclusive bin** `a_i=0, b_i>0`: `U_r(i)=t^(1/r) b_i`.

**Equal shared bin** `a_i=b_i>0`: `U_r(i)=a_i` for ALL `r>0`.

At `t=1/2`, the exclusive-bin *unnormalized* coefficient is `2^(-1/r)`:
- `r=1/4`: coefficient `1/16`.
- `r=1/2` (Hellinger chord): coefficient `1/4`.
- `r=1`: coefficient `1/2`.

Consequently, relative to an equally weighted shared bin of fixed mass, quarter-power gives exactly **one-fourth** the exclusive-to-shared ratio of Hellinger for an exclusive bin. Since normalization cancels in within-predictor ratios, this relative-ratio conclusion is true for the final normalized predictor as well. It does **not** prove quarter-power has lower *absolute* normalized exclusive probability in every possible endpoint configuration.

These identities imply that part of the observed geometry advantage may be *suppression of fragments appearing at only one reference energy*. They are compatible with prior support-only baseline results and with source-specific superiority reversals.

## Frozen endpoint-only adaptive hypothesis

Let `s(a,b)=Σ_i sqrt(a_i b_i)` be reference-spectrum Bhattacharyya affinity in `[0,1]`.

Define `r_*(a,b)=1/4+(1/4)s(a,b)`. This rule is fixed before examining any MultiMS2 outcomes:
- when endpoints have zero affinity (`s=0`), `r_*=1/4`;
- when the endpoint distributions coincide (`s=1`), `r_*=1/2`;
- between them, the rule smoothly interpolates between stronger and weaker transient-peak suppression.

It is a *phenomenological endpoint-only mechanistic hypothesis*, not a theorem that the underlying fragmentation process follows a unique power geodesic. A prospective failure on MultiMS2 counts as evidence against this particular rule, not merely a calibration artifact to be optimized away.

## Scientific consequences

1. Test support-only competitors and answer-aware globally optimized Hellinger interpolation, not just fixed Hellinger.
2. Report endpoint overlap, unmatched target mass and whether the rule improves the actually evaluated downstream structure ranking.
3. Stratify by acquisition family and spectra, but retain full original denominator and block uncertainty to avoid multiple energy observations masquerading as independent biological samples.
4. If quarter power beats Hellinger only because of exclusive support suppression, the next contribution is a calibrated, uncertainty-aware support model, not a discovery of universal thermodynamic geometry.
