# Post-hoc falsification plan for the MCEBIO energy-geometry result

This plan is frozen after the prospective generalized LOEO result was observed. It is therefore a robustness/mechanism analysis, not a new confirmatory lockbox.

## Observed result motivating the audit

The frozen MCEBIO generalized LOEO benchmark found that ordinary probability-linear interpolation beat energy-blind and reversed-energy controls but lost to the best measured endpoint, whereas the Fisher-Rao/Hellinger geodesic predictor beat the best endpoint, linear interpolation, energy-blind mixing, and reversed-energy mixing.

The purpose of this audit is to determine whether the geodesic gain is genuinely tied to the square-root probability geometry or can be reduced to a simpler scalar reweighting artifact.

## Frozen post-hoc controls

All spectra use the same strict diameter-bounded 0.01 Da m/z alignment as the prospective LOEO benchmark.

Let `q` be the held-out normalized spectrum, `p0,p1` the bracketing endpoint distributions, `x0=sqrt(p0)`, `x1=sqrt(p1)`, `theta=acos(<x0,x1>)`, and `t=(e-e0)/(e1-e0)`.

### 1. Hellinger chord control

`x_chord = normalize((1-t)x0 + t x1)`, `p_chord=x_chord^2`.

This lies on the same great-circle path as the Fisher-Rao geodesic but uses linear interpolation in Euclidean square-root coordinates rather than linear interpolation in spherical arc length. Geodesic > chord therefore tests the arc-length parameterization, not merely use of the square-root embedding.

### 2. Matched-coefficient probability mixture

For the geodesic coefficients

`a=sin((1-t)theta)/sin(theta)`, `b=sin(t theta)/sin(theta)`,

define

`w_coeff=b^2/(a^2+b^2)` and `p_coeff=(1-w_coeff)p0+w_coeff p1`.

This preserves the geodesic's endpoint coefficient ratio but deletes the cross-term `2ab sqrt(p0 p1)`. Geodesic > matched-coefficient mixture tests whether the square-root cross-term/shared-fragment geometry contributes beyond scalar endpoint reweighting.

### 3. Orthogonal-limit scalar warp

`w_orth=sin^2(pi t / 2)`, `p_orth=(1-w_orth)p0+w_orth p1`.

This is the scalar mixture weight induced by the geodesic when endpoint square-root vectors are orthogonal. It is an energy-only nonlinear-weight control.

### 4. Oracle probability-mixture upper bound

Choose `w* in [0,1]` *after observing q* to maximize

`BC(q,(1-w)p0+w p1)`.

This is deliberately not a deployable predictor. It is a stringent diagnostic upper bound for every method that differs from ordinary interpolation only by choosing a scalar endpoint weight. If the parameter-free geodesic exceeds this oracle, its gain cannot be represented by any convex probability mixture of the two endpoints.

### 5. Oracle geodesic-path diagnostic

Choose `s* in [0,1]` after observing q to maximize affinity along the same Fisher-Rao/Hellinger great-circle path. Record the oracle path affinity and `|s*-t|`. This diagnoses whether physical collision energy is well calibrated to geodesic arc length.

## Analyses

For each fixed comparison, report query-weighted paired mean affinity difference, win/loss/tie counts, 10,000-draw molecular-formula-block bootstrap 95% interval, and 200,000-draw two-sided molecular-formula-block sign-flip test. Report the same quantities by major collision-energy triplet.

Primary mechanistic questions:

1. Does geodesic beat Hellinger chord? (arc-length parameterization)
2. Does geodesic beat matched-coefficient probability mixture? (cross-term / square-root geometry)
3. Does geodesic beat orthogonal-limit scalar warp? (not just a t-only nonlinear weight)
4. How often, and by how much, does geodesic beat the oracle scalar probability mixture? (outside-convex-hull geometric advantage)
5. How close is the energy coordinate `t` to oracle geodesic coordinate `s*`?

No threshold or control in this document will be changed based on the resulting scores.
