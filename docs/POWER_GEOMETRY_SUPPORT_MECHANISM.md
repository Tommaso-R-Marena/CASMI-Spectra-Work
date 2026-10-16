# Support geometry of the collision-energy power path

This note is a **mathematical mechanism and falsification plan**, not proof of physical fragmentation kinetics or an independent data lockbox. The transfer tests remain frozen at alpha=0.25.

For normalized endpoint spectra `p0,p1`, exponent `r>0`, and collision-energy coordinate `0<t<1`:

```
z_i = ((1-t)*p0_i**r + t*p1_i**r)**(1/r)
P_r(t)_i = z_i / sum(z)
```

## Exact lemma: disjoint supports imply convex mixing

If `p0_i * p1_i=0` for all bins `i`, then on the support of `p0`, `z_i=(1-t)**(1/r)*p0_i`; on the support of `p1`, `z_i=t**(1/r)*p1_i`. Both endpoints have unit total mass, so

```
P_r(t) = (1-w_r(t))*p0 + w_r(t)*p1
w_r(t) = t**(1/r) / ((1-t)**(1/r) + t**(1/r)).
```

Consequently, when supports are completely disjoint, **the entire normalized power path is inside the ordinary probability-mixture segment**. Any empirical affinity advantage over the per-query oracle convex mixture cannot be explained by disjoint-support endpoint-energy warping alone, absent numerical/implementation error.

## Exact endpoint-exclusive fragment law

For a fragment present in `p1` but absent in `p0`, the **unnormalized** contribution is `z_i=t**(1/r)*p1_i`. Thus at r=1, 1/2, and 1/4 it scales as `t`, `t**2`, and `t**4`, respectively. Final normalized intensities also depend on other shared/exclusive peaks; it is invalid to equate unnormalized suppression to a directly measured fragmentation rate.

## Empirical tests capable of falsifying the mechanism

1. Quantify shared-support overlap between bracketing spectra before examining per-query gains; predeclare overlap strata independently of the held-out query.
2. Stratify the fixed r=1/4 minus Hellinger and r=1/4 minus oracle-mixture differences by overlap, energy triplet, and input sparsity with formula-block inference.
3. Test a matched **support-only** suppression control that preserves ordinary mixing of shared peaks but uses the same `t**4` weight on endpoint-exclusive peaks, with complete normalization. If this control ties the power path out of sample, distinctive information-geometric curvature is not established.
4. Repeat on an external instrument/domain, using energy units and instrument metadata that are genuinely comparable, with exact-structure and spectrum-source leakage audited.

An affinity gain is a prediction result under a specific spectral normalization, alignment, and benchmark; it is **not** a physical law, evidence of uniquely optimal r, or an intrinsically causal explanation of fragmentation.
