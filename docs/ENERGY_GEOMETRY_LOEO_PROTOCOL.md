# Prospective generalized collision-energy leave-one-energy-out protocol

Frozen before any MCEDIV_50k_Sub or MCEBIO prediction score from this protocol was computed.

## Scientific question

Treat each normalized MS/MS spectrum as a probability distribution over fragment m/z. Does the known collision energy locate an unmeasured spectrum along a reproducible path between its nearest measured lower- and higher-energy spectra?

This is a spectrum-prediction experiment, not a candidate-ranking experiment. It removes the same-formula-decoy bottleneck while retaining the real-data held-out-energy question.

## Cohort construction (label-free)

For every unambiguous full-InChIKey + adduct group:

1. take each `SINGLE_BEST_SCAN` collision energy `e` that has at least one lower and at least one higher `SAME_ENERGY` reference;
2. withhold all reference information at `e`;
3. choose the nearest available lower reference energy `e0` and nearest available higher reference energy `e1`;
4. define the fixed energy coordinate `t=(e-e0)/(e1-e0)`.

No spectral similarity or outcome is used to select `e`, `e0`, or `e1`.

## m/z alignment

All three spectra are aligned on a tolerance-clustered m/z grid with tolerance 0.01 Da. Every cluster is required to have total diameter <=0.01 Da; transitive single-linkage chaining is forbidden. Each spectrum is L1-normalized after alignment.

Let the resulting fragment distributions be `q`, `p0`, and `p1`.

## Scores and predictors

Similarity is Bhattacharyya affinity

`BC(p,q) = sum_i sqrt(p_i q_i)`.

The following predictors are fixed in advance:

1. **Best endpoint oracle**: `max(BC(q,p0), BC(q,p1))`.
2. **Energy-linear mixture**: `p_lin=(1-t)p0+t p1`.
3. **Energy-blind control**: `p_50=0.5 p0+0.5 p1`.
4. **Reversed-energy control**: `p_rev=t p0+(1-t)p1`.
5. **Fisher-Rao/Hellinger geodesic**: map `p -> x=sqrt(p)` to the positive unit sphere. If `theta=acos(<sqrt(p0),sqrt(p1)>)`, then

   `x_geo = sin((1-t)theta)/sin(theta) * sqrt(p0) + sin(t theta)/sin(theta) * sqrt(p1)`,

   with the coincident-endpoint limit used as `theta -> 0`, and `p_geo=x_geo^2` after numerical renormalization.

The geodesic construction is parameter-free once `e0,e,e1` are known.

## Prospective feasibility gate

A library is scored only if the label-free audit finds all of:

- at least 1,000 held-out query triplets;
- at least 500 distinct full-InChIKey + adduct structures;
- at least 300 molecular-formula blocks.

A failed gate remains unopened and is reported as a feasibility result.

## Inference

Primary sampling/resampling unit: molecular-formula block. The estimand is the query-weighted mean paired change in Bhattacharyya affinity. Report 10,000-draw formula-block bootstrap 95% intervals and a two-sided formula-block sign-flip/randomization test (200,000 Monte Carlo assignments when exact enumeration is infeasible).

### Claim L — energy-aware linear interpolation

Confirmatory support requires **all three** predeclared comparisons to have positive mean delta, bootstrap 95% lower endpoint >0, and two-sided block-randomization p<0.05:

- linear mixture minus best endpoint;
- linear mixture minus 50/50 energy-blind mixture;
- linear mixture minus reversed-energy mixture.

### Claim G — Fisher-Rao/Hellinger geometry

Evidence that the information-geometric path improves on ordinary interpolation requires the same three-part statistical criterion for:

- Hellinger geodesic minus best endpoint;
- Hellinger geodesic minus 50/50 mixture;
- Hellinger geodesic minus reversed-energy mixture;
- and, specifically for superiority over ordinary interpolation, Hellinger geodesic minus linear mixture.

Claim G is not rescued by Claim L; the comparisons remain separate.

## Replication discipline

- MCEDIV_50k_Sub and MCEBIO are evaluated independently under the identical code and criteria.
- No interpolation weight, m/z tolerance, energy selection rule, dataset-specific threshold, or inferential test will be tuned to either library's scores.
- ENAMDISC may subsequently be replayed only as a robustness/triangulation dataset because its earlier 30-eV result has already been observed.
- Success within MSnLib establishes within-ecosystem collision-energy trajectory evidence; cross-platform generalization requires a separate external instrument/domain validation.

## Prior-art guardrail

Spectral interpolation across collision energy is not itself claimed as new: prior work (e.g. SpIfII, J. Chem. Inf. Model. 2022, DOI 10.1021/acs.jcim.2c00620) used vector/PCA-based interpolation to augment MS/MS libraries. The potentially distinct hypothesis tested here is the probability-simplex / Fisher-Rao-Hellinger geometric formulation and its parameter-free geodesic predictor, together with strict prospective leave-one-energy-out controls.
