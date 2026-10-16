# Target-blind mass-alignment falsification of MSnLib power geometry

**Completed October 16, 2026 (UTC).** Source: [GitHub Actions run 38847328313](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38847328313), frozen implementation SHA `fd98527278dfa9a44501857905abf0fe0633655d`, protocol `docs/TARGET_BLIND_ENERGY_GEOMETRY_PROTOCOL.md`.

**Study status: POST-HOC robustness and model-specificity falsification**, not a prospective discovery or independent external instrument validation. The earlier positive transfer results were already known when these controls were designed.

## Key correction: target-independent predictor construction

The preceding grid procedure used `_common_grid_strict([query,low,high], 0.01)`. The strict diameter condition prevented transitive peak chaining but still let the held-out query's m/z positions affect how reference peaks were grouped. The new `reference_only_grid` constructs all bins, centers, endpoint mass vectors, and predictions **solely from low/high reference spectra** before observing the target. The target is then one-to-one m/z matched within 0.01 Da, with **unmatched observed mass retained in the denominator**.

Therefore new absolute affinities are systematically lower; old/new **within-procedure paired differences** may be compared as robustness diagnostics, but not interpreted as equal metrics. Each model receives the same reference-only grid.

## Scored outcomes: four originally gate-passing cohorts (8,296 triplets)

| Cohort | Triplets | Formula blocks | r=.25 minus Hellinger r=.5 | Formula-block bootstrap 95% CI | r=.25 minus support-only | Support-only 95% CI |
|---|---:|---:|---:|---|---:|---|
| MCEBIO negative | 1,511 | 1,341 | **+0.024397** | [+0.022666,+0.026211] | **+0.001445** | [+0.000522,+0.002394] |
| MCESCAF positive | 1,813 | 1,572 | **+0.036163** | [+0.034796,+0.037590] | **+0.006760** | [+0.005593,+0.007943] |
| ENAMMOL positive | 2,054 | 556 | **+0.024427** | [+0.022576,+0.026275] | **+0.019007** | [+0.016759,+0.021197] |
| ENAMDISC negative | 2,918 | 2,030 | **+0.020668** | [+0.019486,+0.021861] | **+0.003964** | [+0.003319,+0.004593] |

The four nonpassing original label-free feasibility cohorts (MCEDRUG+/−, MCESCAF−, ENAMMOL−) had **no prediction outcomes computed**; their jobs completed with feasibility/provenance only.

All four fixed r=.25-versus-Hellinger contrasts have two-sided formula-block sign-flip Monte Carlo p ≈ `0.000020` (50,000 randomizations); a conservative Holm correction across **eight** named cohorts remains below 0.00016. Fixed r=.25 versus support-only has p ≈ 0.000020 in three cohorts; on MCEBIO− it is `p=0.003620` (eight-cohort Holm ≈ 0.0181 when unavailable/gate-failed cohorts are assigned p=1). These are exploratory/post-hoc p-values **and do not replace independent prospective validation**.

## Countermechanism: support-only is strong

The parameter-free (relative to this evaluation) support-only comparator performs ordinary `(1-t) p0+t p1` mixing on reference peaks present at both energies, but downweights endpoint-exclusive low/high fragments by `(1-t)^4` and `t^4`, respectively, then normalizes. It explains most of the apparent r=.25 improvement on MCEBIO− and a large fraction in other cohorts.

A crucial heterogeneity warning: on MCEBIO−, fixed r=.25 beats support-only **on only 457 queries versus 843 losses and 211 ties**, even though its query-weighted mean delta is positive. Thus large improvements in a minority of queries drive the mean; no blanket per-molecule superiority is established. For MCESCAF+, the corresponding counts are 874 wins, 914 losses and 25 ties. This finding is consistent with **mechanism heterogeneity**, not a uniquely correct universal power path.

## Convex mixture oracle and m/z-loss diagnostic

The fixed r=.25 predictor also exceeds the per-query answer-aware best **convex probability mixture** on the new target-blind grid: MCEBIO− +0.021239; MCESCAF+ +0.044469; ENAMMOL+ +0.035181; ENAMDISC− +0.018529 affinity. This rules out reproducing the full effect with **convex endpoint scalar weights alone** under the stated metric and alignment. It does not rule out arbitrary nonlinear transformations, denoising, peak filtering, or more flexible learned spectral models.

Mean unmatched *query* intensity fractions under one-to-one matching were approximately MCEBIO− **8.56%**, MCESCAF+ **4.53%**, ENAMMOL+ **8.49%**, ENAMDISC− **10.79%**, respectively. Unmatched mass remains in the denominator and earns zero predictive affinity.

## Exploratory lower exponent r=.125: no consistent advantage

The *post-hoc exploratory* r=.125-minus-r=.25 contrasts were:

- MCEBIO− +0.001082 (95% CI [−0.000751,+0.002928], p ≈ 0.250).
- MCESCAF+ −0.000478 (95% CI [−0.001966,+0.001049], p ≈ 0.534).
- ENAMMOL+ **−0.009816** (95% CI [−0.011736,−0.007799]).
- ENAMDISC− **−0.015654** (95% CI [−0.017552,−0.013782]).

Thus the earlier assumption that lower exponents necessarily improve prediction is **falsified on these datasets**. No universal optimal exponent has been established; additional exponent searches on the same cohorts would be exploratory tuning.

## Scientific conclusion and limitations

The r=.25 transfer advantage is *robust to removal of target-dependent reference-grid construction*. The result is not just an artifact of letting held-out query peaks determine the interpolation basis.

However, **a non-geometric shared-support suppression heuristic explains much of the effect** and is competitive or better on the majority of queries in some cohorts. This is not evidence that mass spectra physically follow an Amari-connection geodesic, nor that r=.25 generalizes across instruments, collision-energy units, or chemically independent corpora.

For genuinely stronger evidence, the next gate is a new source with explicit comparable eV collision energies, independent instrumentation, and exact full-InChIKey discovery exclusion. See `docs/MASSBANK_EXTERNAL_ENERGY_GEOMETRY_GATE.md`. The original eight-cohort MSnLib family is now fully reused and cannot be treated as untouched validation for subsequent models.

## Provenance and reproducibility

- [GitHub Actions run 38847328313](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38847328313) has one artifact per dataset, including input SHA256, feasibility, provenance, `result.json` and per-query `rows.jsonl` for gate-eligible cohorts.
- No data-dependent reweighting or fitted parameters were added.
- Unit tests verify target-independent reference bins, conservation of positive spectral mass, one-to-one matching, unmatched query mass preservation, exact disjoint-support reduction and optimizer controls.
- Tests and experiment workflow complete; this report is a scientific interpretation of that run and does **not** overwrite the earlier frozen results.
