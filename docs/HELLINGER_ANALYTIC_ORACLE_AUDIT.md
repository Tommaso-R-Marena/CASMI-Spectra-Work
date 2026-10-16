# Independent analytic Hellinger scalar-oracle audit — 2026-10-16

**Completed; post-hoc optimizer validity check, not new external scientific confirmation.** [GitHub Actions run 38918394218](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38918394218), four jobs all successful. Tested on the same four already eligible, already-scored MSnLib cohorts as the numerical-oracle study. The original numerical code and experimental selection were left unchanged.

## Exact global optimization

Write `a=sqrt(p0)`, `b=sqrt(p1)`, with both endpoints normalized to probability mass one. Write `u=sqrt(q)`, where `q` may sum to less than one due to target peaks unmatched by the reference-only mass grid. Let `A=<u,a>`, `B=<u,b>`, `c=<a,b>`, all nonnegative, and `k=w/(1-w)>=0`. The normalized Hellinger-chord affinity is

```
F(k) = (A + k B) / sqrt(1 + 2 c k + k^2).
```

The sign of `dF/dk` equals the sign of `(B-Ac) + k(Bc-A)`. This affine function can have at most one interior maximum. When `B-Ac > 0` and `A-Bc > 0`, the unique maximum has

```
k* = (B-Ac)/(A-Bc)
w* = (B-Ac)/((A+B)(1-c)).
```

Otherwise the maximum is achieved at one of the two endpoints; evaluate the endpoint scores in every case, including coinciding endpoint spectra. Therefore the routine in `src/qfd/hellinger_analytic_oracle.py` computes the global maximum over the entire Hellinger chord without numerical search or an assumption of objective unimodality.

## Independently replayed results

| MSnLib cohort | Held-out triplets | Formula blocks | Max absolute numerical vs analytic affinity discrepancy | Mean r=.25 minus **analytic global** Hellinger oracle |
|---|---:|---:|---:|---:|
| MCEBIO negative | 1,511 | 1,341 | 4.44e-16 | +0.0086721302 |
| MCESCAF positive | 1,813 | 1,572 | 5.55e-16 | +0.0247031192 |
| ENAMMOL positive | 2,054 | 556 | 4.44e-16 | +0.0176327109 |
| ENAMDISC negative | 2,918 | 2,030 | 5.55e-16 | +0.0058342982 |

All four per-query audits passed the predeclared numerical-agreement check of 1e-6; there were **zero** queries on which the frozen numerical optimizer exceeded the analytic global maximum by more than 1e-9. The observed differences are at machine floating-point precision, far smaller than any reported quarter-power advantage.

Unit tests additionally checked 500 seeded random sparse/nonnegative mixtures against independent dense chord scans and numerical maximization, exact targets on a Hellinger chord, coincident endpoints, unmatched target probability mass, and invalid inputs. GitHub CI: **147 tests passed**.

## Interpretation

The post-hoc result `r=.25 > best answer-aware point on Hellinger chord` cannot be explained by the original numerical Hellinger optimizer failing to find a better scalar interpolation. In the four already studied MSnLib cohorts, the fixed quarter-power method's mean affinity is genuinely higher than the *global* Hellinger-path optimum under the same target-blind reference grid and score.

This is **not** a new untouched dataset; it does not establish universal exponent optimality, source-independent acquisition, an Amari physical law, or a specific fragmentation mechanism. The strong support-only suppression control still weakens any unique-geometric interpretation. Only an independently sourced frozen external experiment can address generalization.

Primary run and detailed row artifacts: [38918394218](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38918394218). Original numerical run: [38861113132](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38861113132).
