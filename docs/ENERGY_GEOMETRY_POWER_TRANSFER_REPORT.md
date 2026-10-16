# Frozen power-geometry transfer: real-data replication verdict

**Status:** complete prospective within-MSnLib transfer audit. **Scoring workflow:** [GitHub Actions run 38845952532](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38845952532) (October 16, 2026 UTC). The final single-job ENAMMOL+ retry succeeded after an unrelated truncated Zenodo download (curl error 18). The fail-closed eight-cohort artifact `energy-geometry-power-transfer-verdict` reports `COMPLETE`; all four nonpassing feasibility cohorts remain unscored.

The `r=0.25` power-mean spectral interpolator and energy-warp control were selected **on MCEBIO-positive discovery formula groups**, before any of the eight named independent library/polarity transfer outcomes were computed. The frozen protocol is `docs/ENERGY_GEOMETRY_POWER_TRANSFER_PROTOCOL.md`, the scoring script is `scripts/benchmark_msnlib_energy_geometry_transfer.py`, and the frozen scorer commit is `be7813155a5dada31c8c557952a060b664a1e3ac`. No per-dataset parameter fitting occurred.

## What was actually tested

At each held-out internal collision energy `e`, use only its nearest lower and higher reference energies `e0,e1`. Same-InChIKey/adduct identity, strict diameter-bounded 0.01 Da m/z alignment and normalized peak intensity are fixed. Predict the held-out spectrum with
```
P_r(t)_i = [ (1-t) p0_i**r + t p1_i**r ]**(1/r) / Z_r(t)
t = (e-e0)/(e1-e0).
```
Primary predeclared paired outcome: `Bhattacharyya(P_0.25,q) - Bhattacharyya(P_0.5,q)`, using molecular-formula-block 10,000-draw bootstrap intervals and two-sided 200,000-draw block sign-flip randomization tests. `r=0.5` is the normalized square-root/Hellinger **chord**, not the equal-arc-length spherical geodesic. Holm correction counts **all eight predeclared cohorts** to avoid selectively excluding failed feasibility gates.

## Scored cohorts: all support fixed r=0.25

| MSnLib cohort | Queries | Formula blocks | Mean Δ affinity vs Hellinger chord | Formula-block bootstrap 95% CI | Wins / losses / ties | Two-sided p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCEBIO negative | 1,511 | 1,341 | +0.025481 | [+0.023743, +0.027242] | 1,315 / 196 / 0 | 0.000005 | 0.000040 |
| MCESCAF positive | 1,813 | 1,572 | +0.037474 | [+0.036087, +0.038848] | 1,738 / 75 / 0 | 0.000005 | 0.000040 |
| ENAMMOL positive | 2,054 | 556 | +0.026873 | [+0.025312, +0.028489] | 1,907 / 147 / 0 | 0.000005 | 0.000040 |
| ENAMDISC negative | 2,918 | 2,030 | +0.022646 | [+0.021479, +0.023823] | 2,439 / 475 / 4 | 0.000005 | 0.000040 |

The **8,296** queries are the sum of four separately analyzed cohorts, **not** 8,296 independent chemical or instrumental samples. P values are at the resolution floor of the 200,000-draw Monte Carlo procedure; no smaller p is claimed. Molecular-formula block counts must not be summed and described as independent formulas across datasets without auditing between-library identity overlap.

On the **same four scored cohorts**, the frozen `r=0.25` predictor also exceeded a *per-query answer-aware oracle* that chooses the best convex probability mixture of the measured endpoint spectra, with mean deltas:
- MCEBIO−: **+0.020809**
- MCESCAF+: **+0.044658**
- ENAMMOL+: **+0.038769**
- ENAMDISC−: **+0.019612**

This oracle is intentionally nondeployable; exceeding it demonstrates that a scalar convex endpoint-reweighting family is insufficient to account for the affinity results **under this representation and score**. It does not uniquely identify the correct nonlinear family or establish physical kinetics.

## Label-free feasibility failures; outcomes unopened

The other **four** predeclared cohorts failed at least one original gate: `MCEDRUG+`, `MCEDRUG−`, `MCESCAF−`, and `ENAMMOL−`. Gates require at least 1,000 eligible triplets, 500 structure/adduct groups, 300 formula blocks, and 500 off-center triplets. Their positive/negative efficacy outcomes **were never scored**. A completed CI job with no `result.json` after `gate_pass=false` is the correct result.

## Narrow defensible conclusion

Across four real MSnLib library/polarity cohorts, after freezing all parameters on MCEBIO-positive discovery formulas, **r=0.25 consistently outperforms the r=0.5 Hellinger chord on held-out collision-energy prediction** under Bhattacharyya affinity, including formula-block uncertainty and a conservative eight-family Holm check. A broader nonlinear power geometry is empirically better than Hellinger *for this task*.

This **does not** prove a universal or uniquely optimal power exponent, Amari connection, mechanistic fragmentation law, CASMI identification improvement, or cross-instrument generalization. In particular, r=0.25 was a **lower boundary** of the prior exponent grid, so tests at smaller r require a new frozen plan and genuinely separated validation data.

**Next scientifically decisive falsification:** fixed r=0.25 versus independently frozen lower-power exponents, limit r→0, and support-only suppression controls on a new cross-instrument/chemical domain with leakage and energy-unit harmonization audited. See `docs/POWER_GEOMETRY_SUPPORT_MECHANISM.md` for exact support lemmas and limitations.

## Traceability

- Full workflow and per-library artifacts: [Actions run 38845952532](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38845952532).
- Cross-cohort machine-readable verdict: artifact `energy-geometry-power-transfer-verdict` from the **successful retry attempt**, with `status=COMPLETE`, `SCORED=4`, `GATE_FAILED_UNOPENED=4`.
- Data source: publicly released MSnLib Zenodo record `21105617`, file-specific MD5 values and computed SHA-256 recorded in each artifact.
- GitHub artifact retention is finite; keep permanent lightweight summary results alongside code; do not commit copyrighted/raw dataset dumps.
