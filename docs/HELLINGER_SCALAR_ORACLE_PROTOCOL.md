# Hellinger scalar-oracle falsification: frozen retrospective follow-up

Status: **post-hoc hypothesis test, not untouched scientific confirmation**. Frozen before the new oracle scores are computed, after reviewing earlier target-blind r=.25 vs fixed Hellinger and support-only outcomes.

## Exact question

Can the fixed r=.25 predictor's apparent improvement be explained by a *scalar reparameterization of the Hellinger path*, rather than a genuinely different path through the probability simplex?

For each held-out query, allow an **unfair answer-aware oracle** to select the value `u∈[0,1]` that maximizes Bhattacharyya affinity between the query and the normalized square-root chord `P_.5(u)`. The oracle has full access to the withheld query for choosing u, so it is a strict nondeployable ceiling on *any* scalar energy-coordinate warp along the Hellinger chord for those exact endpoint spectra and the target-blind m/z score.

Use exactly the reference-only grid, one-to-one query matching, 0.01 Da tolerance, frozen r=.25, full nonzero query-mass denominator, and MSnLib selection rules from `docs/TARGET_BLIND_ENERGY_GEOMETRY_PROTOCOL.md`. Do not fit data or change interpolation formulas.

Test only the **four cohorts already known to pass the original label-free feasibility gates**: MCEBIO−, MCESCAF+, ENAMMOL+, ENAMDISC−. The other four are documented as excluded on preexisting label-free grounds, not as scientific failures.

## Primary post-hoc contrast

`affinity(P_.25(t),query)-max_{u∈[0,1]} affinity(P_.5(u),query)`

Report mean paired Δ, win/loss/tie, formula-block bootstrap 95% CI (3k draws) and two-sided formula-block sign-flip randomization test (50k draws) independently for each cohort. Apply Holm over the four tested cohorts for transparent multiple testing (the prospective eight-cohort family remains separately reported). Also report fixed r=.25 vs fixed Hellinger on the *same evaluator*, and oracle weight distribution.

## Interpretation

- Positive Δ means the fixed r=.25 point lies closer to the target than **any point on the Hellinger chord** for those spectra under this score, so a scalar reweighting of Hellinger endpoints alone cannot account for the prediction.
- Nonpositive Δ does not imply r=.25 has no practical predictive gain; a deployable fitted time warp is not the same as an answer-aware oracle.
- A positive difference is **not** uniqueness of Amari geometry, an independent physical law, or out-of-domain validation. It remains sensitive to source preprocessing and target-blind m/z matching.
- The optimization is bounded on [0,1], checks both endpoints, and uses deterministic scalar minimization; unit tests verify exact on-curve recoverability and domination of a dense weight grid.

If the oracle almost always wins, the prior geometry-specific claim needs to be downgraded materially regardless of earlier p-values.
