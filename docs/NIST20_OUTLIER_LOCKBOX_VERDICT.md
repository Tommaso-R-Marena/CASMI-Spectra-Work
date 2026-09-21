# NIST20 Outlier Third-Domain Lockbox — Final Verdict

**Status:** Confirmatory lockbox complete. **Replication gate: FAIL.**

This document is a post-run scientific audit of the immutable one-shot NIST20 Outlier result. It does **not** modify the lockbox result, frozen model, primary estimand, or preregistered decision rule. Any diagnostics below the “Post-hoc” heading are exploratory only.

## Confirmatory result

The CASMI-2016-frozen energy-consensus ranker was evaluated once on 300 unique NIST20 `[M+H]+` truth molecules, with one prospectively selected query spectrum per molecule. Candidate spectra were published pCASMI CFM predictions at 10/20/40 eV. The exact frozen model SHA-256 was `5f5649d8e771babfbb817e780873d1b691b714ecc814c11cae46a9a5e2a871eb`.

Coverage was essentially complete: all 300 preselected truth molecules were scored, 299/300 had complete candidate prediction coverage, minimum coverage was 0.998851, and no query was excluded for missing truth predictions or coverage below 0.99.

| Metric | Static | Frozen gated consensus | Difference |
|---|---:|---:|---:|
| MRR@25 | 0.074434 | 0.072636 | **-0.001798** |
| Top-1 tie-aware credit | 0.023452 | 0.023729 | +0.000278 |
| Top-5 tie-aware credit | 0.125162 | 0.115808 | **-0.009354** |
| Median expected rank | 106.0 | 91.25 | -14.75 ranks |

For MRR@25, the paired block bootstrap 95% CI for the difference was **[-0.016451, +0.013096]**. The two-sided paired sign-flip test gave **p = 0.813596**. The gated method produced 32 reciprocal-rank wins, 32 losses, and 236 ties. It also caused 1 static-top-3 query and 2 static-top-5 queries to fall below rank 25.

The preregistered replication gate therefore failed: the MRR point estimate was not positive, the bootstrap lower bound was not positive, the sign-flip test was not significant, and the zero-catastrophe criterion was violated.

## Why this is a credible negative result

The run passed the complete test suite and frozen-model preflight before NIST20 ranking. Query selection was deterministic and outcome-independent: within each truth `mol_id`, the lowest `spec_id` after stable sorting was selected. The resulting 300 selected `spec_id`s exactly match the 300 per-query result rows and all 300 truth molecules are unique.

The evaluator asserted candidate-order invariance for every method on every scored query. Candidate coverage sensitivity is essentially irrelevant here because 299/300 queries have 100% supported candidate coverage; restricting to those 299 produces the same conclusion (`ΔMRR = -0.001804`, sign-flip `p = 0.814204`).

The workflow artifact is preserved as GitHub Actions artifact `10645119389`, digest `sha256:14ad27fd4eefec676746fccfa4cfd97bc42d3bd532a76632f91630ddede96e78`.

## Relation to the earlier evidence

The CASMI-2016 nested out-of-fold development result was genuinely positive: gated consensus improved MRR@25 by about **+0.05556**, with bootstrap 95% CI `[+0.01960, +0.09596]` and sign-flip `p ≈ 0.00398` on 71 queries.

The one-shot CASMI-2022 lockbox was encouraging but did **not** meet the strict preregistered replication gate: gated `ΔMRR ≈ +0.01540`, bootstrap 95% CI `[+0.00084, +0.03218]`, but two-sided sign-flip `p ≈ 0.05420`.

NIST20 now goes in the opposite direction. The correct combined interpretation is therefore **heterogeneous transfer**, not general cross-domain superiority.

## Post-hoc diagnostics — exploratory, not confirmatory

The NIST20 failure is structured rather than random.

The gate used the expert on exactly 150/300 queries and fell back to static on 150/300. Every fallback query is therefore unchanged. Within the 150 expert-used queries, static MRR was 0.117432 and gated MRR was 0.113835 (`Δ = -0.003597`).

The largest weakness is protection of already-good static hits. Among the 36 queries with static expected rank ≤5, the expert was used on 28; only 3 improved in reciprocal rank while 16 worsened. By contrast, among static ranks 26–100 there were 7 wins and only 2 losses, and among ranks 101–500 there were 4 wins and no reciprocal-rank losses.

This explains the apparent paradox: the median expected rank improves from 106.0 to 91.25 even though MRR and Top-5 worsen. Broader rank coverage improves at some thresholds: Top-25 credit +0.00385, Top-50 +0.01365, and Top-100 +0.01833. But the high-value head worsens: Top-3 -0.00425, Top-5 -0.00935, Top-10 -0.00512. MRR remains lower at every tested reciprocal-rank cutoff (5, 10, 25, 50, 100, 250) and even without truncation.

### Feature ablation

Reconstructing the frozen features from the complete candidate artifact reproduces the stored expert score to numerical precision. A post-hoc ablation suggests that the `z_neg_energy_spread` term is a major transfer weakness. Removing it while keeping the other frozen weights and the same overlap-gating idea yields Top-5 = 0.130586 versus static 0.125162, but MRR is still slightly lower (`ΔMRR = -0.000579`; exploratory bootstrap 95% CI approximately `[-0.00969, +0.00710]`, sign-flip `p ≈ 0.897`).

Thus this is **not** evidence that simply deleting the spread feature solves the problem. It is evidence that the across-energy coherence assumption deserves re-examination across instrument/prediction domains.

### Gate ablation

The frozen overlap threshold was 0.2. Post hoc, a much stricter threshold of 0.8 uses the expert on only 39/300 queries and gives a tiny positive MRR point estimate (`ΔMRR ≈ +0.00117`), but its exploratory bootstrap CI is approximately `[-0.00574, +0.00735]` and sign-flip `p ≈ 0.748`.

Because threshold 0.8 was identified after looking at NIST20, it cannot rescue the lockbox. It is only a candidate hypothesis for a new model.

## Final scientific verdict

**The current frozen consensus method does not establish general third-domain superiority over static CFM matching. NIST20 is a clean negative replication for the primary MRR@25 claim.**

At the same time, the result is scientifically useful: the trajectory/consensus features contain real ranking signal for some moderate and difficult cases, but the present reliability gate does not adequately protect high-confidence static hits, and the energy-spread feature appears poorly portable. The failure is therefore more informative than a simple null result.

The strongest defensible claim now is:

> A CASMI-2016-trained energy-consensus reranker showed strong nested out-of-fold gains and an encouraging but borderline CASMI-2022 transfer, but failed to improve MRR on a prospectively frozen 300-molecule NIST20 third-domain lockbox. Post-hoc analysis indicates a domain-sensitive tradeoff: the model can promote moderate-rank truths while occasionally demoting already-high-ranking truths.

## Next experiment

NIST20 is now **spent as an independent lockbox** and must not be reused to claim independent confirmation after tuning.

Use NIST20 only as development evidence for a next-generation safety gate that protects high-confidence static hits, rethinks or removes the cross-energy-spread assumption, and calibrates expert use to a stronger reliability signal than top-10 set overlap alone. Freeze that revised method before evaluation on a **new untouched fourth domain**. That fourth-domain result—not a tuned NIST20 rerun—should decide whether the revised idea genuinely generalizes.
