# Energy-Consensus Generalization Theory

This note states the mathematical claims used by the CASMI energy-consensus work and, equally importantly, their assumptions and limits.

## 1. Setup

For a query spectrum, let `c*` be the true candidate and `c` a decoy. Across `m` fragmentation conditions or collision energies, define the truth-minus-decoy margin vector

\[
X_e = s_e(c^*) - s_e(c), \qquad e=1,\dots,m.
\]

Positive `X_e` favors the truth. Let `mu = E[X]`.

## 2. Joint sub-Gaussian consensus theorem

Assume the centered margin vector is jointly sub-Gaussian with a positive-semidefinite covariance proxy `Sigma`, meaning

\[
\mathbb E\exp\bigl(u^T(X-\mu)\bigr)
\le
\exp\left(\tfrac12u^T\Sigma u\right)
\]

for every vector `u`. If `1^T mu > 0`, then the probability that the decoy beats the truth under the sum-consensus score obeys

\[
\Pr(1^T X \le 0)
\le
\exp\left[-\frac{(1^T\mu)^2}{2\,1^T\Sigma1}\right].
\]

### Proof

For any `lambda > 0`, Markov's inequality and the sub-Gaussian MGF bound give

\[
\Pr(1^TX\le0)
=\Pr\{e^{-\lambda1^TX}\ge1\}
\le
\mathbb E e^{-\lambda1^TX}
\le
\exp\left(-\lambda1^T\mu + \frac{\lambda^2}{2}1^T\Sigma1\right).
\]

Minimizing the exponent at

\[
\lambda^*=\frac{1^T\mu}{1^T\Sigma1}
\]

gives the stated result whenever the denominator is positive. If `1^T Sigma 1 = 0` while `1^T mu > 0`, the aggregate margin has zero sub-Gaussian variance proxy and is deterministic in that direction, so the failure probability is zero.

## 3. Top-1 corollary

For decoys `c != c*`, let their margin means and covariance proxies be `mu_c` and `Sigma_c`. A union bound gives

\[
\Pr(c^*\text{ is not Top-1})
\le
\sum_{c\ne c^*}
\exp\left[-\frac{(1^T\mu_c)^2}{2\,1^T\Sigma_c1}\right],
\]

with each term treated as the vacuous bound 1 if its aggregate mean is nonpositive. Independence between different decoys is not required for this union bound.

## 4. IID special case

If, for each decoy, the `m` margins are iid sub-Gaussian with common positive mean `mu` and scale `sigma`, then

\[
\Pr(c^*\text{ is not Top-1})
\le
(N-1)\exp\left[-\frac{m\mu^2}{2\sigma^2}\right].
\]

Thus genuinely complementary conditions can compound evidence exponentially in `m` under the assumptions. Correlated conditions are weaker: positive covariance increases `1^T Sigma 1`, loosening the bound. The theorem therefore does **not** say that adding redundant collision energies automatically helps.

## 5. Why max-over-energy can be fragile

The conventional score `max_e s_e(c)` gives every decoy multiple opportunities for one extreme accidental match and discards whether the other energies agree. The consensus representation used here instead combines:

1. strong best-energy similarity,
2. proximity to the ordered 10 -> 20 -> 40 eV spectral trajectory, and
3. low cross-energy spread.

The theorem above directly justifies aggregation only when truth-minus-decoy margins have positive aggregate mean and satisfy an appropriate tail model. The learned three-feature ranker is an empirical approximation to this principle, not a theorem-derived Bayes-optimal score.

## 6. Selective expert identity

Let `G` be the event that a label-free reliability gate chooses the consensus expert, while the baseline is used on `G^c`. For any integrable ranking utility `U`,

\[
\mathbb E[U_{selective}-U_{baseline}]
=\Pr(G)\,\mathbb E[U_{expert}-U_{baseline}\mid G].
\]

This is an exact identity. It does not imply improvement unless the conditional expert advantage is positive on the selected region. In the original CASMI implementation, `G` is defined by overlap between the static and expert top-10 candidate sets; its threshold was chosen only inside CASMI-2016 nested validation and then frozen.

## 7. Bounded-correction rank-preservation theorem

For scores

\[
S_i=B_i+\alpha h_i, \qquad h_i\in[-1,1],
\]

candidate `i` cannot be reordered below candidate `j` whenever

\[
B_i-B_j>2\alpha.
\]

This follows because the worst possible correction difference is `h_j-h_i <= 2`. Equality at `2 alpha` can create a tie and is therefore not claimed to be protected.

## 8. Protected-prefix / expert-tail theorem

Let the baseline score be `B`. Fix an integer `k >= 1`. Form a protected set `H_k` by taking the baseline top-`k` prefix **and every candidate tied with the baseline candidate at the `k`th position**. Construct any new ranking satisfying all three conditions:

1. the baseline weak ordering, including ties, is unchanged inside `H_k`;
2. every candidate in `H_k` remains above every candidate outside `H_k`;
3. candidates outside `H_k` may be reordered arbitrarily.

Then every candidate in `H_k` retains exactly the same tie-aware rank interval as under the baseline. Consequently, tie-aware Top-`k` credit is exactly unchanged for **every possible truth candidate**, not merely for the observed truth.

### Proof

For any candidate `i` in `H_k`, the set of baseline candidates strictly above `i` and the set tied with `i` are both subsets of `H_k` by construction of a prefix closed under the cutoff tie. Conditions 1 and 2 preserve both sets exactly, so `i` retains its baseline rank interval. Any candidate outside `H_k` remains below the complete cutoff tie block and therefore has zero Top-`k` credit both before and after tail reranking. Hence Top-`k` credit is invariant candidate-by-candidate.

The `protected_tail_consensus_scores` implementation realizes this construction using rank-surrogate scores rather than adding expert and baseline scores. This is intentional: it makes the guarantee independent of score calibration or relative scale.

### Top-1 special case

For `k=1`, `H_1` is the complete static maximum-score tie block. Therefore a protected-head / expert-tail reranker has **exactly zero possible Top-1 regression** under the same tie-aware metric, regardless of how adversarial or distribution-shifted the expert scores are. This is a deterministic, distribution-free safety property; it does not imply that MRR, Top-5, or any other unprotected metric must improve.

### Selective protected-tail version

A label-free gate may first decide whether tail reranking is used. If the gate rejects the expert, all baseline scores are retained; if it accepts, only the tail below `H_k` is reordered. The Top-`k` invariance therefore holds regardless of gate quality. The gate affects only whether the expert is allowed to alter unprotected ranks.

## 9. Why this architecture follows from the NIST20 failure

The one-shot NIST20 result falsified the stronger empirical claim that the original whole-ranking selective expert reliably improves MRR across a third domain. Post-hoc diagnostics showed a more specific pattern: moderate and hard baseline cases often moved upward while some already-high baseline hits were damaged. That observation motivates a new development hypothesis: baseline and trajectory scores may be complementary in different rank regions.

The protected-tail architecture is the minimal way to test that hypothesis without allowing an expert to trade away a predeclared protected metric. NIST20 is development data for this v5 hypothesis and cannot be reused as an untouched confirmation. Earlier CASMI-2022 results likewise remain evidence about the older frozen model; any v5 evaluation on those already-inspected data is an external-to-rule audit, not a new pristine lockbox.

A genuinely confirmatory v5 claim requires a new dataset or otherwise untouched evaluation partition fixed before outcomes are inspected.

## 10. What is rigorously established versus empirical

The probability bounds, bounded-correction result, and protected-prefix theorem are mathematical consequences of their explicit assumptions or constructions. The following are empirical questions and require external validation:

- whether CASMI truth-minus-decoy margins satisfy a useful positive-margin sub-Gaussian model;
- whether CFM prediction errors preserve the same structure across years, instruments, chemotypes, and precursor types;
- whether the trajectory expert has positive conditional advantage in the unprotected tail under distribution shift;
- whether a label-free reliability gate identifies a region of positive expert advantage;
- whether the NIST20-developed protected-tail architecture transfers to a genuinely untouched domain;
- whether alternative collision energies, ion modes, prediction engines, or experimental libraries retain the same complementarity.

A finite benchmark cannot prove universal metabolite identification. Candidate-generation recall, prediction-model misspecification, ion/adduct changes, instrument-domain shift, stereochemical ambiguity, and unsupported chemistry remain separate failure modes.

The strongest defensible future empirical claim after a successful untouched v5 evaluation would therefore be that a predeclared protected-tail rule preserved its protected baseline metric by construction while improving specified unprotected retrieval metrics on that external domain. That would be evidence of cross-domain complementarity, not proof that all small-molecule identification is solved.
