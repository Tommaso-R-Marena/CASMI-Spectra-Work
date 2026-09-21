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
=
\Pr\{e^{-\lambda1^TX}\ge1\}
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
=
\Pr(G)\,\mathbb E[U_{expert}-U_{baseline}\mid G].
\]

This is an exact identity. It does not imply improvement unless the conditional expert advantage is positive on the selected region. In the CASMI implementation, `G` is defined by overlap between the static and expert top-10 candidate sets; its threshold was chosen only inside CASMI-2016 nested validation and then frozen.

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

## 8. What is rigorously established versus empirical

The probability bounds and rank-preservation result are mathematical consequences of their explicit assumptions. The following are empirical questions and require external validation:

- whether CASMI truth-minus-decoy margins satisfy a useful positive-margin sub-Gaussian model;
- whether CFM prediction errors preserve the same structure across years, instruments, chemotypes, and precursor types;
- whether the CASMI-2016-trained conditional ranker transfers to CASMI 2022;
- whether the reliability gate identifies a region of positive expert advantage under distribution shift.

A finite benchmark cannot prove universal metabolite identification. Candidate-generation recall, prediction-model misspecification, ion/adduct changes, instrument-domain shift, stereochemical ambiguity, and unsupported chemistry remain separate failure modes.

The strongest defensible empirical claim after a successful CASMI-2022 lockbox is therefore: **a model and gate frozen on CASMI 2016 transferred to the later CASMI 2022 challenge under the published CFM candidate-prediction setting, with predeclared statistical and safety criteria.** That would be evidence of cross-challenge generalization, not a proof that all small-molecule identification is solved.
