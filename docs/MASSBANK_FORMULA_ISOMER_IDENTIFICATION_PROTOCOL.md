# Locked post-release MassBank formula-isomer identification test (2026-10-16)

**Research class:** NEW DOWNSTREAM TASK ON **ALREADY-USED MASSBANK 2026.03**, NOT a new independent or prospective acquisition. The outcome and feasibility of the prior MassBank *affinity* experiments are already known. The identification metrics below have not been inspected at protocol authoring. Report this as a retrospectively designed falsification/translation experiment, never as a new blinded external replication.

## Biological/chemical question

Does fixed power interpolation r=0.25, selected on MSnLib MCEBIO-positive discovery data, improve **identification of the correct molecular connectivity** among isomeric candidate structures over fixed r=0.5 Hellinger, when the precursor formula and experimental conditions are held constant? Better affinity to the correct spectrum alone does not guarantee better ranking against isomers.

## Immutable input and label-free selection

1. Reuse exactly the original MassBank 2026.03 SHA-256 `12c6a922929f9bf2484c46586b6101512a415e44680558883b0630c6e47905fb` (MD5 `1de9018361f7ffc689aa327f92e97e0f`) and the same original metadata-only `label_free_triplets.jsonl` artifact from run 38918658343 (its byte SHA-256 will be checked). No re-selection by model performance.
2. Construct candidate pools **using metadata only**. Group by exact formula + ion mode + adduct + contributor + instrument model + fragmentation type + interior query collision energy (numeric eV), with all entries from the unchanged eligible MassBank manifest. Require at least two candidate molecules with distinct **InChIKey first blocks (14-character 2D connectivity)**; reject any group containing repeated connectivity, rather than choosing the most favorable replicate.
3. This deterministic rule yields **105** groups, **252** target queries and **44** molecular-formula blocks in five contributor families in the already-audited manifest. Abort without scoring if these fixed counts change. Candidate pools contain 2–10 distinct 2D structures. The same set of formula-matched candidates is offered to **every** query in that pool.
4. For leakage sensitivity, use an external, MD5-verified inventory of the four previously evaluated MSnLib cohorts (MCEBIO-negative, MCESCAF-positive, ENAMMOL-positive, ENAMDISC-negative), already constructed by run 38934839645. The fully nonoverlapping *candidate-pool* subgroup excludes **all** groups containing any structure in that inventory, expected **94** groups / **230** targets / **40** formulas. This subgroup is descriptive, not a newly sampled validation dataset. The original MCEBIO-positive discovery keys were already excluded in the frozen MassBank manifest.

## Information barrier

- For every candidate structure, build predictions from **its own lower/higher-energy spectra only** using reference-only 0.01 Da mass bins, positive-intensity normalization, and physical-energy t; **never use the target-energy spectrum** of any candidate during reference construction. All predictions are precomputed before accessing target spectra.
- Compare each withheld query spectrum to *each* candidate's separately constructed predicted reference grid using the existing one-to-one m/z matcher; retain unmatched query mass in the denominator. Do not share bins derived from the target between candidates.
- Exact-match connectivity labels are used **only** to score final ranks. If a correct candidate is absent or duplicated, abort.
- Frozen methods only: r=.25 power mean, r=.5 Hellinger chord, r=1 linear intensity mixture, and previously implemented power=4 support-only suppression. **No fitting, grid tuning, energy warp, per-formula weighting or model selection on target outcomes.**
- Validate that the affinity for each target against its own predicted target energy exactly reproduces the archived original MassBank r=.25, Hellinger and support-only affinities to 1e-9. Abort if not.

## Frozen endpoints and statistical plan

- **Primary endpoint:** paired MRR at candidate pool size (rank reciprocal of correct first-14 InChIKey), quarter minus fixed Hellinger, all 252 targets. Ties broken only by lexicographic connectivity key so no model enjoys favorable truth tie-breaking. Report **top-1 accuracy**, MRR, wins/losses/ties in reciprocal rank, average correct-vs-highest-wrong affinity margins, and chance baseline `mean(1/k)` for top-1 and `mean(H_k/k)` for MRR with uniformly random rankings, where k is pool size.
- **Exploratory comparators:** quarter vs r=1 linear and support-only. Report non-overlap group subgroup and instrument/contributor strata in full, including negative effects and tiny subgroups.
- Use **formula-block bootstrap** with 10,000 whole-formula resamples and **two-sided formula-block sign flips** with 50,000 draws, both seeded 20261016, on the *query-weighted* paired reciprocal-rank difference. Include all blocks even if zero effect; predeclare that a signflip p value is exploratory under repeated reuse of this MassBank source.
- **Do not claim identification breakthrough** for a null top-1/MRR result even if correct-target spectral affinity improves; indeed a null result would be an important falsification of practical utility. Do not claim the fixed geometry uniquely captures fragmentation physics even for positive ranking results.

## Acceptance/guardrails

Only run if original score artifact and eligible manifest have the exact frozen SHA-256 digests and the input MassBank zip hash matches; both the original 2,138-row population and 105/252/44 candidate counts are unchanged. Never modify the original scoring protocol or original MassBank efficacy result. In this evaluation, candidate fingerprints, external structure databases and other CASMI candidate-generation channels are intentionally absent; rankings address a **small exact-formula isomer choice** and are not a proxy for full candidate generation. All results must be permanently reported without post-hoc subset choice.
