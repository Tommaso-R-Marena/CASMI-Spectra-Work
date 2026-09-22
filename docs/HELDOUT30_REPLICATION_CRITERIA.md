# Held-out 30 eV replication criteria

Frozen before the MCEDIV_50k_Sub held-out-30-eV rankings were opened.

## Fixed protocol

- Query: `SINGLE_BEST_SCAN` at exactly 30 eV.
- Candidate references: `SAME_ENERGY` at exactly 20 and 60 eV; 30 eV is unavailable to every candidate reference.
- Candidate pools: same molecular formula and same adduct.
- Identity: full InChIKey + adduct.
- Formula-disjoint deterministic validation split: salt `msnlib-heldout30-formula-v1`, validation fraction 0.5.
- Static baseline: best endpoint Bhattacharyya/Hellinger affinity.
- Interpolation expert: energy-constrained 30 eV interpolation, 0.75×20 eV + 0.25×60 eV.
- Energy-blind control: 0.5×20 eV + 0.5×60 eV.
- v5.3 parameters remain frozen from selector development: overlap threshold 0.8, protected k=1, overlap k=10, head-survival slack=1, minimax tail.
- Independent inference unit: molecular-formula block.

## Feasibility gate

Rankings remain unopened unless the validation cohort has all of:

1. at least 300 queries;
2. at least 100 molecular-formula blocks;
3. at least 100 queries whose candidate pool contains at least five structures.

A failed gate is an underpowered feasibility result, not evidence for or against efficacy. The gate will not be relaxed after inspection.

## Claim A: held-out-energy interpolation mechanism

The MCEDIV replication counts as a confirmatory replication of the ENAMDISC interpolation signal only if the predeclared validation comparison `interpolation - static` simultaneously has:

1. positive query-weighted Δ tie-aware MRR@25;
2. a formula-block bootstrap 95% interval with lower endpoint strictly greater than zero;
3. a two-sided formula-block sign-flip/randomization p-value below 0.05.

Win/loss/tie counts and Top-1 are reported descriptively. A positive comparison against the energy-blind 50/50 mixture strengthens energy-specific interpretation but is not required for the primary replication criterion.

## Claim B: protected-tail v5.3 wrapper efficacy

v5.3 is evaluated separately from the interpolation mechanism. Confirmatory efficacy of the wrapper requires the same three conditions above for `v5.3 - static`. Regardless of efficacy, the protected-head safety claim requires zero protected-candidate Top-k violations and exact Top-1 invariance with `protected_k=1`.

Failure of Claim B does not convert a successful Claim A into a v5.3 success; the two claims must remain separate.

## Interpretation discipline

- No selector threshold, interpolation weight, split salt, pool definition, or inferential test will be changed in response to MCEDIV outcomes.
- Candidate-pool-size subsets are sensitivity analyses, not replacements for the full validation endpoint.
- ENAMDISC is the discovery/prospective-origin result for this held-out-energy mechanism; MCEDIV is the independent confirmatory library. MCEBIO failed the unchanged feasibility gate and its rankings remain unopened.
- A successful MCEDIV Claim A would support reproducible condition-aware interpolation within the MSnLib acquisition/processing ecosystem. It would not, by itself, establish cross-platform, cross-instrument, or universal molecular-identification superiority.
