# Exact one-to-one spectral matching: falsification of the greedy scoring approximation

Research-only continuation on 2026-10-17 after the original 4,398-query
GNPS→MassSpecGym identification outcome was ALREADY OBSERVED (original run
39056094950). Consequently **all results below are retrospective**, and
not a previously untouched, prospective benchmark. No claims of a new
spectral physical law, independently acquired laboratory improvement,
EAD applicability or official CASMI leaderboard gain can be based on it.

## Exact algorithmic statement

For positive-intensity query peaks q_i at m/z x_i, and reference peaks
r_j at m/z y_j, define admissible edges |x_i-y_j| <= **0.01 Da**.
Matchings may select each query or reference peak **at most once**, or
leave it unmatched. Optimize the total weights across every admissible
matching, globally, with deterministic exact assignment (SciPy Hungarian
solver on sparse bipartite connected components; singleton components
are solved by inspection). The weights are fixed as:

- cosine: (q_i / ||q||_2) * (r_j / ||r||_2)
- Hellinger/Bhattacharyya: sqrt(q_i/sum q * r_j/sum r)
- fragment overlap: min(q_i/sum q, r_j/sum r)
- support composite: 0.5 * cosine edge + 0.5 * overlap edge.

**All** unmatched spectral peak mass remains in L1/L2 denominators.
No reference peak grid is allowed to depend on the known chemical
identity, and no outcome truth labels are used for peak assignment.
The exact matching is a purely mathematical refinement of the
previous deterministic *greedy nearest-m/z matching*. Each of the
four per-pair similarity scores is guaranteed >= its old greedy score,
but a structure's ranking can improve OR regress because competitors
also benefit from optimal matching. Bigger affinity != better ID.

Mathematical correctness gate: for >=300 randomized tiny positive-input
problems with up to 4x4 peaks, enumerate ALL admissible matchings by
brute force, including unmatched peaks, and verify optimal scores equal
the brute-force maxima for every metric within 1e-12. Additionally,
>=400 randomized comparisons must confirm monotone nondecreasing
per-pair similarities vs historic greedy matching; deliberately
ambiguous peak graphs must expose a strict greedy failure.

## Strict original data and cohort replay

Reuse exactly the original **FAILED all-polarity V1** and separate
positive-only prescore **PASSED V2** metadata manifests/artifacts;
verify original source SHA-256, 4,398 positive hard native InChIKey
queries, 2,254 native formulas, exact originally SHA-selected reference
candidate indices. Do not change one candidate, source peak, m/z
tolerance, train/split decision, or threshold after observing ranks.
Reference/source annotations may be computationally harmonized, not
independent laboratory ground truth.

Retrieve original 4,398 query-level first-look rank metrics from
run 39056094950, using its immutable artifact.
Before accepting an optimal matching result, recompute the ORIGINAL
historical greedy MRR and fractional top-1 for EVERY query and all
FOUR original methods, and require exact 1e-12 reproducibility.

Then evaluate exactly the SAME source spectra and candidate/replicate
pools using optimal score-dependent matching. Evaluate ALL FOUR
optimal scorers and the old four greedy scorers. Score each candidate
as the max of at most two original selected reference replicates and
respect fractional tied ranks.

## Questions and inferential restrictions

Primary retrospective diagnostic A: `exact-Hellinger vs greedy-Hellinger`.
Primary retrospective diagnostic B: `exact-Hellinger vs greedy-cosine`.
For both, require positive MRR gain, >0 95% native-formula-cluster
bootstrap lower CI, and Holm-adjusted two-sided signflip p<0.05.
These are fixed in code before the **new alignment** is scored, but
because original results are already known, label as
`RETROSPECTIVE_METHOD_AUDIT`, not prospective confirmation.

Also report descriptive MRR/top-1 changes for exact cosine,
exact fragment-overlap, exact composite, number of ambiguous candidate
pairs, number of changed candidate rankings, and per-candidate proof
that exact scores never fall below greedy scores.

A result may be positive (exact matching gives better actual retrieval)
or negative (only spectral similarity improves). Either advances an
auditable methodological understanding and is **NOT a scientific
breakthrough on its own**. If exact matching helps, freeze it unchanged
before testing entirely untouched external acquisition/lab data.

**Explicit preserved negative results:** previous fixed support composite
failed vs Hellinger on 4,398 positive-ion queries (ΔMRR −0.004275,
formula-block p=0.01578), original two-polarity V1 failed (zero negative
eligible queries), and all earlier EAD/ExpoLib and 63-query nulls remain.
