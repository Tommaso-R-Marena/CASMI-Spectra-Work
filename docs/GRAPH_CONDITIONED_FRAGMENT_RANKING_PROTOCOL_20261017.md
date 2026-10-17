# Graph-conditioned, formula-conserving fragment feasibility v1 — fixed retrospective hypothesis

**As of 2026-10-17 prior to running THIS graph-aware score, full 4,398-query
GNPS→MassSpecGym results and global peak matching have ALREADY been seen.**
This is an explicit retrospective mechanism test, not a prospective held-out
trial. The primary Hellinger baseline previously achieved MRR 0.811962,
fractional top1 0.707398. The prior support composite was worse; exact
global peak matching did NOT improve Hellinger MRR.

## Scientific hypothesis

Previously evaluated methods use **experimental spectrum similarity**,
with no knowledge of which fragment ions a particular candidate chemical
structure can physically support. Same-formula compounds may have very
different connected fragments despite sharing elemental composition.
A fixed candidate-graph fragmentation plausibility term might improve
same-formula molecular ranking. This is not an identified physical
kinetic mechanism, and bond-cleavage heuristics are limited.

## Fixed chemistry before THIS method's scoring

Use only the **GNPS reference candidate's native SMILES**, never the
MassSpecGym query SMILES, true answer InChIKey, or test scores to predict
fragments. Verify reference structure against its **own native molecular
formula** by RDKit. A SMILES that cannot be parsed, fails the formula
check, or has no suitable breaks contributes **zero graph score**; never
exclude the molecule, target query, or candidate. Count/report failures.

Enumerate candidate graph connected fragments formed by breaking exactly
one or two NON-RING SINGLE bonds. Include cuts only if the component
contains between 2 and (total heavy atoms minus 2) heavy atoms. If there
are more than 24 bridge-eligible single bonds, select deterministically
without using any spectral outcome: heteroatom-adjacent bonds first,
then original RDKit atom/bond index. This is a computational cap, not
a fitted hyperparameter.

Compute monoisotopic mass from original subgraph atoms and attached
parent-state hydrogens. Candidate positive-ion fragment hypotheses use
the following FIXED four m/z shifts: -H atom, 0, +H atom, and +proton
(1.007276466621 Da), retaining only positive masses. Round only for
near-exact deduplication (1e-6 Da), not for scoring. This heuristic
permits common ion/hydrogen rearrangements, but can miss multi-step or
ring-opening fragments and can admit chemically impossible ions; an
empirical win would NOT validate these pathways physically.

For each heldout query peak and each candidate, check whether at least
one graph-derived fragment m/z lies within **0.01 Da**. For each peak,
weight a candidate's match by its **rarity across ALL fixed chemical
candidates in the same metadata-only pool**, `1-df/K`, where df is
the number of candidate structures able to match the peak and K is
the number of fixed candidate InChIKeys. A universally plausible peak
gets weight zero; a structure-specific fragment gains weight.
Combine contributions with the query's original normalized ion intensity
and preserve all unmatched query mass. This is fully label-blind.

Rank candidates with fixed score:

`graph_conditioned = 0.90 * historical_GREEDY_HELLINGER + 0.10 * graph_plausibility`

No parameter fit, no optimizing the 0.90/0.10 weight, no choosing a
convenient subset. The graph score is in [0,1]; identical formula but
different connectivity is required for any discrimination. All original
GNPS reference replicates use the original fixed MAX Hellinger score.
Every original true label remains in the original metadata-only candidate
pool. All 4,398 positive hard queries / 2,254 formula blocks included.

## Gate, reproduction and negative outcomes

1. Original source SHA256, two-polarity failed V1, positive-only metadata
   V2 passed, full original 4,398-query score artifact all fixed.
2. Reproduce historical Hellinger MRR and fractional top1 for EVERY
   original query exactly (1e-12) before measuring graph score.
3. Predeclare one retrospective primary comparison:
   graph-conditioned MRR - historical Hellinger MRR.
   Bootstrap chemical FORMULA blocks 10,000 times, signflip formula
   blocks 50,000 times; require positive mean MRR, strictly positive
   95% lower CI and two-sided p<0.05.
4. Report total graph parse/chemical formula failures; fraction of
   fragments with ambiguous query mass overlap, true ranking switches,
   and the complete unchanged 4,398-query metrics.
5. If the criterion fails, publish an explicit scientifically NEGATIVE
   outcome, retain in DRAFT, and do not try 0.05/0.2/0.5 blending weights
   on this already evaluated source.

A positive retrospective result would be grounds for a **truly external
new physical laboratory test with completely frozen graph predictor,
source candidate spectra, and molecular labels inaccessible before
scoring**. It is NOT universal fragmentation physics or an official CASMI
leaderboard gain. The original empty negative-ion source remains excluded;
no inference to EAD or other activation methods.
