# Frozen exploratory test: formula-conditional reference-only fragment rarity

Date: 2026-10-17. This benchmark is derived from research/gnps-massspecgym-cross-library-lockbox-20261017 at 6be26208cbe9104385feefd29af5380bca4b61de. The original 4,398-query source and model outcomes are already public (GitHub Actions 39056094950), with the original support-composite worse than Hellinger. This is a retrospective exploratory method test, **not** independently blinded or external-laboratory confirmatory replication.

## Scientific hypothesis

The most common GNPS fragment masses are shared by same-formula competitors, so they may be strong measures of global spectrum similarity while discriminating chemical isomers poorly. A query-target-label-free molecule-document frequency weighting of GNPS fragments may improve actual correct-InChIKey ranking.

## Immutable source, strict frozen controls

Full originally published harmonized GNPS/MassSpecGym source file SHA256: aee5a595960ef1284fe9977b6560a1ff771c07bb28c4a8e2e2ef47020b35724e; DOI 10.5281/zenodo.20042904.

Retain the original full source selection, failed two-polarity native header gate, pre-scoring positive-only V2 rule, 4,398 same-formula hard positive queries, 2,254 formulas, target structures, candidate pools, molecular identity, ion modes, precursor mass matching, fragment match tolerance 0.01 Da, full-intensity unmatched mass penalty, and maximum replicate score within a candidate. The original baseline is rerun as a control, verifying its previously negative primary finding and MRR numeric values before any new report.

For each query candidate pool, group GNPS reference spectra by exact full InChIKey; union reference mass positions for each candidate identity, regardless of replicates. Let N be the number of structures and df(m) the number of distinct candidate molecular reference inventories with at least one peak in a 0.01-Da window of m. The reference-only peak weight is w(m)=1+ln((N+1)/(df(m)+1)). Query intensities and ground-truth labels never enter the document frequencies. Unknown query-only masses receive the maximum weight and remain in the full L1 normalization denominator.

For every query-reference pair, multiply all intensities by these same reference-only mass weights, L1-normalize full spectra and evaluate their Bhattacharyya similarity using exactly the original fixed one-to-one smallest-mass-error matching. Define fixed rarity blend = 0.5 Hellinger baseline + 0.5 rarity-weighted Hellinger; same MAX-over-GNPS scans rule applies. No parameter optimization, no energy interpolation, no learned weights.

**Single primary test:** mean MRR(rarity blend) minus mean MRR(unweighted Hellinger) on all predeclared eligible queries. Use chemical **formula as clustering unit**, rather than falsely treating queries competing in the same isomer pool as independent: 5,000 formula-block bootstrap replicates, 50,000 whole-formula paired signflips, two-sided p < 0.05. The exploratory improvement criterion is a positive mean MRR difference, positive 95% block-bootstrap lower bound, and p < 0.05, all together. Report fractional top-1 and win/loss/ties. Raw rarity and blend versus cosine are exploratory diagnostics only, not alternate success routes. Publish failures and nulls intact.

## External replication needed before stronger inference

Any positive outcome remains hypothesis-generating: chemical identities are computationally harmonized, not fully curator-confirmed; archive provenance does not prove independent physical acquisition; only positive ions pass eligibility; the source has already been evaluated once. A genuinely new prospectively locked independent lab source with exact structures, acquisition audits, independent annotators and CASMI-style blinded candidate ranking is needed before claiming a robust discovery. Do not reweight or rerun with outcome-dependent adjustments to rescue a negative test. This is not an EAD or collision-energy fragmentation-law validation.
