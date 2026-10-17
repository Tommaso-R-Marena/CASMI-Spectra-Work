# 2026 GNPS / MassSpecGym cross-library molecular retrieval — frozen protocol

**Protocol timestamped/committed BEFORE opening any harmonized v3 source peak values.**

Independent *library-provenance* validation of a fixed parameter-free support
score. Source: Luca Cappelletti, [Zenodo 20042904](https://doi.org/10.5281/zenodo.20042904),
harmonized top-128 MGF release published May 2026. File:
`combined-gnps-mass-spec-gym-npc-faithful.harmonized-subset.top128.mgf.zst`
MD5 `e2b792e384a9c9f577b17daad6c4247d`
and SHA256 `aee5a595960ef1284fe9977b6560a1ff771c07bb28c4a8e2e2ef47020b35724e`.
443,905 records; 389,560 GNPS and 54,345 MassSpecGym in source report.
Native fields include RDKit InChIKey, molecular formula, canonical SMILES,
and original library provenance; chemical IDs are computationally harmonized,
not curated experimental ground truths. This is NOT an independent lab claim,
nor a multi-energy or EAD experiment.

This project is independent of the earlier SPLISUM metadata-only failure:
SPLISUM exposed no reliable native label join in its first headers and aborted
on contradictory polarity metadata without scoring. Here an explicit native
InChIKey, formula, and provenance is REQUIRED for any score. No inherited
metadata fallback to source TITLE, chemical name or guessed identifiers.

## Stage 1 — metadata-only, no fragment-spectrum values parsed

Stream zstd decompressed MGF, scan **only header fields**. Require native
source provenance unambiguously GNPS or MassSpecGym, full 27-character InChIKey,
molecular formula, precursor m/z, charge/ion polarity consistent, and a
valid scan index. No fragment peaks read, parsed, counted or matched.
Source classification must be explicit in native metadata. Preserve all
source structure inventory, header key distribution, missingness and
label overlap in machine-readable reports before any spectral outcomes.

Reference library: all GNPS records. Query pool: MassSpecGym records ONLY.
For each same full native InChIKey and native formula, use both sources'
actual identity. The ranking candidate pool is defined by same formula,
polarity, and `max(.02 Da,20ppm)` precursor mass tolerance. If native
adduct is unambiguous in both, also require equality; if missing from any,
restrict by mass and polarity only and disclose adduct uncertainty. No
heldout peak position/intensity or similarity enters candidate selection.

A deterministic SHA256 salted rule chooses ONE query scan per
`(full InChIKey, formula, ion mode)` and, independently, a maximum of TWO
GNPS reference scans per `(identity, formula, ion mode)`. Thus there are no
query replicate pseudoreplications and no different sampling schemes by
scoring method. All identity-eligible and hard-negative queries are retained.
Pool contains ALL distinct structure competitors meeting metadata rules;
no nearest-neighbor retrieval or label-aware candidate pruning.

Feasibility gate: >= 200 same-identity cross-provider structures,
>=100 hard queries with >=2 different chemical candidate identities,
and >=20 hard queries for each polarity. If insufficient, publish
failure and STOP before opening ANY fragment values.

## Stage 2 — UNCHANGED fixed support-sensitive candidate ranker

For each true unknown query, rank full native structure candidates.
Reference spectra are GNPS; query spectra MassSpecGym only.
Use candidate reference-grid mass alignment within 0.01 Da, deterministic
one-to-one smallest error, retaining all unmatched intensity in norms.
Four frozen score types:
- standard full-norm matched cosine;
- L1-normalized Bhattacharyya affinity / Hellinger spectral similarity;
- normalized fragment overlap `Σ_matched min(p_i,q_i)`;
- **composite** = 0.5 cosine + 0.5 fragment-overlap.

For each candidate identity, aggregate up to 2 GNPS scans with their
MAX score, same rule for all methods, no fitted weights and no query
spectra used to filter reference observations. Tie handling fractional
top1 and average reciprocal rank, same as prior CASMI work.

Two co-primary MRR comparisons, corrected with Holm:
`composite > cosine` and `composite > Hellinger`. Require BOTH strictly
positive mean MRR differences, BOTH exact-molecule-block bootstrap 95%
lower bounds positive, and BOTH adjusted two-sided block-signflip p<.05.
Hard-negative queries ONLY (>=2 distinct candidate structures) are primary.
Report per-polarity MRR, top1, candidate group sizes, complete source
overlap, any identical query/reference fingerprints and all negative tests.
No tuning to rescue failures, no claim of multi-energy law or genuine
external-lab confirmation. The predeclared tests provide *first-look
cross-archive evidence* only: source libraries may share original recordings.

This release must remain DRAFT and must not be claimed as a full
scientific breakthrough unless an independent real laboratory replication,
chemical diversity controls, fully blinded challenge labels, and
appropriate downstream CASMI ranking gains also succeed.
