# New fifth-origin actual molecule identification: SPLISUM metadata-only first

**Protocol frozen before accessing any new target/query peak m/z or intensities.**
The corpus is [SPLISUM 20eV v2](https://doi.org/10.5281/zenodo.19834896),
from Indiana University (Zenodo 19834896, 27 April 2026).
It is **single-energy MS/MS**. This is an *independent molecular-retrieval
test*, NOT a CID/EAD multi-energy interpolation replication.

## Motivation / no cherry-picking

Previous CASMI work showed improved pairwise spectral affinity does not
reliably improve exact-isomer candidate ranking. Previous independent
PhenolicsDB study showed support-only suppression can outperform quarter
interpolation. This fifth-origin test asks an actual molecular ID question:
does a fixed support-sensitive score change top-1/MRR vs standard cosine
and Hellinger on *externally supplied independent query vs target MGF*?

This design was written after other sources' results, but before opening
any SPLISUM peak spectra/outcomes in this program. If the native MGF lacks
the metadata necessary to establish molecular identity or precursor
comparability, the project MUST publish the failed metadata gate without
scoring, guessing molecule labels from titles, or cherry-picking another
evaluation subset.

### Immutable external source inputs

Download the following four ORIGINAL DOI 19834896 v2 Zenodo files and
verify the MD5 from the Zenodo listing before even parsing headers:
- `Query_INT_positive_20ev.mgf`  MD5 `3142dab3cd679611d0f0d50b95f248ca`;
- `Query_INT_negative_20ev.mgf`  MD5 `dce2f8f93746fb859c19ae798cf9306f`;
- `TARGET_20ev_positive.mgf`    MD5 `1675977e3d76d5f1b0ada7e5968ee95d`;
- `TARGET_20ev_negative.mgf`    MD5 `ab73102f4a2e008cc2db606f6c144ed5`.

Original source files are NOT committed into the repository. Attribution,
source file names, record checksum and DOI accompany all outputs.
Upstream licenses must be confirmed before any redistribution; we link
original public material and only archive derived metadata/results here.

## Phase 1: metadata-only audit, without reading ANY peak values

Traverse the four MGFs line-by-line, reading header fields exclusively.
Do not parse, convert, count, or score the numeric peak lines. Audit:
- distinct native header names and number of scans by file;
- explicit InChIKey (full 27-character), raw SMILES/other native
  identity fields, precursor mass and ion-mode fields;
- availability of a cross-library exact-native-identifier join;
- candidate set sizes under `max(0.02 Da, 20 ppm)` precursor mass and
  matching fixed ion polarity;
- target/query batch overlap/record IDs and sufficient independent
  molecules for a credible comparison.

Never invent canonical formula or identify molecules by unverified titles.
Molecular equivalence must be supported by actual native identifiers.
If molecule ids cannot be matched unambiguously across input collections,
report `INSUFFICIENT_NATIVE_IDENTITY_METADATA` and do not open peaks.
If target library and query collection share exact spectrum identifiers,
mark the source as potentially contaminated; do not claim independent
measurement validation.

When both target and query have full InChIKeys, use the exact key; otherwise
only literal matched native canonical annotation if unambiguous, describing
the limitation explicitly. Never use labels to filter score candidates.

Feasibility minima fixed before source exposure:
- at least 100 evaluable queries with true native-identity label present
  in fixed mass/polarity candidate pool;
- at least 50 distinct query structures in this population;
- at least 60 evaluable queries in pools of at least **2** structures.
- both polarities reported, but only 1 polarity passing individually
  is insufficient for any claim of broad two-polarity generalization.

No query spectra opened if the gate fails. Excluded unmatched queries
are reported, not hidden. If gate passes, *every* qualifying query and
*every* precursor/polarity-eligible target structure is ranked.

## Phase 2: frozen target-blind structure retrieval (to implement only after phase 1 confirms native metadata feasibility)

Three natural, fixed, parameter-free candidate scoring rules on
**unmodified native 20eV** reference spectra:
1. Matched standard cosine (positive intensities; all measured unmatched
   peaks stay in reference/query norms).
2. Bhattacharyya/Hellinger spectral affinity, with unit-L1 intensity
   vectors and query unmatched mass preserved.
3. Identity-fragment-overlap score, the matched sum of min(reference
   and query normalized intensities), also known as `1-TV` after
   accounting for unmatched mass.

A frozen support-sensitive candidate comparator is `0.5 * cosine
+ 0.5 * (1-TV)`. It is an explicitly simple fixed hypothesis, no
optimization on fifth-source outcomes. A candidate molecule with
multiple reference scans uses the MAX candidate score per compound,
the same rule for all baselines; report the coverage biases of this rule.
A consistent complete bipartite m/z matcher with tolerance 0.01 Da
is used for all methods, with a fixed deterministic tie rule, and
candidate reference bins must never depend on the heldout query.

Predeclare TWO co-primary comparisons for rank MRR:
`support_composite - standard_cosine` and
`support_composite - fixed_Hellinger`, Holm-adjusted block permutation
p<0.05 and chemical-ID-block bootstrap lower 95% CI >0 for BOTH.
Use query full identity as the resampling block, regardless of multiple
spectra, with fixed random seed. Report fractional top-1 and failures.
A positive score or green CI alone is NOT success.

**Scope restriction:** single-energy query/reference matching cannot
by itself validate energy interpolation, electron activation, or a
physical fragmentation law. Even a genuine external retrieval gain
needs independent second-site replication and competition-specific
candidate availability assessment before declaring a full breakthrough.

## Fail-closed decision

Phase 1 and Phase 2 are distinct, auditable GitHub Actions.
Phase 2 cannot be unlocked by positive results elsewhere, by invented
identity mapping, or by debugging based on held-out query peaks.
A metadata-gate failure is an important and final scientific result
for this fifth-source pilot under these rules.
