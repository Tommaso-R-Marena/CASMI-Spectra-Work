# Positive-only V2 amendment — after failed V1 headers, BEFORE any spectra scored

V1 two-polarity GNPS↔MassSpecGym metadata audit was honestly FAILED:
- original run 39055589643; source SHA256
  `aee5a595960ef1284fe9977b6560a1ff771c07bb28c4a8e2e2ef47020b35724e`.
- It found **4,398** positive-ion hard same-formula identification queries,
  zero eligible negative-ion hard queries, **6,237** cross-library
  matched full InChIKeys. It opened **zero fragment peak values**.

This is a **separately versioned positive-ion-only V2 protocol declared
after HEADER coverage but BEFORE any 20eV GNPS/MassSpecGym spectral scores**.
The V1 source-level two-polarity hypothesis remains permanently FAILED.

## Fixed cohort, zero outcome-driven selection

Exactly reuse ALL 4,398 previously selected hard positive-ion queries and
ALL previously frozen GNPS reference scan indices from V1.
Never add/delete candidates, change mass tolerances, change duplicate
handling or alter the preselected one-query-per-native-identity rule.
The original negative-ion zero is retained in V2 metadata. Keep V1
MD5/SHA-256 and the V1 feasibility artifact digest in every V2 result.

V2 separate minima:
- >=3,000 eligible positive hard queries;
- >=3,000 distinct full InChIKey query identities;
- >=500 molecular formulas represented.
All 4,398 original eligible queries must be preserved even if more than the
minimum. If counts fail, do NOT open spectra.

The V2 scoring methods, tolerance, two co-primary hypotheses and statistical
procedures are **IDENTICAL** to original GNPS/MassSpecGym V1 protocol:
cosine, Hellinger, fragment-overlap, 0.5 cosine +0.5 overlap; paired
chemical-identity block bootstrap and Holm-adjusted signflips.

### Interpretive boundary

Any significant positive-only outcome is an exploratory **same-formula
GNPS→MassSpecGym archive retrieval** finding, not prospective two-polarity
confirmation, independent laboratory acquisition, or CID/EAD interpolation.
Computationally harmonized identity labels are not independently curated
ground truth. Never rewrite V1 as a pass, or claim negative-ion efficacy.

This amendment is made because the original native metadata demonstrates
a feasible large positive-only task, not because any predictor succeeded.
