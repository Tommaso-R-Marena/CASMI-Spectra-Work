# Prospective external MassBank scoring protocol, conditional on frozen feasibility

**Status:** design frozen before external MassBank efficacy results are opened. The MassBank metadata-only gate is in `docs/MASSBANK_EXTERNAL_ENERGY_GEOMETRY_GATE.md` and the exact official 2026.03 archive is MD5-pinned. If the prior gate is false, stop without scoring.

## Input and identity

Reproduce the label-free triplet manifest from the unmodified conservative MassBank metadata parser. Explicitly exclude every full InChIKey occurring in the pinned MCEBIO-positive geometry-discovery library regardless of ion/adduct or energy. All three records in each triplet share exact full InChIKey, contributor, instrument model/type, ionization polarity, precursor adduct, fragmentation method (CID/HCD), and an explicitly numeric eV energy unit. Only interior energies bracketed by adjacent measured lower/higher energies are scored. No experiment-specific intensity filtering or formula-specific tuning.

Use `PK$PEAK` m/z and original raw intensity (second numeric column) exactly, keeping positive finite peaks; normalize each spectrum by the full positive intensity. Construct the strict 0.01 Da reference grid from lower+higher reference spectra alone, then one-to-one mass-match the query for scoring with unmatched query intensity retained in the denominator. Spectral prediction must not read query peak locations/intensities.

## Frozen predictors

1. `r=.25`: already chosen on **MCEBIO+ discovery formulas**; never refit.
2. `r=.5`: normalized Hellinger chord with fixed physical-energy fractional t.
3. Support-only suppression with shared linear peaks and (1-t)^4/t^4 endpoint-exclusive weights.
4. Answer-aware optimal Hellinger-chord point (nondeployable diagnostic; per-query t oracle).
5. Convex-mixture answer-aware oracle (nondeployable; optional diagnostic).

## External tests and failure rules

- **Primary predeclared:** paired r=.25 minus fixed Hellinger chord on ALL label-free eligible MassBank triplets. Success requires Δmean>0, formula-block bootstrap 95% lower>0 and two-sided formula-block sign-flip p<0.05.
- **Independent instrumentation stress:** report non-Orbitrap subgroup separately, requiring ≥100 query triplets and ≥50 unique structures as already frozen. No pooled-only positive claim if non-Orbitrap subgroup contradicts the effect.
- **Specificity controls:** r=.25 minus support-only, and r=.25 minus answer-aware Hellinger chord oracle. These tests can falsify geometry-specific interpretations even if fixed Hellinger is worse.
- Perform 3,000 molecular-formula-block bootstrap draws and 50,000 randomized block flips per contrast, fixed RNG seed 20261016. Report individual contributor/instrument cohorts descriptively with n and formula counts; do not claim independence between remeasured compounds across contributors.
- Report fraction of total query intensity unmatched to reference-only bins; fail/flag if it dominates.
- Do not lower any coverage gate, retune m/z tolerance, change intensity normalization, select adducts by observed scores, or pool energies across instrument types to rescue a null result.

## Boundaries

MassBank 2026.03 is a separate repository/release but does not automatically imply independent acquisition from historical MSnLib libraries. Full-InChIKey exclusion protects against direct reuse of discovery compounds; other data-redistribution overlap and instrument harmonization still need scrutiny before a physical-law claim. This new source is strictly a different-data predictive test, *not* automatic proof of genuinely universal geometry or improved CASMI candidate ranking.
