# EAD cross-modality persistent-fragment hybrid: prospective-in-modality lockbox

**Declared 2026-10-16 before inspecting EAD 16-eV query peaks.** This tests a new fragmentation modality in the SAME MultiMS2 source and instrument, following observation of CID results. It is NOT new independent-laboratory validation, and not untouched-source validation.

## Mathematical predictor, frozen now
Normalized low/high endpoint spectra a,b and t=(16−12)/(24−12)=1/3, reference-only 0.01 Da grid:

For shared (a_i>0,b_i>0) bins, U_i = ((1−t)sqrt(a_i)+t sqrt(b_i))^2.

For any endpoint-exclusive bin, U_i=(1−t)^4 a_i+t^4 b_i.

Prediction P_i=U_i/(sum U_j). No free fitted parameters and no smoothing. Exactly preserves both endpoints and reversal symmetry; reduces to quarter-power when endpoint supports are disjoint. Unlike original support-only it uses Hellinger, not linear, mixing of persistent fragments.

## Source and pre-score metadata gate
Raw upstream data: Zamboni Lab MultiMS2, commit 659bd9b608408a75ecbb4b03295ed4c03ba52539, MGF Git blob 5ca1b01d1c6c5b2930658a6cd32ee2659070b49d, 77,183,510 bytes, full SHA256 4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673. Existing independent header-only source inventory found bracketed EAD energy labels [12.0], [16.0], [24.0], counts 2484/2760/2811.

Only header metadata, never numeric m/z intensity rows, may be read before the feasibility verdict. Require EAD, MS2, exact bracketed energy, full InChIKey, molecular formula, adduct, polarity, >=5 *declared* peaks. Group on full InChIKey + formula + adduct + polarity + fragmentation + source dataset. Select FIRST file-position record for each 12/16/24-eV energy; no intensity QC or cherry-picking. Exclude exact full structure identities from MCEBIO+ discovery, all four previously scored MSnLib cohorts, original MassBank 2138-eligible cohort, and original ExpoLib 1125-eligible cohort. Retain EAD identities that also appeared in CID for the primary transfer test; report the pre-defined disjoint-identity sensitivity.

Minimum eligible coverage: >=30 triple queries, >=20 full structures, >=15 formula blocks, >=5 positive, >=5 negative. If the gate fails, NO EAD target peaks or scores may be opened; archive failed gate.

## Frozen scored methods and metrics

1. Persistent Hellinger + transient fourth-power **hybrid** (new).
2. Quarter-power (alpha 0.25).
3. Support-only fourth-power exclusive, linear shared.
4. Fixed Hellinger (alpha 0.5).
5. Linear mixture.
6. Answer-aware Hellinger-path oracle (nondeployable; only diagnostic).

Build each predictor using endpoint spectra ONLY and reference-only 0.01 Da binning; one-to-one align withheld 16-eV peaks at evaluation time, with unmatched observed mass retained in denominator. No hyperparameter tuning or outcome-dependent exclusions.

Primary inference consists of hybrid−quarter AND hybrid−support on the entire population. A joint positive result requires both positive mean differences, both formula-cluster 5,000-bootstrap confidence lower bounds >0, and Holm-corrected p<0.05 using 20,000 formula-block sign flips. Fixed seed 20261016. Additional source/polarity/disjoint-CID summaries descriptive, unless predeclared. Every row, source digest, gate digest, source code SHA, failure and test outcome preserved.

A positive result would support cross-fragmentation prediction within the existing source, not chemical-identification gains, independence across labs, physical-law geometry, or a universal exponent. A negative result falsifies this particular fixed hybrid. Do not refit on EAD to erase it.
