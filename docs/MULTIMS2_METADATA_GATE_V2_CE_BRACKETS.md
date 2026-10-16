# MultiMS2 v2: isolated exported collision-energy field syntax repair (pre-outcome)

**Original v1 result remains permanently FAILED at [run 38969490208](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38969490208)**. It read metadata from all 43,728 records but accepted zero CID records because its strict regex expected unbracketed `20.0`, `40.0`, `60.0` while the published MGF actually encodes nominal numeric energies as `[20.0]`, `[40.0]`, `[60.0]`. No peak mass/intensity or efficacy score was opened.

An independent **HEADER-ONLY** diagnostic [run 38969970105](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38969970105) confirmed exactly:
- 35,673 CID records: `[20.0]` = **9,538**, `[40.0]` = **13,089**, `[60.0]` = **13,046**.
- 8,055 EAD records at `[12.0]`, `[16.0]`, `[24.0]`. ALL EAD records remain **out of scope**.
- Upstream pinned raw MGF `data/multims2_spectra.mgf` Git blob SHA-1 `5ca1b01d1c6c5b2930658a6cd32ee2659070b49d`, byte SHA-256 `4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673`, 77,183,510 bytes, commit `659bd9b608408a75ecbb4b03295ed4c03ba52539`.

## Narrow v2 change — frozen before target scoring

The **sole scientific-parser change** is to accept **exactly** the three observed CID header strings `[20.0]`, `[40.0]` and `[60.0]` as representing the respective numeric published CE settings. Do NOT accept arbitrary brackets, uncertainty bounds, ranges, energy ramps, missing units in unrelated archives, or any different CE value. The upstream public acquisition publication explicitly states the instrument's 20, 40 and 60 eV CID settings. Source format is unambiguous in that context; eV is attributed to the publication/instrument protocol, not blindly inferred from the brackets.

All original v1 structure exclusions and deterministic duplicate-energy de-duplication remain unchanged: **full exact InChIKey**, discovery MCEBIO+, four prior MSnLib evaluations, MassBank eligible identities, ExpoLib eligible identities; CID/MS2, full formula, adduct, ion mode, >=5 *declared* positive fragment peaks (not intensity values). The minimum feasibility thresholds remain **100 triplets / 80 full keys / 60 formulas / 20 positive and 20 negative**, without revision.

Retain all original metadata. Validate the upstream SHA-256 and independently observed exact header counts **before** accepting any v2 result. Do not open spectral peaks if the corrected gate still fails. A passing v2 gate may authorize the already-frozen MultiMS2 scorer committed before diagnostic inspection (original scorer SHA `c062562c2934353dc8191cd4689a9fa2cab6b47b`; subsequent synthetic test SHA `9fae077a6776864c4aac47803cc55d9a0a9889f4`), but it must only rebind CE decoding and not modify the learned or fixed geometries.

No efficacy values were seen in preparing this v2 correction. This is a pre-outcome metadata fix, not a post-hoc scoring optimization.
