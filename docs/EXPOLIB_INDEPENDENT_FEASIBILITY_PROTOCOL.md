# ExpoLib separate-acquisition feasibility and leakage gate (locked 2026-10-16)

**New independent source:** University of Vienna / Global Exposomics and Biomonitoring Laboratory, ExpoLib1.0 first immutable Zenodo release `18186811`, two small `.msp` files pinned by MD5:
- `ExpoLib1.0_ESI+.msp` md5 `c08dcc4dc3c11eef188dff1eb1f2bc1d`
- `ExpoLib1.0_ESI-.msp` md5 `3a7544d71bf9ee3564243095b3134d3d`

**Rationale:** As a newly examined acquisition source, this can test whether fixed alpha=.25 from MCEBIO+ survives new instruments and chemistry instead of further fitting on reused MassBank outcomes. The source reports 13 nominal collision energies but the exported MSP headers provide `Collision_energy: [40.0]` without explicit unit. Therefore **the physical-eV/independent-acquisition confirmation gate is UNKNOWN/FAILED pending documentation**; numeric CE positions alone cannot justify a universal energy-law claim.

## Information barrier: metadata-only first

1. Download and MD5-verify original immutable small MSP files. Parse only record headers and `Num Peaks` count, never load m/z or spectral intensities in the feasibility stage.
2. Accept uniquely identified full InChIKey, formula, adduct, single positive numeric collision-energy in brackets, MS2 and CID/HCD, declared ion mode, and at least five declared peaks. Reject ramps, multi-valued collision energies, nonexistent or invalid identities and mixed ambiguous metadata.
3. Group by full InChIKey, adduct, ion mode, fragmentation method and file source. A duplicate energy slot is resolved by **earliest file position**, independent of results; record how many repeated slots were discarded. Do not select spectra based on observed intensity or spectral agreement.
4. For at least three ordered energies, take each interior energy bracketed by adjacent observed energies. Exclude all full InChIKeys from pinned MCEBIO-positive `20241003_mcebio_pos_ms2.json` (MD5 `806f33c9c60af74a0aa21fb66aeb84f3`) **before** opening any efficacy outcome. Report all record, unique structure, collision energy, polarity, replicate and exclusion counts.
5. Before independent validation, require **≥100 distinct interior triplets, ≥50 unique structures, ≥30 formula blocks and ≥30 off-center triplets** after discovery exclusions and source audit. This is a newly frozen ExpoLib criterion, not retrospectively taken from results.
6. Strictly distinguish `metadata_feasibility_pass` from `physical_eV_unit_validation_pass`. The latter remains false pending independent verification that nominal `Collision_energy` is comparable linearly within a source. Do not silently treat the exported bare numbers as explicitly eV.

## Conditional scoring only if metadata gate passes

If metadata feasibility passes, a **separately committed immutable scoring protocol** may test fixed alpha=.25 versus .5 Hellinger with 0.01 Da target-independent reference binning, unmatched-intensity mass conservation, formula-block uncertainty, support-only control and answer-aware optimal Hellinger oracle. No tuning. If CE unit cannot be independently verified, such results must be explicitly called **nominal-coordinate exploratory cross-source transfer**, never a physical-unit cross-instrument replication. All discovered failures and source artifacts must be retained.

No efficacy values were consulted in designing this gate.
