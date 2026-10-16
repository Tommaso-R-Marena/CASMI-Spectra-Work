# External-domain collision-energy geometry gate (MassBank 2026.03)

**Status:** label-free protocol frozen before reading efficacy outcomes from the external source. Dataset: MassBank official release 2026.03, DOI 10.5281/zenodo.19590880 (239.6 MB archive; MD5 `1de9018361f7ffc689aa327f92e97e0f`). Model-discovery exclusion: exact full InChIKeys present anywhere in MSnLib MCEBIO positive `20241003_mcebio_pos_ms2.json` (MD5 `806f33c9c60af74a0aa21fb66aeb84f3`), regardless of adduct/energy.

**Why feasibility first:** MassBank archives aggregate spectra from many instruments, laboratories and collision-energy conventions. Naively mixing normalized collision energy, voltages, eV, ramps or instrument conditions would produce an uninterpretable "geometry" result.

## Conservative label-free requirements

- Accept only explicit *one-number eV* collision-energy headers; reject bare numbers, multiple or stepped values, voltages, normalized collision energy, and ambiguous ranges.
- Accept only MS2 CID/HCD with full InChIKey, reported formula, precursor adduct, source contributor, ion mode, instrument type/model, and at least five positive peaks listed in `PK$NUM_PEAK`.
- Construct groups by full InChIKey + precursor adduct + instrument model + instrument type + polarity + fragmentation mode + contributor. Reject groups with any duplicate energy slot or inconsistent formula rather than selecting replicates post-hoc.
- For each group with three or more strictly ordered energies, choose interior query energy bracketed by adjacent lower/higher measured spectra. Do not read the `PK$PEAK` intensities during feasibility.
- Explicitly exclude **all** MCEBIO-positive discovery structures using the exact full InChIKey across energy/ionization settings.
- Require (after exclusions) at least **300 triplets, 200 structures, 150 formula blocks, 100 off-center triplets, 100 non-Orbitrap triplets, and 50 non-Orbitrap structures**. Fail closed and do not compute any efficacy outcome if **any** criterion fails.
- Report contributors, instrument types, triplet energies, input hashes, excluded discovery overlap, excluded ambiguous groups and formula conflicts.

If this gate passes, a **separate** frozen implementation should score the same unchanged `r=0.25` and Hellinger `r=0.5` predictors with target-blind m/z matching. The primary external outcome would be paired Δ Bhattacharyya affinity, formula-block bootstrap and block randomization, with contributor/instrument stratification and an additional check against the support-only control. Do not fit `r` or an energy warp.

If this gate fails, state exactly why. Do not lower the gate post-hoc and call the resulting score confirmatory.

**Important limitations:** Strict 'eV' spelling is conservative and may exclude real spectra with usable calibrated values; failure of this gate is a feasibility outcome only, not evidence against nonlinear geometry. A snapshot of a different repository is still not necessarily independent acquisition if records were redistributed; provenance overlaps and duplicate spectra require a separate check before asserting cross-instrument replication. Matching the full InChIKey to MCEBIO discovery prevents one major structural leak, but **does not** prove independence from all other MSnLib libraries or historical MassBank analyses.
