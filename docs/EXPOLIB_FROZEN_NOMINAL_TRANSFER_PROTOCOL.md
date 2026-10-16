# Frozen ExpoLib v1 nominal collision-energy interpolation transfer (2026-10-16)

**Design frozen before opening ANY ExpoLib held-out spectrum prediction accuracy.** Previous MSnLib and MassBank results have been seen. This independent acquisition-source follow-up evaluates an **unqualified nominal CE coordinate** and cannot alone verify a physical energy law.

## Immutable inputs

- ExpoLib1.0 v1 Zenodo record 18186811, ESI+ MSP MD5 `c08dcc4dc3c11eef188dff1eb1f2bc1d`, SHA-256 `aa838492fe11bd068819d3bc872541d7ec727217440dd5d2595af8985fb56959`, ESI− MD5 `3a7544d71bf9ee3564243095b3134d3d`, SHA-256 `b26e11b25c4ee5d24507e5aa9b525540096a4347cfdfbce94d230e4e858fa838`.
- Independently frozen metadata-only gate run [38964190641](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38964190641). The exact `feasibility.json` SHA-256 is `fbf4a6ab01bdf536dad09edc6dfcb100ee133ada0f4cabddb1d4d6b956f7af42`, and `nominal_energy_triplets.jsonl` SHA-256 is `095340d967b177077ce517da7aa332795ce691899cc1320563cce07a76a68560`.
- Fixed metadata results: 1,125 triplets, 122 distinct full InChIKeys, 114 molecular-formula blocks, 103 off-center triplets, 538 negative and 587 positive. All four coverage thresholds passed. Original MCEBIO+ discovery full InChIKeys were already excluded. CE duplicate slots resolved by earliest record position independent of intensities.
- The physical-eV validation gate remains **FALSE**. Labels must explicitly say **exploratory nominal CE transfer** even if statistics are positive.

## Fixed evaluators

For each query at nominal CE `e` bracketed by adjacent measurements `e0,e1` from the *same molecule, adduct and polarity*, compute `t=(e-e0)/(e1-e0)` with no retuning. The same labeled specimen and processing library may be reused at other CE slots (within-sample dependence addressed by formula-block resampling).

Using only lower/higher endpoint spectra, construct the existing fixed **0.01 Da reference-only binning** and normalize all positive peak intensities. Query intensities are normalized across the full query spectrum; unmatched target mass is retained in the denominator and earns zero affinity. No target spectrum may influence predicted reference m/z bins or method parameters. Original r=.25 selected on MCEBIO-positive MSnLib; compare **frozen r=.25** against fixed physical-fraction r=.5 Hellinger, support-only power 4, and answer-aware globally optimal Hellinger scalar chord diagnostic (nondeployable). Include linear r=1 as a descriptive control.

No selection or reweighting based on spectral outcomes; all 1,125 predeclared eligible triplets must be scored or the overall run fails. Require finite 0≤affinity≤1, positive masses and no loss of query mass in normalization. Record exact input hashes and audited metadata. Preserve all per-query scores.

## Outcome analysis

Primary **new-source nominal-coordinate** contrast is the paired mean `BC(alpha=.25,heldout)-BC(alpha=.5,heldout)` across all 1,125 predeclared query triplets. Report formula-cluster whole-block bootstrap 95% interval (10,000 draws) and two-sided formula-block sign flips (50,000 draws) with RNG seed 20261016. Call a result directionally supported only if paired Δ>0, block-CI lower>0 and p<.05. Report positive and negative ion subgroups, offcenter subgroup, every method mean, wins/losses/ties and unmatched intensity fractions regardless of primary direction.

As specificity falsifications compare r=.25 with **support-only suppression** and the **nondeployable optimal Hellinger chord point**. If Hellinger oracle wins, report that even if fixed Hellinger loses. Do not fit exponents, intensity normalization, m/z tolerance, support threshold, energy warp, CE-unit conversion, or per-chemical weights to this dataset.

## Evidence boundary

1. A positive result would provide independent *acquisition-source* transfer for the predefined nominal-coordinate interpolation task (subject to remaining compound-provenance and library-overlap audits), **not** universal physical-energy physics.
2. A negative result would be a crucial limitation, not a broken workflow, and MUST be published.
3. The new-source data are not a new CASMI identification ranking study. The prior isomer-ranking test failed to establish an MRR advantage. Reproducing affinity is not equivalent to demonstrating correct molecular identification.
4. If extraction, checksums, duplicate slots or full-query scoring fail, preserve this initial attempt and all failure reasons. Any subsequent protocol repair must be new-version, documented and strictly separate from the original experiment.
