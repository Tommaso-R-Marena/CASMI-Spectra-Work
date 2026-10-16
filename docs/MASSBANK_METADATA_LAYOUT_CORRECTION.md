# MassBank 2026.03 metadata path-layout correction (not a scoring result)

## Observed failure
GitHub Actions run [38847649109](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38847649109) downloaded and MD5-verified the official 2026.03 release, then reported 139,240 MassBank record files, 0 parsable conservative metadata records, 0 eligible triplets, and a closed feasibility gate. No query spectra were scored.

The original parser required the literal directory segment `record` and extracted the contributor from the next segment. The official MassBank-data 2026.03 GitHub release tree instead organizes record files under top-level contributor directories such as `AAFC/MSBNK-AAFC-AC000001.txt` and `Eawag/...`. A ZIP may add an archive-root directory but is not required to add a `record/` directory. This packaging assumption could discard valid records before their collision-energy and identity metadata were examined.

## Correction and retained controls
Only contributor directory extraction changes: the immediate parent of a record file is the contributor, with an actual directory required. The path `archive/record/contributor/accession.txt` is also supported. All original scientific filters remain as frozen: explicitly numeric single-value `eV`; MS2 CID/HCD; full InChIKey, formula, adduct, ion mode, named instrument/contributor, at least five peaks; exact source/instrument/adduct/energy matching, duplicate slots rejected; discovery full-InChIKey exclusions; and the same six numerical minimum-coverage gates.

Unit tests exercise the official contributor path layout and confirm that `NCE`, `V`, `%`, bare energies, and ramped energies remain rejected. This is a correction of file-container interpretation, **not** an authorization to relax eligibility or optimize against benchmark outcomes.

## Mandatory next check
Run the corrected feasibility audit against the **identical MD5-pinned MassBank 2026.03 ZIP and MCEBIO discovery input**; verify actual ZIP members and report stagewise metadata attrition before drawing conclusions. A nonzero parsed-record count is **not** a gate pass: all predeclared coverage thresholds still apply. If any gate fails, do **not** run `scripts/benchmark_massbank_external_energy_geometry.py` or open any external spectral efficacy outcomes. Do not characterize the original zero as definitively explained until inspecting the archive and rerunning.

The original frozen audit commit `b23ceee0a0d67adfb04cccc9536debb312930805` and its failed feasibility artifact remain intact for provenance. A corrected workflow must pin a new implementation commit while identifying this as a post-release parser correction. Never rewrite the old experiment's reported outcome.
