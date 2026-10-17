# Prospective external laboratory breakthrough-validation contract

## Frozen research method

**Reference implementation frozen at Git commit**
`64aced60540b5d57e191ff0ead73b445c71e1732`.
Entrypoints:
`src/qfd/graph_fragment_signatures.py` and
`scripts/benchmark_graph_conditioned_fragments_4398.py`.
Reproducibility anchor: completed Actions run [39058512753](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/39058512753),
source SHA256 `aee5a595960ef1284fe9977b6560a1ff771c07bb28c4a8e2e2ef47020b35724e`.

**Do not alter** the fixed graph model: up to 24 nonring single-bond
cuts, 2-atom minimum component, deterministic heteroatom prioritization,
four exact ion-mass shifts (−H, 0, +H, +proton), 0.01 Da matching,
candidate-pool 1−df/K rarity, zero chemistry fallback without excluding
candidates, and score 0.90 historical Hellinger + 0.10 graph.
Reference spectra Hellinger matching remains the ORIGINAL nearest-mass
one-to-one baseline, not the separately falsified optimal assignment.

## What justifies the next experiment

On the already-seen GNPS→MassSpecGym 4,398 positive-ion
same-formula hard queries, the fixed graph model had
MRR **0.813106** versus Hellinger **0.811962**, Δ **+0.001144**,
native-formula-cluster 95% CI **[+0.000123,+0.002258]** and
nominal retrospective p **0.032019**, but fractional top1 changed from
**0.707398 DOWN to 0.707360**. The model passed a single nominal
within-variant test yet came after many exploratory prior variants.
Thus this is **NOT confirmatory evidence** and does not establish a
new physical law, general scientific structure elucidation, EAD gains,
or a breakthrough.

The 2026 harmonized source can overlap underlying original GNPS
acquisitions; a different archive label alone does NOT prove
independent laboratory data.

## Independent acquisition constraints, before labels are opened

Seek a *new* laboratory/site/instrument-acquired experimental LC-MS/MS
source (not a simulated or re-exported GNPS/MassBank/MoNA spectrum).
Records must include:
- raw acquisition provenance (facility, instrument/scan, date, file ID,
  collisional activation, collision-energy calibration or honest unknown);
- precursor m/z, charge/adduct, ion polarity, actual numeric peak list,
  exact full InChIKey and native SMILES/formula in an independently
  answer-sealed manifest;
- source license and immutable archive checksum;
- candidate reference spectra for all same-formula/adduct/ion-mode
  compounds, acquired independently of the target query scans;
- exact spectral/USI/native scan identity checks against ALL previous
  GNPS, MassSpecGym, MultiMS2, PhenolicsDB and MSnLib development
  sources. Full spectrum equality, near-duplicate fingerprints,
  raw file aliases and source record overlap must be checked; matching
  original chemical structures across labs is ALLOWED, but not
  reusing the same underlying physical scan.

Intake must first inspect **metadata only**, with peak intensities and
correct chemical labels isolated in separate integrity-pinned locations.
A named dataset is NOT automatically independent. If provenance or
chemical IDs cannot establish actual independent acquisition,
fail the scientific gate before scoring.

## Predeclared eligibility (before seeing target peaks)

- >= 500 unique query structures with independently confirmed labels,
  >= 200 chemical formulas, >= 200 different formula/adduct hard
  competition pools with >= 2 structures per pool;
- >= 500 eligible hard queries, with all qualification based on metadata
  alone; no selection by outcomes or intensity;
- exact full InChIKeys (never InChIKey14), formula, polarity,
  precursor/adduct, acquisition provenance and reference candidate
  availability required for every included query;
- both positive and negative polarity must be reported, including
  zero/insufficient counts. A separately approved positive-only gate
  cannot be retroactively described as validating both polarities;
- all source-laboratory clusters reported and >= 2 **distinct** new
  facilities for a claim of robust cross-laboratory generalization.

For the independent test, rank EVERY original candidate. Compare
fixed graph-conditioned score against (a) original Hellinger and
(b) a strong established current laboratory library-search baseline
whose specification is frozen with the acquisition. Require BOTH MRR
and fractional top1 improvement over (a), nonregression against (b),
all structure/formula cluster bootstrap intervals for the MRR/top1
improvements with strictly positive 95% lower endpoints, and a
familywise Holm-adjusted two-sided p<0.05 on all registered primary
comparisons. Laboratory cluster variability and per-instrument
heterogeneity must be separately reported; two labs are not enough
for precise lab-population uncertainty.

**Failure outcome:** any unchanged/negative effect, metadata
contamination, nonindependent acquisition, top1 non-improvement,
or insufficient laboratory/chemical cohort means
`NO_CONFIRMED_BREAKTHROUGH`. Publish all failures and raw-compatible
scripts/artifacts, never retune this specific holdout, and never merge
scientific claims based solely on GitHub CI success.

This is the next high-value empirical gate, not evidence already
obtained. It cannot be executed on genuinely unseen laboratory data
until that independent source is actually secured.
