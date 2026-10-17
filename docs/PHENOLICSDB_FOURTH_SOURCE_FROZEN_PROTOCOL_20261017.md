# Independent PhenolicsDB: fourth-origin CID energy-transfer experiment

**Frozen before opening any PhenolicsDB query peak scores in this program.**
The hypothesis is a direct, unchanged external replay of the prior fixed
quarter-power predictor. Scientific success is NOT presumed.

## Independent source provenance

PhenolicsDB authentic-standard CID spectra collected by CooperstoneLab on
Agilent QTOF 6545/6546, independently of prior MultiMS2/MassBank/ExpoLib
measurements. Cite: `CooperstoneLab/PhenolicsDB`, upstream Git commit
`381693c8435a851710953f6b818f6008e2227377`, and follow original
CC-BY-4.0 attribution terms. Download ONLY exact commit paths:
- `inst/extdata/PhenolicsDB/mgf/PhenolicsDB_Positive.mgf` 137,354 bytes,
  Git blob SHA-1 `748ec688c8eeee952260189623275ff892aec1c9`
- `inst/extdata/PhenolicsDB/mgf/PhenolicsDB_Negative.mgf` 169,770 bytes,
  Git blob SHA-1 `19a4cb3a3afac23cc6637f03961aa1bf5ad5851e`

These are original, historically fixed versioned public MGF files. SHA1 is
Git *blob content identity* (git hash-object), not a provenance claim about
the data generator's physical instrument settings. Inputs are refused unless
they exactly match size and blob digest. Raw MGF is never committed here.

The earlier CID and EAD outcomes were already observed when this independent
fourth-source protocol was designed. However, these **specific PhenolicsDB
held-out target spectra are not opened until the protocol and all scorer
code is committed**. Thus independent-source first scoring, but NOT
preregistered before seeing the previous studies.

## Metadata-only gate — stop on failure

MGF `NAME` energy suffix, `INCHI` (containing full InChIKey), positive
or negative ion mode, MSLEVEL=2, PEPMASS. Parse no numeric target peak
locations or intensities. Require a unique, full chemical identity within a
mode with exactly a single record at each of **20,40,60,80 eV**; the four
precursor masses must agree within 0.01 Da. All complete groups selected;
no selection by affinity, fragment support or outcomes.

Gate thresholds (conservative for this small intentionally
narrow-chemistry library):
- at least 8 complete positive-ion groups;
- at least 8 complete negative-ion groups;
- at least 12 distinct FULL InChIKey structures across polarities;
- at least 40 held-out queries in total.

The group list is the full cohort without human-picked interesting molecules.
If the gate fails, publish coverage and stop BEFORE query peaks are opened.
This is a feasibility failure, regardless of whether the CI job succeeds.

## Scorer — exactly fixed methods, no optimization

For each `(full InChIKey, ion mode)` group, use **20 and 80 eV** as two
measured reference endpoints. Hold out **40 and 60 eV** as independent target
positions on that interval with energy coordinates `1/3` and `2/3`.
The nominal CE values are used as published; physical equivalence of
Agilent collision-energy calibration to other instruments is NOT assumed.
In this source the individual instrument/batch may differ between standards.

Build deterministic reference-only strict complete-link 0.01 Da m/z bins.
Query peaks NEVER enter reference prediction grids; query intensities,
including unmatched peak mass, are accounted for only in final scoring.
Exactly zero training or fitting on this new source.

Evaluate raw Bhattacharyya affinity at target spectra with:
1. **quarter-power** `alpha=0.25`, the previously selected CID model;
2. **Hellinger** `alpha=0.5`, the frozen principal comparator;
3. **support-only**, fixed fourth-power endpoint-exclusive suppression;
4. **linear** `alpha=1.0`, nongeometry control;
5. **best endpoint**, answer-aware *diagnostic* only, NOT a deployable method.

No identifying information is used to alter the predictor. These are
**within-compound held-out collision-energy spectra**, NOT candidate ranking
or actual unknown-molecule identification.

## Predeclared hypotheses and negative verdicts

Single primary comparison: mean affinity of fixed quarter minus fixed
Hellinger, over all selected query triplets. Count the two held-out energies
and both polarities of the same full molecule as ONE independent resampling
block: no pseudo-replication of 50 rows as 50 molecules. Use 10,000
molecular-identity-block bootstrap samples and 100,000 two-sided block
signflips (deterministic seeds). Require:
- positive mean difference;
- bootstrap 95% lower bound > 0;
- two-sided p < 0.05.

If any condition fails: **fixed-quarter superiority NOT confirmed**.
A CI-successful or high-affinity workflow is not enough. Positive and
negative ion polarity-specific comparisons are DESCRIPTIVE only; do not
count two polarities as separate labs. Any positive result is a small-cohort
external CID transfer observation; no claimed universal law or new EAD
mechanism. A negative result is equally important and retained permanently.

Secondary comparison quarter versus support-only tests geometry-specificity,
but is *posthoc motivationally linked to known support-only findings* and
cannot rescue a failed primary gate.

## Main limitations before any score is read

This is a small phenolics-enriched dataset of approximately 15 distinct
molecules across two ion modes, with correlated queries and multiple
nominal CE values from potentially differing 6545/6546 batches.
The source MGF does not provide verified formula/adduct/instrument
information per spectrum. Therefore resampling blocks are exact full
identities, never assumed independent formula groups. The source does not
permit matched CID/EAD causal claims, formal fragmentation kinetics,
strong heterogeneous-molecule generalization, or CASMI rank improvements.

The next-level evidence threshold is still an independently acquired,
substantially larger, diversified laboratory cohort, with genuinely
unknown-molecule exact-formula candidate ranking and well calibrated
instrument/energy metadata. Retain previous ExpoLib transfer failure,
EAD hybrid failure, and MassBank isomer MRR null.
