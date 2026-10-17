# Dual-modality CID/EAD same-formula identification: exploratory test

**Status:** new experiment implemented 2026-10-17. The earlier CID/EAD transfer
affinity outcomes were **already observed** when this study was conceived.
Consequently **any result here is retrospective/exploratory, even though the
new identification endpoint and analysis rules are fixed in code before this
specific Actions run**. This does NOT count as fourth-lab prospective evidence.

## Why this changes the scientific question

Previously the quarter-power transfer advantage in some CID sources failed
to improve 252-query real MassBank isomer MRR, and quarter-power was worse than
Hellinger in EAD. An affinity gain by itself does not demonstrate improved
chemical identification. A separate question is whether adding measured EAD
information to measured CID information enables more reliable *same-formula*
structure discrimination.

We evaluate the acquisition modes side by side on **exact full-InChIKey** matched
MultiMS2 positive-ion spectra from the same original open, checksum-verified
77MB source. CID uses 20/40/60-eV brackets (40 withheld); EAD uses
12/16/24-eV brackets (16 withheld). This comparison does not imply these
nominal energies are physically comparable across modalities.

## Stage 1 — metadata-only, fail closed

Re-use originally frozen CID manifest from run 38970318925 (SHA256
5345f532ded9baa4134d38b840a6e05a1b1f85b3cf19923569ea80d96f46289d)
and complete positive-ion EAD V2 manifest from run 38985767245, after the V1
all-polarity gate failed due to only one eligible negative-ion triplet. No
peak positions/intensities or existing score rows are consulted to select the
cohort. Reuse of a measured spectrum at two different candidate positions or
duplicate full key is fatal.

Join on **full InChIKey, molecular formula, adduct, polarity, dataset ID**.
Only formula/adduct/polarity/dataset candidate sets with at least two distinct
full chemical identities qualify as actual same-formula identification tasks;
all singletons are excluded without looking at their scores. Require at least
eight isomer groups and twenty eligible target structures. If the gate fails,
publish coverage and **do not open any target peaks**. Report the failure as a
scientific negative feasibility result, irrespective of CI success.

## Stage 2 — reference-only predictions, frozen ranking

For each candidate, form separate reference-only bins for its low/high CID
and EAD spectra, with complete-link 0.01-Da tolerance. Apply the pre-existing
parameter-free power-mean predictor to its *reference* intensities.

At scoring time only, match held-out target CID/EAD peaks to each candidate's
fixed prediction grid, one-to-one, without renormalizing unmatched target mass.
Every candidate in the entire metadata-eligible same-formula group is ranked;
do not filter hard candidates based on scores. For each withheld compound query,
the comparison is a legitimate reference-library energy-interpolation task, **not
de novo structure elucidation**, because both candidate reference energies were
experimentally measured for that compound.

Five predeclared retrieval methods:
- **cid_h:** CID Hellinger-chord (alpha=0.5)
- **ead_h:** EAD Hellinger-chord (alpha=0.5)
- **fused_hh:** equal-weight mean of CID-Hellinger and EAD-Hellinger
- **cid_q:** CID fixed quarter-power (alpha=0.25)
- **fused_qh:** equal-weight CID-quarter and EAD-Hellinger

No fitted thresholds, train/validation mixing, or candidate-specific fusion
weights. Deterministic reciprocal average rank accounts for all near-ties
(1e-12); top-1 is fractional for a tie at rank 1. No top-K metric can count
the correct identity uniquely when a tie exists.

## Hypotheses and falsification

Two **co-primary** questions compare fused_hh MRR with CID-only Hellinger
and EAD-only Hellinger. A joint positive verdict requires both paired MRR
deltas positive, both formula-block bootstrap 95% lower endpoints positive,
and both Holm-corrected two-sided molecular-formula-block signflip p<0.05.
This is within-dataset formula uncertainty, not lab-level inference.

One **secondary, nonconfirmatory** comparison tests fused_qh versus fused_hh.
That test cannot rescue a failed primary gate, and because the CID-versus-EAD
reversal was already observed, it is especially vulnerable to hindsight bias.

Outputs are machine-readable all-query metrics, MRR/top1 per comparator,
confidence intervals, paired win/loss/ties and an explicit scientific
success/failure flag. Workflow retains raw source checksum and manifest hashes
and publishes both metadata-only and optional scored artifacts.

## Interpretation rules

- Gate FAIL: no suitable identification benchmark under the frozen exact-formula
  rules. No claims about identification efficacy.
- Gate PASS and fused_hh primary FAIL: no evidence for this equal-weight fusion
  improving *both* single-modality identification baselines.
- Gate PASS and joint primary PASS: new within-instrument evidence of
  complementary CID+EAD retrieval under exact-formula competition. It would be
  worth further independent validation, **not a universal identification law**.
- A successful secondary fixed CID-quarter/EAD-Hellinger contrast is a
  hypothesis for a future untouched fourth source, not validation here.

**Required next major breakthrough gate:** an independently acquired fourth
lab/instrument dataset, actually calibrated energies, new same-formula
candidates and enough independent chemical blocks. Freeze the complete
modality-aware scoring/identification procedure before viewing those outcomes.
Never overwrite earlier EAD V1 failure, V2 hybrid failure, ExpoLib negative
transfer, or MassBank identification null results.
