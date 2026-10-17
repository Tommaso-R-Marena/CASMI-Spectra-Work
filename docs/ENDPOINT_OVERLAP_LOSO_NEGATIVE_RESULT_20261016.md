# Retrospective endpoint-overlap selector — source-heldout evaluation

**Status: NEGATIVE/mixed exploratory result; outcomes known before developing selector.** Completed GitHub Actions workflow [38986325643](https://github.com/Tommaso-R-Marena/CASMI-Spectra-Work/actions/runs/38986325643). Do not call prospective validation or a general physical-law discovery.

Three distinct acquisition origins: MassBank, ExpoLib, MultiMS2. BOTH MultiMS2 CID and EAD modalities are held out together in the MultiMS2 origin. Frozen original score and endpoint descriptor artifacts re-used. 5,738 triplets in total (2,138 MassBank; 1,125 ExpoLib; 1,781 MultiMS2 CID; 694 EAD positive). The EAD original V1 gate failure is retained separately.

## Model and outcome

Exploratory deterministic selector chooses quarter-power when endpoint Bhattacharyya overlap s is below threshold theta, else Hellinger. Candidate theta grid 0,.05,...,.95,1.0000001. Tune threshold by maximizing equal-origin-weight average mean advantage over fixed Hellinger on the **other** origins, with two MultiMS2 modality means weighted equally within their shared origin. Evaluate only original heldout source scores, no target read when making predictions.

| Heldout source | Train-selected theta | Quarter share | Policy minus Hellinger | 95% formula-block bootstrap CI | Policy minus support-only |
|---|---:|---:|---:|---|---:|
| MassBank | 0.55 | 37.0% | **+0.004696** | [+0.003706,+0.005699] | +0.006839 |
| ExpoLib | 0.65 | 11.8% | **−0.001386** | [−0.002492,−0.000405] | +0.002405 |
| MultiMS2 CID | 0.80 | 99.8% | **+0.064517** | [+0.062330,+0.066830] | +0.001257 |
| MultiMS2 EAD positive | 0.80 | 52.9% | **−0.011326** | [−0.012840,−0.009777] | +0.006628 |

The source-heldout selector is competitive with fixed quarter on CID, improves MassBank vs Hellinger, and reduces the severe EAD losses of quarter. Nevertheless, it significantly underperforms Hellinger on ExpoLib and EAD. It does NOT solve generalization across the measured origins.

Because this analysis was conceived after the exact four outcomes were known, even origin-heldout fold testing is **retrospective method search**, not independent external confirmation. Three distinct origins cannot support strong source-level generalization inference, especially with potential cross-library compound duplication and different collision-energy conventions. Formula-block bootstrap here describes within-source uncertainty, not independent new experiments.

## Decision
Retire both claims: (1) universal quarter superiority, and (2) sufficiency of endpoint Bhattacharyya overlap as a universal geometry selector. Keep support survival as a useful phenomenological control. Next decisive test requires genuinely independent cross-fragmentation source and calibration/metadata aware model fixed before opening that source, plus harder genuine identification ranking and explicit negative results.

Pre-outcome frozen EAD hybrid remains failed, not modified or retroactively rescued.
