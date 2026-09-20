# Data-layer implementation notes

- `collision_energy_ev` is treated as a list-valued field. Length > 1 means a merged acquisition and is not collapsed to a mean energy.
- `base_peak_intensity` is retained so normalized spectra can be mapped back to raw-count scale where available.
- The visible Kaggle test file is excluded from validation/model selection.
- All structure-held-out validation uses `inchikey14` as the leakage boundary.
- The first benchmark is deliberately CPU-first and library-retrieval focused; learned structure-to-spectrum models come only after the static/dynamic signal is established.
