# Benchmark Results Archiving Invariant

## Core Rule
**Never overwrite previous benchmark graphs or results directly.**
Before executing any experiment or benchmark script that generates output plots (`.png`) or data (`.json`, `.csv`) in `results/`:

1. **Check for Existing Artifacts**:
   Identify whether previous results or PNG graphs for that experiment already exist in `results/`.

2. **Archive to `previous_results/`**:
   Before running the new benchmark, move the existing results into a dedicated subfolder within `previous_results/`:
   ```
   previous_results/
   ├── <experiment_run_name_or_timestamp>/
   │   ├── <previous_plot_1>.png
   │   ├── <previous_plot_2>.png
   │   └── <previous_data>.json
   └── <another_experiment_run>/
       └── ...
   ```
   Folder names should be descriptive of the run (e.g., `previous_results/phase8_noise_sweep/`, `previous_results/phase4_training_seed5_run1/`, etc.).

3. **Maintain Canonical Headline Results in `results/`**:
   The active `results/` folder holds the latest/headline results directly referenced by `README.md` and repository documentation.
