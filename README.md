# Master Thesis Project

A clean, reproducible layout for your thesis: LaTeX paper in `paper/`, notebooks and reusable code separated, explicit data lifecycle, and Windows-friendly tasks.

## Structure
- paper/: LaTeX sources, figures, tables, and build outputs
- notebooks/: Jupyter notebooks (exploration, experiments, reports)
- src/: Reusable Python modules importable from notebooks/scripts
- data/: Raw → interim → processed data 
- results/: Experiments, models, metrics, logs, and derived figures
- assets/: Shared images/logos/diagrams (svg/drawio)
- scripts/: Helper CLIs and automation scripts
- config/: YAML configs (base + per-experiment)
- docs/: Notes, small design docs, and decisions
- Presentations/: Slides and presentation assets

## Data Policy
- Do NOT commit large/raw data to Git. Use Git LFS for modest binaries (pdf/png/model files) or DVC for datasets/models with remote storage.
- Place inputs under `data/raw/`. Promote to `data/interim/` and `data/processed/` via scripts.

## Reproducibility
- Keep parameters in `config/*.yaml`; copy the resolved config into each run folder under `results/experiments/<run>/config.yaml`.
- Set a single `seed` value and apply it consistently (NumPy/PyTorch/etc.).
- Name runs like `YYYYMMDD-HHMM_task_seed42`.

## Notebooks
- `notebooks/exploration/`: ad-hoc exploration and EDA
- `notebooks/experiments/`: reproducible, parameterized notebooks
- `notebooks/reports/`: executed notebooks for sharing

## Results
Each run under `results/experiments/<run>/` should contain:
- `config.yaml`, `metrics.csv/json`, `logs/`, `figures/`, and optionally `model/`

## Tasks
Run with PowerShell:
```powershell
.\tasks.ps1 help         # list tasks
.\tasks.ps1 env          # print env info
.\tasks.ps1 paper        # build LaTeX PDF to paper/build/
.\tasks.ps1 lint         # ruff + nbqa
.\tasks.ps1 test         # pytest (if tests added)
.\tasks.ps1 run-notebooks # placeholder for papermill runs
```

## Next Steps
- Move existing notebooks from `code/` to `notebooks/exploration/`.
- Add your bibliography to `paper/bibliography.bib` and update `paper/main.tex` to use it.
- Add an experiment config under `config/experiment/` and a parameterized notebook under `notebooks/experiments/`.
