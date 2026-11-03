# Scripts

Place small command-line helpers here to orchestrate data prep, training, or evaluation.
Prefer reusable logic in `src/` and thin CLI wrappers here.

To regenerate the battery reachability-violation and shaped-reward figure, run
`python3 scripts/plot_battery_reachability_reward.py` from the repository root.
The figure is written to `paper/figures/battery_reachability_reward.pdf`.

To generate the manuscript PDF with changes highlighted against the latest Git
commit (`HEAD`), save your edits and run from the repository root:

```bash
./scripts/build_paper_diff.sh
```

The script requires `git`, `latexdiff`, `latexdiff-vc`, and `latexmk`, plus the
LaTeX packages and bibliography tools needed by the manuscript. It can be called
from any directory using its full path. It overwrites the generated diff files in
`paper/build/changes/`; open `paper/build/changes/main-diff.pdf` to view the result.
Run it again after further edits to refresh the PDF.
