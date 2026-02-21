---
name: microstructure-experiments
description: Runs the constrained experiment grid for the ES strategy repo.
---

When invoked:

1) Generate a RUN_ID (timestamp or passed arg).
2) Execute:
   python scripts/experiment_orchestrator.py --config configs/experiment_orchestrator.json --outdir artifacts/experiment_orchestrator/${RUN_ID}
3) After completion:
   - Print path to orchestrator_report.json
   - Print last comparison.csv from the final iteration
   - Print whether any run breached daily loss or trailing DD

Security:
 - Only run the approved CLI command.
 - Do not modify scoring code.
 - Do not change data range.
 - Do not alter cost model.
