---
name: microstructure-experiments
description: Runs the constrained experiment grid for the ES strategy repo.
---

When invoked:

1) Generate a RUN_ID (timestamp or passed arg).
2) Execute:
   python scripts/experiment_agent.py --config configs/experiment_agent.json --outdir artifacts/experiment_agent/${RUN_ID}
3) After completion:
   - Print path to report.md
   - Print top 3 rows from comparison.csv
   - Print whether any run breached daily loss or trailing DD

Security:
 - Only run the approved CLI command.
 - Do not modify scoring code.
 - Do not change data range.
 - Do not alter cost model.