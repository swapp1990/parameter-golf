# Experiment Reports

Each experiment gets its own folder with a standardized set of files.

## Folder Structure

```
reports/
├── README.md (this file)
├── exp17/          ← current best (submission model)
│   ├── analysis.md         — comprehensive report (8 sections + appendix)
│   ├── training_results.md — training run details, val_bpb trajectory
│   ├── data_tables.md      — raw generated tables from checkpoint_analysis.py
│   └── analysis_data.json  — raw JSON data from checkpoint_analysis.py
├── exp14/          ← (future)
├── exp10/          ← (future)
└── exp8a/          ← (future)
```

## Standard Report Sections (analysis.md)

1. **Journey Summary** — what changed from previous experiment, key numbers
2. **Architecture Decisions** — each technique, why, measured impact
3. **Bits Budget** — loss distribution, hard token breakdown, juncture analysis
4. **Inside the Model** — layer ablation, head ablation, MLP ablation, component ablation
5. **Confidence Analysis** — entropy quadrants
6. **What Didn't Work** — failed techniques since last experiment
7. **Evaluation** — standard vs sliding window vs TTT, submission numbers
8. **Next Steps** — what would improve further

## How to Generate

```bash
# On pod (GPU):
python checkpoint_analysis.py --name <exp_name> --windows 500 --ablation_windows 200

# Locally (generates report from JSON):
python generate_report.py dashboard/reports/<exp>/analysis_data.json
```
