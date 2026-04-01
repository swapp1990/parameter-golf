# Experiment Documentation Convention

One file per experiment at `dashboard/exp<N>_<short_name>.md`. Each file follows this structure:

```
# Exp N: Title

## Hypothesis
What we expect and why.

## Plan
Config, variants, evaluation criteria, go/no-go thresholds.

## Phase 1: Quick Signal (300s)
### Config
### Results        ← fill after run
### Go/No-Go       ← fill after run

## Phase 2: Per-Token Analysis  (or other deeper eval)
### Results        ← fill after run

## Phase 3: Full Training  (if Phase 1 passes)
### Results        ← fill after run

## Verdict
What we learned. Link to next experiment if applicable.
```

## Rules

- **One document per experiment** — plan and results live together, no separate files.
- Fill in sections top-to-bottom as phases complete.
- Reference from the master table in `dashboard/experiments_plan.md`.
- Failed experiments are just as valuable — document why they failed.
