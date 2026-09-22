# Portfolio: Neuromuscular monitoring toolkit
Purpose: a research portfolio for a performance science / data analytics research co-op application
at a wearable-performance company. Research rigor and honest framing matter more than feature count.

Three projects, one repo, built strictly in order: **Project 1 -> 2 -> 3**.
This file is the standing instruction set. Read it at the start of every session.

## Read order at the start of every session
1. `STATUS.md`  (what is done, what is next, blockers, session log)
2. `specs/00_shared_contract.md`  (the only interface between projects)
3. The spec of the current project = the first project in STATUS.md that still has unticked tasks

## Rules
1. One project at a time. Do not start Project N+1 until Project N meets its
   "Definition of done" (all tasks ticked, `pytest` green, README written).
2. Inside a project: if `PLAN.md` does not exist yet, write it first (file tree, schemas,
   assumptions) and stop for approval. After that, work milestone by milestone.
3. After EVERY milestone: run `pytest`, fix failures, tick the finished tasks in `STATUS.md`
   (`- [x]`; use `- [~]` for half-done), run `python scripts/progress.py`, then commit.
4. Projects may import only from `shared/`. Never import one project from another.
   Projects talk to each other only through the files defined in the shared contract.
5. Python 3.11+, type hints, docstrings that state the formula used, logging (no print),
   no hardcoded thresholds: everything tunable lives in a YAML config.
6. If a spec is ambiguous: record the assumption in `PLAN.md` under "Assumptions" and
   keep going. If you genuinely need a decision from the user, add it to
   "Blocked / questions" in `STATUS.md` and move to the next unblocked task.
7. Honesty rules (they matter for a portfolio):
   - Synthetic data demonstrates that the pipeline works. It is NOT a research finding.
     Say so in every README and put "synthetic" in plot titles that use it.
   - The composite readiness index is a heuristic, not a validated score. Say so.
   - Alerts are "possible fatigue flags", never diagnoses. Correlations are never causal.
8. Data provenance (mandatory):
   - Every dataset used is logged in `DATA_SOURCES.md`: source/URL/DOI, licence or consent basis, n
     (athletes, sessions, trials), device and sampling rate, file format, what it is used for.
   - Real athlete data is de-identified, never committed (`data/raw/` and `data/private/` are gitignored; `*.xlsx` is gitignored),
     and never published without the user confirming they have permission.
   - Third-party data is cited and its licence respected. Do not redistribute it.
   - PROFESSOR PERMISSION & PUBLIC ARTIFACT GUARDRAILS:
     - The professor has approved analysing `CMJ.xlsx` and `CMJ_Force_Time_Analysis.xlsx` (datasets D1 & D1b) with this cloud AI tool.
     - Anonymized results (aggregate metrics, summary plots, statistics) are permitted in the public repository and README.
     - The raw files themselves (`CMJ.xlsx`, `CMJ_Force_Time_Analysis.xlsx`, or any raw force trace) must NEVER be committed, published, or shown in public screenshots.
     - Confirm `*.xlsx` stays in `.gitignore` at all times.
     - ANONYMIZATION CONSTRAINT: Do NOT treat `A01-A10` as a confirmed anonymization scheme. Do not put those IDs (`A01-A10`) or anything that could re-identify an athlete in public outputs, tables, plots, or READMEs.

9. Research conventions:
   - Each project README opens with the question it addresses, the analysis plan, and closes with
     limitations. Report uncertainty (CIs), not just point estimates.
   - For measurement claims report reliability/agreement properly: CV, ICC, SEM, MDC, bias and limits
     of agreement (Bland-Altman). With few athletes or trials, say the sample is small.
   - Reproducibility: fixed random seeds, versioned config, one command to regenerate every figure.
   - No WHOOP device or WHOOP data is used anywhere. The wearable schema mirrors publicly documented
     metric names (recovery, HRV, resting HR, strain, sleep) so it could accept such data later. The repo
     is independent work, not affiliated with WHOOP; do not imply otherwise.
10. AI-use transparency: each README has a short section on how AI tools were used and how their output
    was verified (tests with known ground truth, manual spot checks, code review of key formulas).
11. Priorities when time runs short: real-data analysis and statistics > reproducibility > figures >
    extra features. Stretch goals in STATUS.md come last. See specs/04_target_roles.md.
12. Before ending a session: append to the session log in `STATUS.md` (what you did, what is
   next, anything a different model or account needs to know to continue).

## Repo layout
```
AGENTS.md  STATUS.md  PLAN.md (created by the agent)
specs/                      project specs + shared contract
scripts/progress.py         prints progress per project
shared/                     contract.py (schemas + validators), synth.py (synthetic data)
p1_nmt_pipeline/            Project 1
p2_nmt_reports/             Project 2
p3_nmt_profiler/            Project 3
data/                       synthetic + processed data (data/raw is gitignored)
tests/
```
