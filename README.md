# Portfolio: Neuromuscular monitoring toolkit

Target: WHOOP Performance Science & Health Outcomes Co-Op (main) and Data Analytics Co-Op (fallback); see specs/04_target_roles.md.

Three projects, one shared data contract:
1. Project 1: automatic jump analysis engine (identify the jump, metrics, VALD-style graph, drop-a-file app)
2. Project 2: baselines, readiness, SQL layer and longitudinal reporting
3. Project 3: real-world-evidence statistics on public wearable data (mixed models, causal thinking) plus asymmetry module

See AGENTS.md for how the AI agent should work, STATUS.md for progress, and START_PROMPT.md for the message that starts any session.

## First-day checklist (human steps)
1. Email/meet the professor (permissions, data questions). Also ask whether the athlete data may be analysed with cloud AI coding tools,
   because the tool sends file contents to an external model provider. Until they answer, keep the real files OUT of the workspace and use synthetic data only.
2. Unzip this kit, `git init`, first commit (`.gitignore` already excludes data, .xlsx and .npz files). Open the folder in Antigravity as the workspace.
3. Paste START_PROMPT.md into a new Agent Manager conversation (Planning mode). Approve PLAN.md. Work milestone by milestone.
4. Only after the professor approves: copy `CMJ.xlsx` and `CMJ_Force_Time_Analysis.xlsx` into `data/private/`, then run
   `python scripts/import_cmj_xlsx.py --athletes data/private/CMJ.xlsx --reference data/private/CMJ_Force_Time_Analysis.xlsx --out data/raw`.
5. Download the open White et al. CMJ files (see specs/05_input_data_profile.md) into `data/raw/open/`; this data is CC BY and safe to use with the tool.
6. After every session run `python scripts/progress.py` and `pytest`, then commit.
7. Verify what the AI produces (see below).

## What you must check yourself
- Formulas in the CMJ module against a textbook or paper: impulse-momentum velocity, jump height, RSImod, peak power.
- Every detected onset/take-off/landing on the plots for the 10 real trials, especially athletes 3 and 1.
- That tests use known ground truth (synthetic trials with analytic jump height), not just "runs without errors".
- That no README number, plot or screenshot comes from the professor's data unless permission is recorded in DATA_SOURCES.md.
- That every claim in the README and your CV is something the repo actually does.

## Minimum viable portfolio (if time runs short)
Tell the agent to work in this order and leave the rest as "future work":
1. Project 0 setup and shared contract.
2. Project 1 up to M2f (CMJ engine, real-data validation, sensitivity) plus M5a (single-trial report).
3. Project 3 M1a-M3 (wearable adapter, written analysis plan, mixed-effects analysis, causal-thinking section).
4. Project 2 M1b (SQL layer) plus M2d (mixed model on labelled synthetic data).
5. The 2-page research brief.
