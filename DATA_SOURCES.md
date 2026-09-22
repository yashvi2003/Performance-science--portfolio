# DATA_SOURCES

Log every dataset before it is used. The agent fills this in; the user confirms permissions.

## Data Governance & Approval Guardrails
- **Professor Approval**: The professor has approved analysing `CMJ.xlsx` and `CMJ_Force_Time_Analysis.xlsx` (datasets D1 & D1b) with this cloud AI tool.
- **Permitted Public Outputs**: Anonymized results (aggregate metrics, summary plots, statistics) may appear in the public repository and README.
- **Raw Data Protection**: Raw files (`CMJ.xlsx`, `CMJ_Force_Time_Analysis.xlsx`, or any raw force-time traces) must NEVER be committed, published, or shown in public screenshots.
- **Gitignore Confirmation**: Confirmed that `*.xlsx`, `data/raw/`, and `data/private/` remain in `.gitignore`.
- **Anonymization Constraint**: Do NOT treat `A01-A10` as a confirmed anonymization scheme. Do not put those IDs (`A01-A10`) or anything that could re-identify an athlete in public outputs, tables, plots, or documentation.

| id | source / URL / DOI | licence or consent basis | n (athletes / sessions / trials) | device, sampling rate | format | used for | status |
|----|--------------------|--------------------------|----------------------------------|-----------------------|--------|----------|--------|
| D1 | CMJ.xlsx supplied by the user's professor (collector, protocol, consent: user to confirm) | Approved by professor for cloud AI analysis. Anonymized results (metrics, plots, statistics) permitted in public repo/README. Raw file must never be committed, published, or screenshotted; `*.xlsx` is gitignored. Athlete IDs `A01-A10` are not a confirmed anonymization scheme and must not appear in public outputs. | 10 athletes / 1 session each / 1 trial each | force plate (model unknown), 2000 Hz, total vertical force (N) | Excel, athlete blocks of Time/Force | P1 data-quality, method comparison, sensitivity; NOT reliability or longitudinal | Approved for cloud AI analysis; anonymized results public; raw files private |
| D1b | CMJ_Force_Time_Analysis.xlsx (Excel analysis following Chavda et al. 2018; author/provenance to confirm) | Approved by professor for cloud AI analysis (as D1). Anonymized results permitted in public repo; raw file private and gitignored (`*.xlsx`). | 1 trial (not found among D1's athletes) | 2000 Hz | Excel with formulas and cached values | P1 golden test (unverified reference) | Approved for cloud AI analysis; anonymized results public; raw file private |
| D2 | PMData (Kaggle) | check on dataset page | 16 people, ~5 months | Fitbit Versa 2 + PMSys logs | CSV/JSON | P3 wearable side (proxy strain) | to inspect |
| D3 | shared/synth.py | n/a (generated) | configurable | configurable | CSV | tests, demos | synthetic |
| D4 | White et al., Zenodo, DOI 10.5281/zenodo.19136480 (Swansea University; preprocessed from White et al. 2022, PLOS ONE) | CC BY 4.0 (attribution required); participants gave consent, ethics approval covers further analysis | 67 participants / 663 CMJ trials (331 without arm swing, 332 with arm swing) | two Kistler platforms 1000 Hz, downsampled to 250 Hz; vGRF in body-weight units; 2 s pre-take-off window; plus 250 Hz accelerometer | .npz (3 files, 36 MB) | P1 open-data validation (M2g), reliability; PUBLIC demo figures (with attribution) | to download and inspect |
| D5 | Philipp et al. 2023, PLOS ONE, DOI 10.1371/journal.pone.0286581; dataset at osf.io/bq7n9 | paper is CC BY; dataset licence to check on OSF | 12 elite D1 basketball players / 219 weekly screenings (mean of 3 CMJs) | VALD ForceDecks dual platforms | unknown (paper says "minimal dataset") | P2 longitudinal demo if session-level metrics | contents unverified |
