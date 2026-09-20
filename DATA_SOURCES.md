# DATA_SOURCES

Log every dataset before it is used. The agent fills this in; the user confirms permissions.
Real athlete data: de-identified IDs only, stored in data/raw/ (gitignored), never published without permission.

| id | source / URL / DOI | licence or consent basis | n (athletes / sessions / trials) | device, sampling rate | format | used for | status |
|----|--------------------|--------------------------|----------------------------------|-----------------------|--------|----------|--------|
| D1 | CMJ.xlsx supplied by the user's professor (collector, protocol, consent: user to confirm) | PERMISSION UNKNOWN: treat as private until the professor confirms (levels: analysis only / anonymised results public / raw data shareable) | 10 athletes / 1 session each / 1 trial each | force plate (model unknown), 2000 Hz, total vertical force (N) | Excel, athlete blocks of Time/Force | P1 data-quality, method comparison, sensitivity; NOT reliability or longitudinal | imported via scripts/import_cmj_xlsx.py |
| D1b | CMJ_Force_Time_Analysis.xlsx (Excel analysis following Chavda et al. 2018; author/provenance to confirm) | as D1 (unknown, treat as private) | 1 trial (not found among D1's athletes) | 2000 Hz | Excel with formulas and cached values | P1 golden test (unverified reference) | imported via scripts/import_cmj_xlsx.py |
| D2 | PMData (Kaggle) | check on dataset page | 16 people, ~5 months | Fitbit Versa 2 + PMSys logs | CSV/JSON | P3 wearable side (proxy strain) | to inspect |
| D3 | shared/synth.py | n/a (generated) | configurable | configurable | CSV | tests, demos | synthetic |
| D4 | White et al., Zenodo, DOI 10.5281/zenodo.19136480 (Swansea University; preprocessed from White et al. 2022, PLOS ONE) | CC BY 4.0 (attribution required); participants gave consent, ethics approval covers further analysis | 67 participants / 663 CMJ trials (331 without arm swing, 332 with arm swing) | two Kistler platforms 1000 Hz, downsampled to 250 Hz; vGRF in body-weight units; 2 s pre-take-off window; plus 250 Hz accelerometer | .npz (3 files, 36 MB) | P1 open-data validation (M2g), reliability; PUBLIC demo figures (with attribution) | to download and inspect |
| D5 | Philipp et al. 2023, PLOS ONE, DOI 10.1371/journal.pone.0286581; dataset at osf.io/bq7n9 | paper is CC BY; dataset licence to check on OSF | 12 elite D1 basketball players / 219 weekly screenings (mean of 3 CMJs) | VALD ForceDecks dual platforms | unknown (paper says "minimal dataset") | P2 longitudinal demo if session-level metrics | contents unverified |
