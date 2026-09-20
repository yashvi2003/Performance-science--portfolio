# Input data profile (professor-supplied CMJ files)

Exploratory first pass, done once by hand with a quick script. Treat every number as provisional until the pipeline
reproduces it with tests.

## Files
- `CMJ.xlsx`, sheet `Sheet1`: 10 athlete blocks side by side (row 1 "Athlete N", row 2 "Time (sec)" / "Force (N)",
  data from row 3, one spacer column between blocks; a few stray formula cells sit in the spacer columns of block 1).
  Other sheets: `Sheet4` (net force for athletes 1-2 with hard-coded body weights) and an empty `Sheet2`.
- `CMJ_Force_Time_Analysis.xlsx`: a single-trial Excel workbook (Data / Results / Chart / Notes & Method) following
  Chavda et al. (2018). Its trial starts at t = 3.0669 s and did not match any of the 10 athletes' samples.
  Notes in the workbook document where it deviates from the paper (0.5 s baseline instead of 1 s; sustained-threshold
  landing detection; braking search range). Provenance and author must be confirmed.
- The textbook PDF (Sport and Exercise Biomechanics, 401 pages) is general background; it is not the source of the
  workbook's method and contains no CMJ dataset.

## Structure of the athlete data
- 10 athletes, ONE trial each, 2000 Hz (0.0005 s), 3.4-3.6 s per trial, total vertical force in N, time starting at 0.
- Six trials have 28-30 timestamps that deviate slightly from the 0.0005 s spacing (athletes 3, 4, 5, 6, 8). Check and
  handle (resample or document).
- Quiet lead-in before movement is short (first movement roughly 0.55-0.86 s), so a 1 s baseline is not available.

## First-pass observations (provisional)
- Take-off and landing were found for all 10 trials with simple force thresholds. Jump heights came out at roughly
  21-44 cm by impulse-momentum.
- Flight-time and impulse-momentum heights agree within about 1.5 cm for six athletes (2, 4, 5, 6, 8, 9). The other four
  differ by 2.5-6.6 cm (athletes 1, 3, 7, 10). Athletes 3 and 1 also have a noisy quiet phase (SD about 12 N and 10 N vs about
  3 N for most), so baseline and onset choices may drive part of the gap; athlete 3 (6.6 cm, impulse method higher) is the
  first to investigate.
- Flight-phase force averages about +3 to +6 N instead of 0 (small zero offset). Several trials contain negative force values.
- Maximum force is 7-8 x body weight in some trials: that is the landing peak, not the propulsion peak.

## Implications
- No repeated trials or sessions: no reliability statistics, no within-athlete change, no personal baselines from real data.
- What the data is good for: data-quality reporting, method comparison, sensitivity to analyst choices, and a
  reproducible pipeline validated against a reference workbook.
- Ask the professor: who collected the data and when, protocol (arm swing, instructions, surface), device and model,
  whether more trials or sessions exist per athlete, consent and permission to use and publish, and whether names were removed.

## Open-data candidates (checked once by reading their public record pages; download and inspect before relying on them)
- White et al., Zenodo 19136480, CC BY 4.0. 67 participants, 663 CMJ trials (with and without arm swing), vertical GRF from
  two Kistler platforms (recorded at 1000 Hz, provided at 250 Hz, body-weight units, aligned to take-off, 2 s pre-take-off
  window), plus accelerometer signals, participant IDs, jump height (m) and peak power (W/kg). Files: cmj_dataset_noarms.npz,
  cmj_dataset_arms.npz, cmj_dataset_both.npz (36 MB total). Download (untested from here):
  `curl -L -o cmj_dataset_both.npz "https://zenodo.org/records/19136480/files/cmj_dataset_both.npz?download=1"`.
  Inspect with `np.load(path, allow_pickle=False).files` and print shapes before writing the adapter; key names are not documented on the page.
  Good for: repeated trials per participant (reliability, mixed models), checking impulse-momentum height and peak power against reference values, a public demo with attribution.
  Limits: only pre-take-off data (no flight time, landing, or take-off detection: take-off is given), 250 Hz (below the 1000 Hz minimum that one sampling-frequency study recommends), CMJ only, already preprocessed.
- Philipp et al. 2023 (PLOS ONE), season-long CMJ metrics for 12 elite basketball players measured with VALD ForceDecks; the paper points to
  osf.io/bq7n9 for a minimal dataset. The OSF page could not be read from here, so whether it holds raw curves or session-level metrics is unknown.
- Not found in my searches: open raw force-plate datasets for squat jump, drop jump or isometric mid-thigh pull, or open bilateral (left/right) CMJ files.
  Options: synthetic data for those tests, contacting corresponding authors, and checking data-availability statements of the many IMTP/DJ papers.
- Reference implementation for cross-checking numbers (not data): the open-source JumpMetrics package (GitHub stevenhirsch/force-plate-jump-analyses)
  for countermovement and squat jumps from force-plate text exports. Check its licence before reusing any code; cite it if used for comparison.
