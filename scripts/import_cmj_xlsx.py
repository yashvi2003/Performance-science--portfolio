"""Convert the professor-supplied Excel files to tidy, de-identified CSV/JSON in data/raw/ (gitignored).

Usage:
    python scripts/import_cmj_xlsx.py --athletes CMJ.xlsx [--reference CMJ_Force_Time_Analysis.xlsx] [--out data/raw]

Expected layout of the athletes workbook (sheet "Sheet1"):
    row 1: "Athlete N" every 3rd column starting at A; row 2: "Time (sec)", "Force (N)"; data from row 3.
Each athlete block is two columns (time, force) followed by one spacer column. Stray formula cells that sit in the
spacer columns are ignored. Output IDs are A01, A02, ... (no names).

Reference workbook (optional): sheet "Data" columns A:B (time, Fz) and cached values from "Results" and Data!M:N are
saved as an UNVERIFIED reference for a regression test. Its provenance must be confirmed before it is used as ground truth.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def import_athletes(path: Path, out: Path, sheet: str = "Sheet1") -> pd.DataFrame:
    ws = load_workbook(path, data_only=True)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    blocks = [(i, h) for i, h in enumerate(header) if isinstance(h, str) and re.match(r"Athlete\s*\d+", h.strip())]
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for col, label in blocks:
        num = int(re.search(r"\d+", label).group())
        pairs = [(r[col], r[col + 1]) for r in rows[2:] if _num(r[col]) and _num(r[col + 1])]
        t = np.array([p[0] for p in pairs], dtype=float)
        f = np.array([p[1] for p in pairs], dtype=float)
        athlete_id = f"A{num:02d}"
        pd.DataFrame({"time_s": t, "force_n": f}).to_csv(out / f"{athlete_id}.csv", index=False)
        dt = np.diff(t)
        med = float(np.median(dt))
        manifest.append(
            {
                "athlete_id": athlete_id,
                "n_samples": len(t),
                "duration_s": round(float(t[-1] - t[0]), 4),
                "fs_hz": round(1 / med, 1),
                "n_irregular_dt": int(np.sum(np.abs(dt - med) > 1e-6)),
                "force_min_n": float(f.min()),
                "force_max_n": float(f.max()),
            }
        )
    man = pd.DataFrame(manifest)
    man.to_csv(out / "cmj_manifest.csv", index=False)
    return man


def import_reference(path: Path, out: Path) -> None:
    wb = load_workbook(path, data_only=True)
    ref = out / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    rows = [(r[0], r[1]) for r in wb["Data"].iter_rows(min_row=2, values_only=True) if _num(r[0]) and _num(r[1])]
    pd.DataFrame(rows, columns=["time_s", "force_n"]).to_csv(ref / "reference_trial.csv", index=False)
    results = {}
    for r in wb["Results"].iter_rows(values_only=True):
        if isinstance(r[0], str) and _num(r[1]):
            results[r[0]] = r[1]
    params = {}
    for r in wb["Data"].iter_rows(min_row=1, max_row=80, min_col=13, max_col=14, values_only=True):
        if isinstance(r[0], str) and _num(r[1]):
            params[r[0]] = r[1]
    (ref / "reference_expected.json").write_text(
        json.dumps({"UNVERIFIED": "values cached in the workbook; verify provenance before using as ground truth",
                    "results": results, "parameters": params}, indent=2)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    a = ap.parse_args()
    man = import_athletes(a.athletes, a.out / "cmj")
    print(man.to_string(index=False))
    if a.reference:
        import_reference(a.reference, a.out)
        print("reference trial and expected values written to", a.out / "reference")


if __name__ == "__main__":
    main()
