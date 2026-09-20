"""Print progress per project from STATUS.md.

Usage: python scripts/progress.py [path/to/STATUS.md]
[x] = 1 point, [~] = 0.5, [ ] = 0.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HEADING = re.compile(r"^##\s+(Project\s+\d+\b.*)$")
ITEM = re.compile(r"^\s*-\s+\[( |x|X|~)\]\s+(.*)$")
SCORE = {" ": 0.0, "~": 0.5, "x": 1.0, "X": 1.0}


def parse(text: str) -> list[dict]:
    projects: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        h = HEADING.match(line)
        if h:
            current = {"name": h.group(1).strip(), "score": 0.0, "total": 0, "next": None}
            projects.append(current)
            continue
        if line.startswith("## "):
            current = None
            continue
        m = ITEM.match(line)
        if m and current is not None:
            mark, label = m.groups()
            current["total"] += 1
            current["score"] += SCORE[mark]
            if mark not in ("x", "X") and current["next"] is None:
                current["next"] = label.strip()
    return projects


def bar(pct: float, width: int = 20) -> str:
    filled = round(width * pct / 100)
    return "#" * filled + "-" * (width - filled)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "STATUS.md"
    projects = parse(path.read_text(encoding="utf-8"))
    if not projects:
        print("No projects found in", path)
        return 1
    total = sum(p["total"] for p in projects)
    score = sum(p["score"] for p in projects)
    for p in projects:
        pct = 100 * p["score"] / p["total"] if p["total"] else 0.0
        print(f"{p['name'][:58]:<58} [{bar(pct)}] {pct:5.1f}%  ({p['score']:g}/{p['total']})")
    overall = 100 * score / total if total else 0.0
    print("-" * 100)
    print(f"{'OVERALL':<58} [{bar(overall)}] {overall:5.1f}%  ({score:g}/{total})")
    nxt = next((p for p in projects if p["next"]), None)
    if nxt:
        print(f"\nNext task ({nxt['name'].split(' - ')[0]}): {nxt['next']}")
    else:
        print("\nAll tasks complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
