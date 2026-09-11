#!/usr/bin/env python3
"""Recomputes every result file and every table, in one command.

    python3 scripts/reproduce.py

Set SOURCE_ROOT to the directory holding the analysed projects (jetty.project,
java-errors-benchmark). Without it the grounding measure (H4, table 7.5) is skipped
and the Holm family shrinks accordingly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def _pricing_path() -> Path:
    for candidate in (ROOT / "docs" / "methodology" / "model-pricing.json", ROOT / "model-pricing.json"):
        if candidate.exists():
            return candidate
    return ROOT / "model-pricing.json"

PRICING = _pricing_path()

STEPS: list[tuple[str, list[str]]] = [
    ("clustering, sampling stage 1", ["thesis/metrics_baselines.py", "--run-dir", "3"]),
    ("metrics, benchmark", ["thesis/metrics_baselines.py", "--run-dir", "4"]),
    ("metrics, main sample", ["thesis/metrics_baselines.py", "--run-dir", "6"]),
    ("hypotheses, benchmark", ["thesis/run_all.py", "--run-dir", "4", "--pricing", str(PRICING)]),
    ("hypotheses, main sample", ["thesis/run_all.py", "--run-dir", "6", "--ablation-run", "7",
                                 "--pricing", str(PRICING)]),
    ("error analysis", ["thesis/error_analysis.py", "--run-dir", "6"]),
    ("RQ7 reading", ["thesis/rq7_reading.py", "--run-dir", "6"]),
    ("thesis tables", ["thesis/thesis_tables.py"]),
]


def main() -> None:
    if not os.environ.get("SOURCE_ROOT"):
        print("SOURCE_ROOT is not set: the grounding measure will be skipped.\n")
    steps = [(label, argv) for label, argv in STEPS if (ROOT / "scripts" / argv[0]).exists()]
    failed: list[str] = []
    for index, (label, argv) in enumerate(steps, 1):
        started = time.monotonic()
        print(f"[{index:>2}/{len(steps)}] {label} ... ", end="", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / argv[0]), *argv[1:]],
                                cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"ok ({time.monotonic() - started:.1f}s)")
        else:
            print("FAILED")
            print(result.stderr.strip()[-600:])
            failed.append(label)
    print()
    if failed:
        print(f"{len(failed)} step(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("All results and tables recomputed.")


if __name__ == "__main__":
    main()
