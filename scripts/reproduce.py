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

# Run 003 keeps the default resampling scheme: its Error Prone strata are real and
# do not carry across sampling stages, so the merged sample must cluster by rule instead.
# population_concentration.py is not listed: it needs run 002's full tool reports,
# which are 60 MB and not shipped. Its output is included as a stored result.
STEPS: list[tuple[str, list[str]]] = [
    ("metrics, run 003", ["thesis/metrics_baselines.py", "--run-dir", "3"]),
    ("metrics, run 004", ["thesis/metrics_baselines.py", "--run-dir", "4", "--cluster-by", "rule"]),
    ("metrics, run 005", ["thesis/metrics_baselines.py", "--run-dir", "5", "--cluster-by", "rule"]),
    ("metrics, run 006", ["thesis/metrics_baselines.py", "--run-dir", "6", "--cluster-by", "rule"]),
    ("hypotheses, run 003", ["thesis/run_all.py", "--run-dir", "3", "--pricing", str(PRICING)]),
    ("hypotheses, run 004", ["thesis/run_all.py", "--run-dir", "4", "--cluster-by", "rule", "--pricing", str(PRICING)]),
    ("hypotheses, run 005", ["thesis/run_all.py", "--run-dir", "5", "--cluster-by", "rule", "--pricing", str(PRICING)]),
    ("hypotheses, run 006", ["thesis/run_all.py", "--run-dir", "6", "--ablation-run", "7",
                             "--cluster-by", "rule", "--pricing", str(PRICING)]),
    ("unanimity gate, run 003", ["thesis/h3_consistency_gate.py", "--run-dir", "3"]),
    ("unanimity gate, run 004", ["thesis/h3_consistency_gate.py", "--run-dir", "4", "--cluster-by", "rule"]),
    ("unanimity gate, run 005", ["thesis/h3_consistency_gate.py", "--run-dir", "5", "--cluster-by", "rule"]),
    ("unanimity gate, run 006", ["thesis/h3_consistency_gate.py", "--run-dir", "6", "--cluster-by", "rule"]),
    ("code context, run 007", ["thesis/h6_code_context.py", "--main-run", "6", "--ablation-run", "7",
                               "--json", str(next(iter(sorted(ROOT.glob("results/007_*")))) / "analysis" / "h6.json")]),
    ("grounding diagnostics", ["thesis/grounding_diagnostics.py", "--run-dir", "6"]),
    ("retry scale", ["thesis/retry_scale.py"]),
    ("thesis tables", ["thesis/thesis_tables.py"]),
]


def main() -> None:
    if not os.environ.get("SOURCE_ROOT"):
        print("SOURCE_ROOT is not set: the grounding measure will be skipped.\n")
    failed: list[str] = []
    for index, (label, argv) in enumerate(STEPS, 1):
        started = time.monotonic()
        print(f"[{index:>2}/{len(STEPS)}] {label} ... ", end="", flush=True)
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
