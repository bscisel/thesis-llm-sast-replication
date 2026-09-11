#!/usr/bin/env python3
"""Wczytywanie runów: klucz findingu, rozwiązywanie katalogu, scalanie powtórzeń."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from analysis.constants import TRUNCATION_REASONS
from typing import Any, Sequence




def _column_of(finding: dict[str, Any]) -> str:
    """Odczytuje kolumnę niezależnie od tego, w którym polu rekord ją trzyma."""
    value = finding.get("column")
    if value in (None, ""):
        value = (finding.get("tool_specific") or {}).get("column")
    return "" if value in (None, "") else str(value)


def _latest_run_dir(project_root: Path) -> Path:
    pattern = re.compile(r"^[0-9]+_.+$")
    runs = [path for path in (project_root / "results").iterdir() if path.is_dir() and pattern.match(path.name)]
    if not runs:
        print(f"No run directories found in {project_root / 'results'}.", file=sys.stderr)
        sys.exit(1)
    return sorted(runs, key=lambda path: int(path.name.split("_", 1)[0]))[-1]


def modal(values: Sequence[str]) -> str | None:
    """Wartość najczęstsza; przy remisie pierwsza z wejścia, nie najczęstsza alfabetycznie."""
    if not values:
        return None
    counts = Counter(values)
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    return winners[0] if len(winners) == 1 else values[0]


def _finding_key(tool: str, finding: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Jednoznacznie identyfikuje ostrzeżenie w obrębie jednej migawki raportu."""
    # start_line równe None daje napis "None" i tak zapisano je w ground_truth.json;
    # zamiana na pusty napis odłączyłaby ostrzeżenia klasowe od ich etykiet.
    return (
        tool,
        str(finding.get("sourcefile", "")),
        str(finding.get("start_line", "")),
        str(finding.get("type", "")),
        str(_column_of(finding)),
    )


def _resolve_run_dir(project_root: Path, value: Path | None) -> Path:
    if value is None:
        return _latest_run_dir(project_root)
    if value.is_absolute():
        return value
    if len(value.parts) == 1 and value.parts[0].isdigit():
        run_id = f"{int(value.parts[0]):03d}"
        matches = sorted((project_root / "results").glob(f"{run_id}_*"), key=lambda path: int(path.name.split("_", 1)[0]))
        if not matches:
            print(f"No run found with number: {value}", file=sys.stderr)
            sys.exit(1)
        return matches[-1]
    if value.parts and value.parts[0] == "results":
        return project_root / value
    return project_root / "results" / value


def _aggregate_runs(runs: list[dict[str, Any]]) -> tuple[bool | None, dict[str, Any]]:
    """Ustala werdykt głosem większościowym z powtórzeń jednego ostrzeżenia."""
    valid: list[bool] = [
        bool(run["is_true_positive"])
        for run in runs
        if "is_true_positive" in run
        and isinstance(run["is_true_positive"], bool)
        and str(run.get("stop_reason") or "") not in TRUNCATION_REASONS
    ]
    if not valid:
        return None, {
            "valid_runs": 0,
            "total_runs": len(runs),
            "stability": None,
            "true_votes": 0,
            "false_votes": 0,
        }

    counts = Counter(valid)
    true_votes = counts[True]
    false_votes = counts[False]
    predicted_tp = true_votes >= false_votes
    stability = max(true_votes, false_votes) / len(valid)
    return predicted_tp, {
        "valid_runs": len(valid),
        "total_runs": len(runs),
        "stability": stability,
        "true_votes": true_votes,
        "false_votes": false_votes,
    }

ROOT = Path(__file__).resolve().parents[2]


def find_run_dir(spec: int | str | Path, project_root: Path = ROOT) -> Path:
    """Znajduje katalog przebiegu po numerze, nazwie albo ścieżce."""
    text = str(spec)
    if text.isdigit():
        matches = sorted((project_root / "results").glob(f"{int(text):03d}_*"))
        if not matches:
            print(f"No run found with number: {text}", file=sys.stderr)
            sys.exit(1)
        return matches[-1]
    path = Path(text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "results":
        return project_root / path
    return project_root / "results" / path
