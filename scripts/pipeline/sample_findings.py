#!/usr/bin/env python3
"""Tworzy odtwarzalną próbę warstwową ostrzeżeń jako nowy katalog przebiegu."""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_FILES = {
    "spotbugs": "spotbugs.json",
    "error_prone": "error_prone.json",
    "sonarqube": "sonarqube.json",
}

_EXCLUDED_MODULE_SUBSTRINGS: list[str] = [
    "jetty-ee-test-resources",
]

_SPOTBUGS_STRATA: list[tuple[str, frozenset[str], int]] = [
    ("CORRECTNESS",    frozenset({"1", "2"}),  6),
    ("SECURITY",       frozenset({"1", "2"}),  3),
    ("PERFORMANCE",    frozenset({"1", "2"}), 17),
    ("MT_CORRECTNESS", frozenset({"1", "2"}), 34),
    ("BAD_PRACTICE",   frozenset({"1", "2"}), 29),
    ("MALICIOUS_CODE", frozenset({"1", "2"}), 11),
]

_ERROR_PRONE_NAMED_STRATA: list[tuple[str, int]] = [
    ("FutureReturnValueIgnored", 17),
    ("MissingCasesInEnumSwitch", 12),
    ("ReferenceEquality",        10),
    ("IntLongMath",              10),
    ("SameNameButDifferent",     10),
    ("OperatorPrecedence",       10),
    ("BanJNDI",                   3),
    ("MissingOverride",           8),
    ("UnnecessaryParentheses",    4),
    ("UnusedVariable",            3),
    ("MissingSummary",            3),
]
_ERROR_PRONE_REMAINING_CAP = 10

_SONARQUBE_STRATA: list[tuple[str, str, int]] = [
    ("VULNERABILITY", "BLOCKER",   2),
    ("BUG",           "BLOCKER",  21),
    ("BUG",           "CRITICAL",  2),
    ("BUG",           "MAJOR",    35),
    ("CODE_SMELL",    "BLOCKER",  22),
    ("CODE_SMELL",    "CRITICAL", 13),
    ("CODE_SMELL",    "MAJOR",     5),
]


def _run_sort_key(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def _resolve_run_dir(value: Path) -> Path:
    if value.is_absolute():
        return value
    if len(value.parts) == 1 and value.parts[0].isdigit():
        run_id = f"{int(value.parts[0]):03d}"
        matches = sorted((PROJECT_ROOT / "results").glob(f"{run_id}_*"), key=_run_sort_key)
        if not matches:
            raise SystemExit(f"Nie znaleziono przebiegu o numerze: {value}")
        return matches[-1]
    if value.parts and value.parts[0] == "results":
        return PROJECT_ROOT / value
    return PROJECT_ROOT / "results" / value


def _next_run_dir() -> Path:
    pattern = re.compile(r"^[0-9]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    runs = [
        path
        for path in (PROJECT_ROOT / "results").iterdir()
        if path.is_dir() and pattern.match(path.name)
    ]
    next_id = _run_sort_key(sorted(runs, key=_run_sort_key)[-1]) + 1 if runs else 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return PROJECT_ROOT / "results" / f"{next_id:03d}_{timestamp}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pick(pool: list[int], cap: int, rng: random.Random) -> list[int]:
    n = min(cap, len(pool))
    return sorted(rng.sample(pool, n))


def _sample_spotbugs(findings: list[dict], rng: random.Random) -> tuple[list[int], list[dict]]:
    selected: set[int] = set()
    strata: list[dict] = []
    for category, priorities, cap in _SPOTBUGS_STRATA:
        pool = [
            i for i, f in enumerate(findings)
            if i not in selected
            and f.get("tool_specific", {}).get("category") == category
            and f.get("tool_specific", {}).get("priority") in priorities
        ]
        chosen = _pick(pool, cap, rng)
        selected.update(chosen)
        strata.append({"category": category, "cap": cap, "available": len(pool), "sampled": len(chosen)})
    return sorted(selected), strata


def _sample_error_prone(findings: list[dict], rng: random.Random) -> tuple[list[int], list[dict]]:
    selected: set[int] = set()
    strata: list[dict] = []
    for type_name, cap in _ERROR_PRONE_NAMED_STRATA:
        pool = [i for i, f in enumerate(findings) if i not in selected and f.get("type") == type_name]
        chosen = _pick(pool, cap, rng)
        selected.update(chosen)
        strata.append({"type": type_name, "cap": cap, "available": len(pool), "sampled": len(chosen)})
    remaining_pool = [i for i in range(len(findings)) if i not in selected]
    chosen = _pick(remaining_pool, _ERROR_PRONE_REMAINING_CAP, rng)
    selected.update(chosen)
    strata.append({"type": "__remaining__", "cap": _ERROR_PRONE_REMAINING_CAP, "available": len(remaining_pool), "sampled": len(chosen)})
    return sorted(selected), strata


def _sample_sonarqube(findings: list[dict], rng: random.Random) -> tuple[list[int], list[dict]]:
    selected: set[int] = set()
    strata: list[dict] = []
    for issue_type, severity, cap in _SONARQUBE_STRATA:
        pool = [
            i for i, f in enumerate(findings)
            if i not in selected
            and f.get("tool_specific", {}).get("issue_type") == issue_type
            and f.get("tool_specific", {}).get("severity") == severity
        ]
        chosen = _pick(pool, cap, rng)
        selected.update(chosen)
        strata.append({"issue_type": issue_type, "severity": severity, "cap": cap, "available": len(pool), "sampled": len(chosen)})
    return sorted(selected), strata


_SAMPLERS = {
    "spotbugs": _sample_spotbugs,
    "error_prone": _sample_error_prone,
    "sonarqube": _sample_sonarqube,
}


def _sample_report(
    source_path: Path,
    output_path: Path,
    tool: str,
    rng: random.Random,
    exclude_test_sources: bool = True,
) -> dict[str, Any]:
    data = _load_json(source_path)
    findings = list(data.get("findings", []))

    excluded_test_count = 0
    if exclude_test_sources:
        filtered = [f for f in findings if f.get("source_set") != "test"]
        excluded_test_count = len(findings) - len(filtered)
        findings = filtered

    excluded_module_count = 0
    if _EXCLUDED_MODULE_SUBSTRINGS:
        filtered = [
            f for f in findings
            if not any(s in (f.get("source_path") or "") for s in _EXCLUDED_MODULE_SUBSTRINGS)
        ]
        excluded_module_count = len(findings) - len(filtered)
        findings = filtered

    selected_indices, strata = _SAMPLERS[tool](findings, rng)

    sampled_findings = []
    for index in selected_indices:
        item = dict(findings[index])
        item["original_finding_index"] = index
        sampled_findings.append(item)

    sampled = dict(data)
    sampled["total_findings"] = len(sampled_findings)
    sampled["sample"] = {
        "source_file": str(source_path.relative_to(PROJECT_ROOT)),
        "source_total_findings": len(findings),
        "sample_size": len(sampled_findings),
        "method": "stratified",
        "exclude_test_sources": exclude_test_sources,
        "excluded_test_count": excluded_test_count,
        "excluded_module_count": excluded_module_count,
        "original_indices": selected_indices,
        "strata": strata,
    }
    sampled["findings"] = sampled_findings
    _write_json(output_path, sampled)

    return {
        "source_total_findings": len(findings),
        "excluded_test_count": excluded_test_count,
        "excluded_module_count": excluded_module_count,
        "sampled_findings": len(sampled_findings),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "strata": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tworzy katalog results/<przebieg>/ z próbą warstwową ostrzeżeń dla każdego narzędzia."
    )
    parser.add_argument("--source-run", required=True, type=Path, help="Numer albo katalog przebiegu, z którego losujemy.")
    parser.add_argument("--output-run", type=Path, help="Katalog wynikowy; domyślnie kolejny wolny numer.")
    parser.add_argument("--seed", type=int, default=20260503, help="Ziarno generatora, żeby losowanie dało się powtórzyć.")
    parser.add_argument("--exclude-test-sources", action=argparse.BooleanOptionalAction, default=True, help="Wyklucza ostrzeżenia z kodu testowego przed losowaniem (domyślnie tak).")
    args = parser.parse_args()

    source_run = _resolve_run_dir(args.source_run)
    output_run = _resolve_run_dir(args.output_run) if args.output_run else _next_run_dir()
    if output_run.exists():
        raise SystemExit(f"Katalog wynikowy już istnieje: {output_run}")

    source_processed = source_run / "static_analysis" / "processed"
    output_processed = output_run / "static_analysis" / "processed"
    rng = random.Random(args.seed)

    summaries: dict[str, Any] = {}
    for tool, filename in TOOL_FILES.items():
        summaries[tool] = _sample_report(
            source_processed / filename,
            output_processed / filename,
            tool,
            rng,
            exclude_test_sources=args.exclude_test_sources,
        )

    run_info = _load_json(source_run / "run_info.json")
    run_info["run"] = output_run.name
    run_info["timestamp"] = output_run.name.split("_", 1)[1]
    run_info["status"] = "sampled"
    run_info["source_run"] = source_run.name
    run_info["sample"] = {
        "seed": args.seed,
        "method": "stratified",
        "source_run": source_run.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    run_info["tools"] = {
        name: {
            **(run_info.get("tools", {}).get(label, {}) if isinstance(run_info.get("tools"), dict) else {}),
            "findings": str(summary["sampled_findings"]),
            "source_findings": str(summary["source_total_findings"]),
            "status": "sampled",
        }
        for name, label, summary in (
            ("SpotBugs",    "SpotBugs",    summaries["spotbugs"]),
            ("Error Prone", "Error Prone", summaries["error_prone"]),
            ("SonarQube",   "SonarQube",   summaries["sonarqube"]),
        )
    }
    _write_json(output_run / "run_info.json", run_info)


    _write_json(
        output_run / "sample_info.json",
        {
            "source_run": source_run.name,
            "output_run": output_run.name,
            "seed": args.seed,
            "method": "stratified",
            "tools": summaries,
        },
    )

    print(f"Próbę zapisano w: {output_run}")
    for tool, summary in summaries.items():
        print(f"\n{tool}: {summary['sampled_findings']}/{summary['source_total_findings']}")
        for s in summary["strata"]:
            label = s.get("category") or s.get("type") or f"{s['issue_type']} {s['severity']}"
            print(f"  {label}: {s['sampled']}/{s['available']} (cap {s['cap']})")


if __name__ == "__main__":
    main()