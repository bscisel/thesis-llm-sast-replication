#!/usr/bin/env python3
"""Tworzy próbę rozszerzającą: po jednym ostrzeżeniu z reguł nieobecnych we wskazanym przebiegu."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pipeline.sample_findings as sf


_SPOTBUGS_ELIGIBLE = {
    (category, priority)
    for category, priorities, _ in sf._SPOTBUGS_STRATA
    for priority in priorities
}
_SONARQUBE_ELIGIBLE = {
    (issue_type, severity) for issue_type, severity, _ in sf._SONARQUBE_STRATA
}


def _is_eligible(tool: str, finding: dict[str, Any]) -> bool:
    details = finding.get("tool_specific") or {}
    if tool == "spotbugs":
        return (details.get("category"), details.get("priority")) in _SPOTBUGS_ELIGIBLE
    if tool == "sonarqube":
        return (details.get("issue_type"), details.get("severity")) in _SONARQUBE_ELIGIBLE
    return True


def _filtered_pool(findings: list[dict], tool: str) -> tuple[list[dict], int, int]:
    non_test = [f for f in findings if f.get("source_set") != "test"]
    excluded_test = len(findings) - len(non_test)
    kept = [
        f for f in non_test
        if not any(s in (f.get("source_path") or "") for s in sf._EXCLUDED_MODULE_SUBSTRINGS)
    ]
    excluded_module = len(non_test) - len(kept)
    return [f for f in kept if _is_eligible(tool, f)], excluded_test, excluded_module


def _rules_in_run(run_dir: Path, filename: str) -> set[str]:
    data = sf._load_json(run_dir / "static_analysis" / "processed" / filename)
    return {str(f.get("type")) for f in data.get("findings", [])}


def _sample_tool(
    source_path: Path,
    output_path: Path,
    tool: str,
    excluded_rules: set[str],
    rules_cap: int,
    rng: random.Random,
) -> dict[str, Any]:
    data = sf._load_json(source_path)
    pool, excluded_test, excluded_module = _filtered_pool(list(data.get("findings", [])), tool)

    by_rule: dict[str, list[int]] = {}
    for index, finding in enumerate(pool):
        by_rule.setdefault(str(finding.get("type")), []).append(index)

    new_rules = sorted(rule for rule in by_rule if rule not in excluded_rules)
    chosen_rules = sorted(rng.sample(new_rules, min(rules_cap, len(new_rules))))
    selected = sorted(rng.choice(by_rule[rule]) for rule in chosen_rules)

    sampled_findings = []
    for index in selected:
        item = dict(pool[index])
        item["original_finding_index"] = index
        sampled_findings.append(item)

    sampled = dict(data)
    sampled["total_findings"] = len(sampled_findings)
    sampled["sample"] = {
        "source_file": str(source_path.relative_to(sf.PROJECT_ROOT)),
        "source_total_findings": len(pool),
        "sample_size": len(sampled_findings),
        "method": "rule_extension_one_per_rule",
        "exclude_test_sources": True,
        "excluded_test_count": excluded_test,
        "excluded_module_count": excluded_module,
        "excluded_rule_count": len(excluded_rules),
        "eligible_new_rules": len(new_rules),
        "rules": {
            rule: {"pool_size": len(by_rule[rule])} for rule in chosen_rules
        },
        "original_indices": selected,
    }
    sampled["findings"] = sampled_findings
    sf._write_json(output_path, sampled)

    return {
        "source_total_findings": len(pool),
        "eligible_new_rules": len(new_rules),
        "sampled_findings": len(sampled_findings),
        "rules": chosen_rules,
        "output": str(output_path.relative_to(sf.PROJECT_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Creates a results/<run>/ directory with one warning per rule absent from the given run."
    )
    parser.add_argument("--source-run", required=True, type=Path, help="Run number or directory holding the raw warnings.")
    parser.add_argument("--exclude-run", required=True, type=Path, help="Run whose rules are excluded from the draw.")
    parser.add_argument("--rules-per-tool", type=int, default=20, help="How many new rules to draw from each tool.")
    parser.add_argument("--output-run", type=Path, help="Output directory; the next free number by default.")
    parser.add_argument("--seed", type=int, default=20260810, help="Generator seed, so the draw can be repeated.")
    args = parser.parse_args()

    source_run = sf._resolve_run_dir(args.source_run)
    exclude_run = sf._resolve_run_dir(args.exclude_run)
    output_run = sf._resolve_run_dir(args.output_run) if args.output_run else sf._next_run_dir()
    if output_run.exists():
        raise SystemExit(f"Output directory already exists: {output_run}")

    source_processed = source_run / "static_analysis" / "processed"
    output_processed = output_run / "static_analysis" / "processed"
    rng = random.Random(args.seed)

    summaries: dict[str, Any] = {}
    for tool, filename in sf.TOOL_FILES.items():
        summaries[tool] = _sample_tool(
            source_processed / filename,
            output_processed / filename,
            tool,
            _rules_in_run(exclude_run, filename),
            args.rules_per_tool,
            rng,
        )

    run_info = sf._load_json(source_run / "run_info.json")
    run_info["run"] = output_run.name
    run_info["timestamp"] = output_run.name.split("_", 1)[1]
    run_info["status"] = "sampled"
    run_info["source_run"] = source_run.name
    run_info["sample"] = {
        "seed": args.seed,
        "method": "rule_extension_one_per_rule",
        "source_run": source_run.name,
        "exclude_run": exclude_run.name,
        "rules_per_tool": args.rules_per_tool,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    run_info["tools"] = {
        name: {
            **(run_info.get("tools", {}).get(name, {}) if isinstance(run_info.get("tools"), dict) else {}),
            "findings": str(summary["sampled_findings"]),
            "source_findings": str(summary["source_total_findings"]),
            "status": "sampled",
        }
        for name, summary in (
            ("SpotBugs", summaries["spotbugs"]),
            ("Error Prone", summaries["error_prone"]),
            ("SonarQube", summaries["sonarqube"]),
        )
    }
    sf._write_json(output_run / "run_info.json", run_info)

    sf._write_json(
        output_run / "sample_info.json",
        {
            "source_run": source_run.name,
            "exclude_run": exclude_run.name,
            "output_run": output_run.name,
            "seed": args.seed,
            "method": "rule_extension_one_per_rule",
            "rules_per_tool": args.rules_per_tool,
            "tools": summaries,
        },
    )

    print(f"Sample written to: {output_run}")
    for tool, summary in summaries.items():
        print(f"\n{tool}: {summary['sampled_findings']} findings, 1 per rule "
              f"(new rules available: {summary['eligible_new_rules']})")
        for rule in summary["rules"]:
            print(f"  {rule}")


if __name__ == "__main__":
    main()
