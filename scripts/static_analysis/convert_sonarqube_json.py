#!/usr/bin/env python3
"""Przekształca surowy JSON z API SonarQube do wspólnego formatu."""

import json
import os
import re
import sys
from collections import Counter
from static_analysis.paths import JAVA_SOURCE_DIR_RE, after_colon


BENCHMARK_CLASS_PREFIX_RE = re.compile(r"^_\d{2}_")


def parse_component_path(component: str) -> str:
    return after_colon(component)


def parse_classname(sourcefile: str) -> str:
    """Wyprowadza nazwę klasy z nazwy pliku źródłowego."""
    if sourcefile.endswith(".java"):
        return sourcefile[:-5]
    return sourcefile


def parse_path_metadata(source_path: str, sourcefile: str, classname: str) -> dict:
    """Wyprowadza metadane lokalizacji ze ścieżki źródeł w układzie Mavena."""
    metadata = {
        "source_path": source_path,
        "source_set": None,
        "package": None,
        "fully_qualified_classname": classname,
        "module": None,
    }

    match = JAVA_SOURCE_DIR_RE.match(source_path)
    if not match:
        return metadata

    package_path = match.group("package_path")
    package_name = package_path.replace("/", ".") if package_path else None
    metadata.update(
        {
            "source_set": match.group("source_set"),
            "package": package_name,
            "fully_qualified_classname": f"{package_name}.{classname}" if package_name else classname,
            "module": match.group("module") or None,
        }
    )
    return metadata


def parse_abbrev(rule: str) -> str:
    return after_colon(rule)


def should_filter_issue(issue: dict, sourcefile: str, classname: str) -> tuple[bool, str | None]:
    """Odrzuca ostrzeżenia SonarQube nieistotne dla bazy przykładowych błędów."""
    rule = issue.get("rule") or ""
    message = issue.get("message") or ""

    if rule == "java:S106" and "System.out" in message:
        return True, "system_out_logger_rule"

    if rule == "java:S101" and (
        BENCHMARK_CLASS_PREFIX_RE.match(classname) or BENCHMARK_CLASS_PREFIX_RE.match(sourcefile)
    ):
        return True, "benchmark_class_prefix_naming_rule"

    return False, None


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    sonarqube_version = raw.get("sonarqube_version", "unknown")
    issues = raw.get("issues", [])
    apply_benchmark_filters = os.getenv("SONAR_APPLY_BENCHMARK_FILTERS", "").lower() in {"1", "true", "yes"}

    findings = []
    filtered_reasons = Counter()
    for issue in issues:
        rule = issue.get("rule") or ""
        component = issue.get("component") or ""
        message = issue.get("message") or ""
        text_range = issue.get("textRange") or {}

        start_line = text_range.get("startLine") or issue.get("line")
        end_line = text_range.get("endLine") or issue.get("line")

        source_path = parse_component_path(component)
        sourcefile = source_path.split("/")[-1]
        classname = parse_classname(sourcefile)
        path_metadata = parse_path_metadata(source_path, sourcefile, classname)
        abbrev = parse_abbrev(rule)
        if apply_benchmark_filters:
            filtered, reason = should_filter_issue(issue, sourcefile, classname)
            if filtered:
                filtered_reasons[reason] += 1
                continue

        finding = {
            "type": rule,
            "abbrev": abbrev,
            "sourcefile": sourcefile,
            "start_line": str(start_line) if start_line is not None else None,
            "end_line": str(end_line) if end_line is not None else None,
            "classname": classname,
            "fully_qualified_classname": path_metadata["fully_qualified_classname"],
            "source_path": path_metadata["source_path"],
            "source_set": path_metadata["source_set"],
            "package": path_metadata["package"],
            "message": message,
            "description": message,
            "tool_specific": {
                "severity": issue.get("severity"),
                "issue_type": issue.get("type"),
                "effort": issue.get("effort"),
                "status": issue.get("status"),
                "tags": issue.get("tags") or [],
                "component": component,
                "module": path_metadata["module"],
            },
        }
        findings.append(finding)

    output = {
        "tool": "sonarqube",
        "version": sonarqube_version,
        "total_findings": len(findings),
        "filtered_out": {
            "total": sum(filtered_reasons.values()),
            "reasons": dict(sorted(filtered_reasons.items())),
        },
        "findings": findings,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    filtered_total = sum(filtered_reasons.values())
    print(
        f"Converted {len(findings)} warnings -> {output_path}"
        f" (filtered out {filtered_total})"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
