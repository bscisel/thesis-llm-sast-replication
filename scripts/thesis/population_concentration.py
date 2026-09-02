"""Skupienie ostrzeżeń w regułach: populacja runu 002 wobec próby runu 003."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.runs import find_run_dir

NAMES = {"spotbugs": "SpotBugs", "error_prone": "Error Prone", "sonarqube": "SonarQube"}
EXCLUDED_MODULE = "jetty-ee-test-resources"


def _populacja(source_name: Path, tool: str) -> list[dict]:
    path = source_name / "static_analysis" / "processed" / f"{tool}.json"
    findings = json.loads(path.read_text(encoding="utf-8"))["findings"]
    return [
        f for f in findings
        if f.get("source_set") != "test" and EXCLUDED_MODULE not in (f.get("source_path") or "")
    ]


def _sample(target: Path) -> collections.Counter:
    data = json.loads((target / "ground_truth.json").read_text(encoding="utf-8"))
    findings = data["findings"] if isinstance(data, dict) and "findings" in data else data
    if isinstance(findings, dict):
        findings = list(findings.values())
    return collections.Counter((f.get("tool"), f.get("type") or f.get("rule")) for f in findings)


def _tool_key(tool: str) -> str:
    return "error-prone" if tool == "error_prone" else tool


def count_of(source_name: Path, target: Path, top_count: int) -> dict:
    expected = json.loads((target / "sample_info.json").read_text(encoding="utf-8"))["tools"]
    in_sample = _sample(target)
    result = {"source_run": source_name.name, "sample_run": target.name, "tools": {}}

    for tool, name in NAMES.items():
        findings = _populacja(source_name, tool)
        key = _tool_key(tool)
        saved = expected[tool]["source_total_findings"]
        if len(findings) != saved:
            raise SystemExit(
                f"{name}: the filters yield {len(findings)} warnings while sample_info.json says "
                f"{saved} - the population filters have drifted apart"
            )

        counts = collections.Counter(f["type"] for f in findings)
        downloaded = expected[tool]["sampled_findings"]
        leading = []
        for rule, n in counts.most_common(top_count):
            leading.append({
                "rule": rule,
                "population": n,
                "population_share": n / len(findings),
                "proportional_draw": n / len(findings) * downloaded,
                "in_sample": in_sample[(key, rule)],
                "example_message": next(f["message"] for f in findings if f["type"] == rule),
            })

        result["tools"][tool] = {
            "name": name,
            "population": len(findings),
            "distinct_rules": len(counts),
            "sampled": downloaded,
            "top_rules": leading,
        }

    result["dominant"] = _dominant(result)
    result["s116_underscore"] = _s116_podkreslenie(source_name)
    return result


def _dominant(result: dict) -> dict:
    """Najliczniejszy wzorzec każdego narzędzia, zsumowany po trzech narzędziach."""
    selected, proportional, actual_value, downloaded = [], 0.0, 0, 0
    for data in result["tools"].values():
        head = data["top_rules"][0]
        rules = [head]
        if head["rule"] == "EI_EXPOSE_REP2":
            rules += [r for r in data["top_rules"] if r["rule"] == "EI_EXPOSE_REP"]
        for r in rules:
            selected.append(r["rule"])
            proportional += r["proportional_draw"]
            actual_value += r["in_sample"]
        downloaded += data["sampled"]
    return {
        "rules": selected,
        "proportional_count": proportional,
        "proportional_share": proportional / downloaded,
        "actual_count": actual_value,
        "actual_share": actual_value / downloaded,
        "sample_size": downloaded,
    }


def _s116_podkreslenie(source_name: Path) -> dict:
    findings = [f for f in _populacja(source_name, "sonarqube") if f["type"] == "java:S116"]
    underscore = [f for f in findings if re.search(r'Rename this field "_', f.get("message", ""))]
    return {
        "total": len(findings),
        "leading_underscore": len(underscore),
        "share": len(underscore) / len(findings) if findings else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", type=int, default=2)
    ap.add_argument("--sample-run", type=int, default=3)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    source_name, target = find_run_dir(args.source_run), find_run_dir(args.sample_run)
    result = count_of(source_name, target, args.top)

    directory = target / "analysis"
    directory.mkdir(exist_ok=True)
    path = directory / "population_concentration.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(path)
    for data in result["tools"].values():
        print(f"\n{data['name']}: {data['population']} warnings, {data['distinct_rules']} rules")
        for r in data["top_rules"]:
            print(f"  {r['population']:>5} ({r['population_share']:5.1%})  {r['rule']:<38}"
                  f" proportional {r['proportional_draw']:5.1f}, in sample {r['in_sample']}")

    d, s = result["dominant"], result["s116_underscore"]
    print(f"\njava:S116 with a leading underscore: {s['leading_underscore']} of {s['total']} ({s['share']:.1%})")
    print(f"najliczniejsze wzorce ({', '.join(d['rules'])}):")
    print(f"  proporcjonalnie {d['proportional_count']:.0f} z {d['sample_size']} "
          f"({d['proportional_share']:.1%}), faktycznie {d['actual_count']} ({d['actual_share']:.1%})")


if __name__ == "__main__":
    main()
