"""Skupienie ostrzeżeń w regułach: populacja runu 002 wobec próby runu 003."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

NAMES = {"spotbugs": "SpotBugs", "error_prone": "Error Prone", "sonarqube": "SonarQube"}
MODUL_WYKLUCZONY = "jetty-ee-test-resources"


def _populacja(zrodlo: Path, tool: str) -> list[dict]:
    path = zrodlo / "static_analysis" / "processed" / f"{tool}.json"
    findings = json.loads(path.read_text(encoding="utf-8"))["findings"]
    return [
        f for f in findings
        if f.get("source_set") != "test" and MODUL_WYKLUCZONY not in (f.get("source_path") or "")
    ]


def _sample(target: Path) -> collections.Counter:
    data = json.loads((target / "ground_truth.json").read_text(encoding="utf-8"))
    findings = data["findings"] if isinstance(data, dict) and "findings" in data else data
    if isinstance(findings, dict):
        findings = list(findings.values())
    return collections.Counter((f.get("tool"), f.get("type") or f.get("rule")) for f in findings)


def _tool_key(tool: str) -> str:
    return "error-prone" if tool == "error_prone" else tool


def policz(zrodlo: Path, target: Path, top_count: int) -> dict:
    expected = json.loads((target / "sample_info.json").read_text(encoding="utf-8"))["tools"]
    w_probie = _sample(target)
    result = {"source_run": zrodlo.name, "sample_run": target.name, "tools": {}}

    for tool, name in NAMES.items():
        findings = _populacja(zrodlo, tool)
        key = _tool_key(tool)
        zapisane = expected[tool]["source_total_findings"]
        if len(findings) != zapisane:
            raise SystemExit(
                f"{name}: filtry dają {len(findings)} ostrzeżeń, a sample_info.json mówi "
                f"o {zapisane} — filtry populacji się rozjechały"
            )

        licznosci = collections.Counter(f["type"] for f in findings)
        pobrano = expected[tool]["sampled_findings"]
        czolowe = []
        for rule, n in licznosci.most_common(top_count):
            czolowe.append({
                "rule": rule,
                "population": n,
                "population_share": n / len(findings),
                "proportional_draw": n / len(findings) * pobrano,
                "in_sample": w_probie[(key, rule)],
                "example_message": next(f["message"] for f in findings if f["type"] == rule),
            })

        result["tools"][tool] = {
            "name": name,
            "population": len(findings),
            "distinct_rules": len(licznosci),
            "sampled": pobrano,
            "top_rules": czolowe,
        }

    result["dominant"] = _dominant(result)
    result["s116_underscore"] = _s116_podkreslenie(zrodlo)
    return result


def _dominant(result: dict) -> dict:
    """Najliczniejszy wzorzec każdego narzędzia, zsumowany po trzech narzędziach."""
    wybrane, proporcjonalnie, faktycznie, pobrano = [], 0.0, 0, 0
    for data in result["tools"].values():
        czolo = data["top_rules"][0]
        rules = [czolo]
        if czolo["rule"] == "EI_EXPOSE_REP2":
            rules += [r for r in data["top_rules"] if r["rule"] == "EI_EXPOSE_REP"]
        for r in rules:
            wybrane.append(r["rule"])
            proporcjonalnie += r["proportional_draw"]
            faktycznie += r["in_sample"]
        pobrano += data["sampled"]
    return {
        "rules": wybrane,
        "proportional_count": proporcjonalnie,
        "proportional_share": proporcjonalnie / pobrano,
        "actual_count": faktycznie,
        "actual_share": faktycznie / pobrano,
        "sample_size": pobrano,
    }


def _s116_podkreslenie(zrodlo: Path) -> dict:
    findings = [f for f in _populacja(zrodlo, "sonarqube") if f["type"] == "java:S116"]
    podkreslenie = [f for f in findings if re.search(r'Rename this field "_', f.get("message", ""))]
    return {
        "total": len(findings),
        "leading_underscore": len(podkreslenie),
        "share": len(podkreslenie) / len(findings) if findings else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", type=int, default=2)
    ap.add_argument("--sample-run", type=int, default=3)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    zrodlo, target = find_run_dir(args.source_run), find_run_dir(args.sample_run)
    result = policz(zrodlo, target, args.top)

    directory = target / "analysis"
    directory.mkdir(exist_ok=True)
    path = directory / "population_concentration.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(path)
    for data in result["tools"].values():
        print(f"\n{data['name']}: {data['population']} ostrzeżeń, {data['distinct_rules']} reguł")
        for r in data["top_rules"]:
            print(f"  {r['population']:>5} ({r['population_share']:5.1%})  {r['rule']:<38}"
                  f" proporcjonalnie {r['proportional_draw']:5.1f}, w próbie {r['in_sample']}")

    d, s = result["dominant"], result["s116_underscore"]
    print(f"\njava:S116 z podkreśleniem: {s['leading_underscore']} z {s['total']} ({s['share']:.1%})")
    print(f"najliczniejsze wzorce ({', '.join(d['rules'])}):")
    print(f"  proporcjonalnie {d['proportional_count']:.0f} z {d['sample_size']} "
          f"({d['proportional_share']:.1%}), faktycznie {d['actual_count']} ({d['actual_share']:.1%})")


if __name__ == "__main__":
    main()
