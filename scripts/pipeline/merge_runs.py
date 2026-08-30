#!/usr/bin/env python3
"""Scala kilka przebiegów w jeden katalog, na którym działa reszta narzędzi."""

from __future__ import annotations

import argparse
import json
import sys
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analysis.runs import _finding_key
from analysis.constants import MODEL_DIRS, TOOL_FILES
RESULTS = PROJECT_ROOT / "results"
PROMPT_KEYS = ("system_prompt_variant", "system_prompt_sha256", "user_prompt_sha256")


def _resolve(spec: str) -> Path:
    path = Path(spec)
    if path.is_dir():
        return path
    matches = sorted(RESULTS.glob(f"{int(spec):03d}_*"))
    if not matches:
        raise SystemExit(f"Nie znaleziono runu: {spec}")
    return matches[-1]


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)




def _next_run_dir(name: str | None) -> Path:
    if name:
        return RESULTS / name
    numbers = [int(m.group(1)) for p in RESULTS.glob("[0-9][0-9][0-9]_*") if (m := re.match(r"(\d{3})_", p.name))]
    nxt = max(numbers, default=0) + 1
    return RESULTS / f"{nxt:03d}_{datetime.now():%Y-%m-%d_%H-%M-%S}"


def _merge_static(sources: list[Path], out: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[tuple[str, ...]] = set()
    for tool_file in TOOL_FILES:
        merged: dict[str, Any] | None = None
        findings: list[dict[str, Any]] = []
        for run in sources:
            path = run / "static_analysis" / "processed" / f"{tool_file}.json"
            if not path.exists():
                continue
            payload = _load(path)
            if merged is None:
                merged = {k: v for k, v in payload.items() if k != "findings"}
            for entry in payload["findings"]:
                key = _finding_key(str(payload["tool"]), entry)
                if key in seen:
                    raise SystemExit(f"Kolizja findingu miedzy runami: {key}")
                seen.add(key)
                findings.append(entry)
        if merged is None:
            continue
        merged["findings"] = findings
        merged["merged_from"] = [run.name for run in sources]
        _write(out / "static_analysis" / "processed" / f"{tool_file}.json", merged)
        counts[tool_file] = len(findings)
    return counts


def _teraz() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _merge_ground_truth(sources: list[Path], out: Path) -> int:
    merged: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = []
    for run in sources:
        payload = _load(run / "ground_truth.json")
        if merged is None:
            merged = {k: v for k, v in payload.items() if k != "findings"}
        findings.extend(payload["findings"])
    assert merged is not None
    merged["findings"] = findings
    merged["merged_from"] = [run.name for run in sources]
    merged["run"] = out.name
    merged["total_findings"] = len(findings)
    merged["labeled_findings"] = sum(1 for f in findings if f.get("is_true_positive") is not None)
    merged.pop("created_at", None)
    merged["merged_at"] = _teraz()
    liczniki: dict[str, int] = {}
    for wpis in findings:
        liczniki[wpis["tool"]] = liczniki.get(wpis["tool"], 0) + 1
    merged["reports"] = liczniki
    _write(out / "ground_truth.json", merged)
    return len(findings)


def _merge_categories(sources: list[Path], out: Path) -> int:
    rules: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    base: dict[str, Any] | None = None
    for run in sources:
        path = run / "category_ground_truth.json"
        if not path.exists():
            continue
        payload = _load(path)
        if base is None:
            base = {k: v for k, v in payload.items() if k not in ("rules", "overrides")}
        for key, value in payload.get("rules", {}).items():
            if key in rules and rules[key].get("category") != value.get("category"):
                raise SystemExit(f"Sprzeczna kategoria dla {key} miedzy runami")
            rules[key] = value
        overrides.update(payload.get("overrides", {}))
    if base is None:
        return 0
    base["rules"] = rules
    base["overrides"] = overrides
    base["merged_from"] = [run.name for run in sources]
    base["run"] = out.name
    base["total_rules"] = len(rules)
    base["labeled_rules"] = sum(1 for v in rules.values() if v.get("category"))
    base.pop("updated_at", None)
    base["merged_at"] = _teraz()
    _write(out / "category_ground_truth.json", base)
    return len(rules)


def _merge_llm(sources: list[Path], out: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model in MODEL_DIRS:
        for tool_file in TOOL_FILES:
            merged: dict[str, Any] | None = None
            findings: list[dict[str, Any]] = []
            prompts: dict[str, Any] | None = None
            present = 0
            for run in sources:
                path = run / "llm" / model / f"{tool_file}.json"
                if not path.exists():
                    continue
                present += 1
                payload = _load(path)
                current = {key: payload.get(key) for key in PROMPT_KEYS}
                if prompts is None:
                    prompts = current
                elif prompts != current:
                    raise SystemExit(
                        f"Różne prompty dla {model}/{tool_file}: {prompts} wobec {current}. "
                        "Scalanie takich wynikow dawaloby metryki z dwoch roznych pytan."
                    )
                if merged is None:
                    merged = {k: v for k, v in payload.items() if k != "findings"}
                findings.extend(payload.get("findings", []))
            if merged is None:
                continue
            if present != len(sources):
                raise SystemExit(
                    f"{model}/{tool_file} istnieje w {present} z {len(sources)} przebiegów — "
                    "scalanie niepelnego modelu ucialoby porownania parami."
                )
            for position, entry in enumerate(findings):
                entry["finding_id"] = position
            merged["findings"] = findings
            merged["merged_from"] = [run.name for run in sources]
            _write(out / "llm" / model / f"{tool_file}.json", merged)
            counts[f"{model}/{tool_file}"] = len(findings)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Scala runy w jeden katalog wynikowy.")
    parser.add_argument("--runs", required=True, help="numery albo sciezki po przecinku, np. 3,5")
    parser.add_argument("--output-run", default=None, help="nazwa katalogu wynikowego (domyslnie kolejny numer)")
    parser.add_argument("--force", action="store_true", help="nadpisz istniejacy katalog wynikowy")
    args = parser.parse_args()

    sources = [_resolve(spec.strip()) for spec in args.runs.split(",")]
    if len(sources) < 2:
        raise SystemExit("Podaj co najmniej dwa runy.")

    out = _next_run_dir(args.output_run)
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} juz istnieje — uzyj --force.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"Scalam: {', '.join(run.name for run in sources)}")
    static_counts = _merge_static(sources, out)
    gt_count = _merge_ground_truth(sources, out)
    cat_count = _merge_categories(sources, out)
    llm_counts = _merge_llm(sources, out)

    stages = []
    for run in sources:
        info_path = run / "sample_info.json"
        info = _load(info_path) if info_path.exists() else {}
        stages.append(
            {
                "run": run.name,
                "seed": info.get("seed"),
                "method": info.get("method", "stratified"),
                "findings": len(_load(run / "ground_truth.json")["findings"]),
            }
        )

    _write(
        out / "sample_info.json",
        {
            "method": "merge",
            "merged_from": [run.name for run in sources],
            "stages": stages,
            "note": (
                "Proba laczona z dwoch etapow o roznych schematach losowania. Etapy nalezy opisac "
                "osobno w metodyce; wyniki raportowane na calosci, rozbicie na etapy w zalaczniku."
            ),
            "resampling_note": (
                "Dla error-prone schemat 'stratum' przestaje byc sensowny: w drugim etapie wszystkie "
                "findingi wpadaja do warstwy '__remaining__' i zwinelyby sie w jeden klaster. "
                "Uzywaj --cluster-by rule. W pierwszym etapie oba podzialy sa identyczne "
                "(12 regul, 12 warstw), wiec zmiana nie rusza jego liczb."
            ),
        },
    )
    _write(
        out / "run_info.json",
        {
            "run": out.name,
            "kind": "merged",
            "merged_from": [run.name for run in sources],
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print(f"  static_analysis: {static_counts}")
    print(f"  ground_truth: {gt_count} findingow")
    print(f"  kategorie: {cat_count} regul")
    print(f"  llm: {len(llm_counts)} plikow, {sum(llm_counts.values())} rekordow")
    print(f"\nZapisano: {out}")
    print(f"Dalej: .venv/bin/python scripts/thesis/metrics_baselines.py --run-dir {out.name} --cluster-by rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
