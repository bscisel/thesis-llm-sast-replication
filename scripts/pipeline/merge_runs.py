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
    counters: dict[str, int] = {}
    for entry_line in findings:
        counters[entry_line["tool"]] = counters.get(entry_line["tool"], 0) + 1
    merged["reports"] = counters
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
                        f"Different prompts for {model}/{tool_file}: {prompts} versus {current}. "
                        "Scalanie takich wynikow dawaloby metryki z dwoch roznych pytan."
                    )
                if merged is None:
                    merged = {k: v for k, v in payload.items() if k != "findings"}
                findings.extend(payload.get("findings", []))
            if merged is None:
                continue
            if present != len(sources):
                raise SystemExit(
                    f"{model}/{tool_file} exists in {present} of {len(sources)} runs - "
                    "scalanie niepelnego modelu ucialoby porownania parami."
                )
            for position, entry in enumerate(findings):
                entry["finding_id"] = position
            merged["findings"] = findings
            merged["merged_from"] = [run.name for run in sources]
            _write(out / "llm" / model / f"{tool_file}.json", merged)
            counts[f"{model}/{tool_file}"] = len(findings)
    return counts


def keep_unmerged_files(snapshot: Path, out: Path, produced: set[str]) -> list[Path]:
    """Wklada z powrotem to, czego scalanie nie produkuje — takze z wnetrza llm/ i static_analysis/."""
    kept: list[Path] = []
    for item in sorted(snapshot.iterdir()):
        if item.name not in produced:
            target = out / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
            kept.append(target)
            continue
        if not item.is_dir():
            continue
        for old_file in sorted(item.rglob("*")):
            if old_file.is_dir():
                continue
            target = out / old_file.relative_to(snapshot)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_file, target)
            kept.append(target)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Scala runy w jeden katalog wynikowy.")
    parser.add_argument("--runs", required=True, help="numbers or paths, comma separated, e.g. 3,5")
    parser.add_argument("--output-run", default=None, help="nazwa katalogu wynikowego (domyslnie kolejny numer)")
    parser.add_argument("--force", action="store_true", help="nadpisz istniejacy katalog wynikowy")
    args = parser.parse_args()

    sources = [_resolve(spec.strip()) for spec in args.runs.split(",")]
    if len(sources) < 2:
        raise SystemExit("Podaj co najmniej dwa runy.")

    out = _next_run_dir(args.output_run)
    kept: list[Path] = []
    snapshot: Path | None = None
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} juz istnieje — uzyj --force.")
        # Scalanie odtwarza run ze zrodel, wiec wszystko, czego samo nie produkuje — reczne
        # kodowania, raporty z lektury, przeliczone analizy — przepadloby razem z katalogiem.
        # Stary katalog idzie wiec na bok, a pliki spoza listy produkowanej wracaja po scaleniu.
        snapshot = out.with_name(f"{out.name}.przed_scaleniem_{datetime.now():%Y-%m-%d_%H-%M-%S}")
        out.rename(snapshot)
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

    if snapshot is not None:
        produced = {
            "sample_info.json",
            "run_info.json",
            "ground_truth.json",
            "category_ground_truth.json",
            "static_analysis",
            "llm",
        }
        kept = keep_unmerged_files(snapshot, out, produced)

        # Poprawki naniesione po scaleniu zylyby wylacznie w runie wyjsciowym, wiec scalanie
        # ze zrodel cofneloby je bez sladu. Nie nadpisujemy ich w ciszy: rozjazd trafia na ekran,
        # a poprzedni plik zostaje obok do porownania.
        for name in ("category_ground_truth.json", "run_info.json", "ground_truth.json"):
            old_file = snapshot / name
            if not old_file.exists():
                continue
            old_item, new_item = _load(old_file), _load(out / name)
            if name == "run_info.json":
                extra_items = {k: v for k, v in old_item.items() if k not in new_item}
                if extra_items:
                    new_item.update(extra_items)
                    _write(out / name, new_item)
                    print(f"  {name}: przeniesiono z poprzedniej wersji {', '.join(sorted(extra_items))}")
                continue
            entry_key = "rules" if name == "category_ground_truth.json" else "findings"
            old_entries, new_entries = old_item.get(entry_key), new_item.get(entry_key)
            if isinstance(old_entries, dict):
                differing = [k for k in old_entries if old_entries[k] != (new_entries or {}).get(k)]
            else:
                # ground_truth.json trzyma etykiety w liscie, a nie pod kluczem — porownanie idzie
                # po identyfikatorze findingu, bo kolejnosc po scaleniu nie musi byc ta sama.
                po_id = {w.get("finding_id"): w for w in (new_entries or [])}
                differing = [
                    w.get("finding_id")
                    for w in (old_entries or [])
                    if w != po_id.get(w.get("finding_id"))
                ]
            if differing:
                copy_path = out / f"{Path(name).stem}.przed_scaleniem.json"
                shutil.copy2(old_file, copy_path)
                print(
                    f"  UWAGA {name}: {len(differing)} wpisow rozni sie od poprzedniej wersji tego runu "
                    f"({', '.join(differing[:5])}{', ...' if len(differing) > 5 else ''}). "
                    f"Poprawki naniesione po poprzednim scaleniu NIE przenosza sie ze zrodel — "
                    f"poprzedni plik zostawiono jako {copy_path.name}."
                )

    print(f"  static_analysis: {static_counts}")
    print(f"  ground_truth: {gt_count} findingow")
    print(f"  kategorie: {cat_count} regul")
    print(f"  llm: {len(llm_counts)} plikow, {sum(llm_counts.values())} rekordow")
    if kept:
        print(f"  zachowane spoza scalania: {', '.join(sorted(p.name for p in kept))}")
    if snapshot is not None:
        print(f"  poprzedni katalog: {snapshot.name} (usun recznie, gdy wynik bedzie sprawdzony)")
    print(f"\nZapisano: {out}")
    print(f"Dalej: .venv/bin/python scripts/thesis/metrics_baselines.py --run-dir {out.name} --cluster-by rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
