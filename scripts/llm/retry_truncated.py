#!/usr/bin/env python3
"""Ponawia wywołania, które nie dały werdyktu."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from analysis.constants import TRUNCATION_REASONS
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import llm.run_analysis as ra
from llm.base import _build_code_context, _digest, _load_template

logger = logging.getLogger(__name__)



def _failure_kind(run: dict[str, Any]) -> str | None:
    if str(run.get("stop_reason") or "") in TRUNCATION_REASONS:
        return "ucięty"
    if run.get("error") == "api_error":
        return "błąd API"
    if run.get("error") == "parse_failed" and not str(run.get("raw") or "").strip():
        return "pusta odpowiedź"
    return None


def _failed_slots(data: dict[str, Any]) -> list[tuple[int, int, str]]:
    slots = []
    for f_idx, finding in enumerate(data.get("findings", [])):
        for r_idx, run in enumerate(finding.get("runs", [])):
            kind = _failure_kind(run)
            if kind:
                slots.append((f_idx, r_idx, kind))
    return slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ra.MODELS)
    parser.add_argument("--tool", required=True, choices=ra.TOOLS)
    parser.add_argument("--run-dir")
    parser.add_argument("--source-root")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Wypisuje tylko to, co zostałoby powtórzone.")
    parser.add_argument(
        "--bez-kodu",
        action="store_true",
        help="Powtorka w wariancie ablacyjnym B3 (H6): prompty z sufiksem _b3, bez fragmentu "
             "zrodla. Musi byc podana dla przebiegu, ktory tak powstal — inaczej skroty promptow "
             "sie nie zgodza i skrypt odmowi powtorki.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    ra._load_env()

    project_root = ra._project_root()
    run_dir = ra._resolve_run_dir(project_root, args.run_dir)
    result_path = run_dir / "llm" / args.model / ra.TOOL_FILENAMES[args.tool]
    findings_path = run_dir / "static_analysis" / "processed" / ra.TOOL_FILENAMES[args.tool]

    data = json.loads(result_path.read_text(encoding="utf-8"))
    slots = _failed_slots(data)
    if not slots:
        logger.info("Nothing to repeat in %s.", result_path.name)
        return

    source_data = json.loads(findings_path.read_text(encoding="utf-8"))
    source_findings = source_data["findings"]
    variant = ra._resolve_system_prompt(None, run_dir)
    system_prompt, user_template = _load_template(source_data["tool"], variant, args.bez_kodu)

    for key, digest in (
        ("system_prompt_sha256", _digest(system_prompt)),
        ("user_prompt_sha256", _digest(user_template)),
    ):
        if data.get(key) != digest:
            logger.error(
                "Prompt zmienił się od czasu przebiegu (%s: %s w pliku, %s teraz). Odmowa powtórzenia.",
                key,
                data.get(key),
                digest,
            )
            sys.exit(1)

    logger.info("%d nieudanych powtórzeń w %s:", len(slots), result_path.name)
    for f_idx, r_idx, kind in slots:
        finding = data["findings"][f_idx]
        logger.info(
            "  finding %s %s:%s run %s (%s)",
            finding.get("finding_id"),
            finding.get("sourcefile"),
            finding.get("start_line"),
            finding["runs"][r_idx].get("run"),
            kind,
        )
    if args.dry_run:
        return

    source_root = ra._resolve_source_root(args.source_root, run_dir)
    analyzer = ra._build_analyzer(
        args.model,
        {"source_root": source_root, "num_runs": 1, "temperature": ra._resolve_temperature(os.getenv("LLM_TEMPERATURE")), "concurrency": 1, "bez_kodu": args.bez_kodu},
    )

    backup = result_path.with_name(
        f"{result_path.stem}.przed_powtorka_{datetime.now(timezone.utc):%Y-%m-%d_%H-%M-%S}.json"
    )
    if backup.exists():
        logger.error("Kopia %s już istnieje. Odmowa nadpisania.", backup.name)
        sys.exit(1)
    shutil.copy2(result_path, backup)
    logger.info("Backup: %s", backup.name)

    repaired = failed = 0
    for f_idx, r_idx, _kind in slots:
        finding = data["findings"][f_idx]
        finding_id = finding["finding_id"]
        source = source_findings[finding_id] if finding_id < len(source_findings) else None
        if source is None or (
            source.get("sourcefile"),
            source.get("start_line"),
            source.get("type"),
        ) != (finding.get("sourcefile"), finding.get("start_line"), finding.get("type")):
            logger.error(
                "Ostrzeżenie %s nie zgadza się z %s na tej pozycji — pominięte.",
                finding_id,
                findings_path.name,
            )
            failed += 1
            continue

        kontekst = ("(code fragment intentionally omitted in this run)" if args.bez_kodu
                    else _build_code_context(source, source_root))
        system_msg, user_msg = analyzer.build_prompt(
            source, kontekst, system_prompt, user_template
        )
        original = finding["runs"][r_idx]
        run_number = original.get("run")

        for attempt in range(1, args.max_attempts + 1):
            (result,) = analyzer._run_finding(system_msg, user_msg, start_run=run_number)
            if isinstance(result.get("is_true_positive"), bool) and _failure_kind(result) is None:
                result["retry_of_truncated"] = True
                result["retry_attempts"] = attempt
                result["superseded"] = original
                finding["runs"][r_idx] = result
                repaired += 1
                logger.info(
                    "Ostrzeżenie %s, powtórzenie %s naprawione za %d podejściem.", finding_id, run_number, attempt
                )
                break
            logger.warning(
                "Ostrzeżenie %s, powtórzenie %s, podejście %d: stop_reason=%s błąd=%s — ponawiam.",
                finding_id,
                run_number,
                attempt,
                result.get("stop_reason"),
                result.get("error"),
            )
        else:
            failed += 1
            logger.error(
                "Ostrzeżenie %s, powtórzenie %s nadal nieudane po %d podejściach — zostaje jako brak danych.",
                finding_id,
                run_number,
                args.max_attempts,
            )

    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    history = data.setdefault("retry_history", [])
    history.append(
        {
            "at": data["updated_at"],
            "reason": "failed_runs",
            "kinds": sorted({kind for _, _, kind in slots}),
            "attempted": len(slots),
            "repaired": repaired,
            "still_missing": failed,
            "max_attempts": args.max_attempts,
            "configuration_changed": False,
        }
    )

    tmp = result_path.with_name(result_path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, result_path)
    logger.info("Naprawiono %d/%d; %d zostaje jako brak danych.", repaired, len(slots), failed)


if __name__ == "__main__":
    main()
