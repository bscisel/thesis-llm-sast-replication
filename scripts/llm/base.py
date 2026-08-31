"""Wspólna podstawa analizatorów opartych na modelach językowych."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

MAX_FULL_FILE_LINES = 200
CONTEXT_RADIUS_LINII_PRZED_I_PO = 150

MAX_DEGENERATE_CONTEXT_SHARE = 0.05

CATEGORIES = [
    "CORRECTNESS",
    "CONCURRENCY",
    "SECURITY",
    "PERFORMANCE",
    "MAINTAINABILITY",
    "OTHER",
]

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "is_true_positive": {"type": "boolean"},
        "confidence": {"type": "number"},
        "category": {"type": "string", "enum": CATEGORIES},
        "issue_signature": {"type": "string"},
        "developer_explanation": {"type": "string"},
    },
    "required": [
        "reasoning",
        "is_true_positive",
        "confidence",
        "category",
        "issue_signature",
        "developer_explanation",
    ],
    "additionalProperties": False,
}


def _load_template(
    tool: str, system_prompt_variant: str | None = None, bez_kodu: bool = False
) -> tuple[str, str]:
    """Zwraca szablony promptu systemowego i użytkownika dla danego narzędzia."""
    sufiks = "_b3" if bez_kodu else ""
    system_path = _PROMPTS_DIR / f"system_prompt{sufiks}.txt"
    if system_prompt_variant:
        variant_path = _PROMPTS_DIR / f"system_prompt_{system_prompt_variant}{sufiks}.txt"
        if variant_path.exists():
            system_path = variant_path
    tool_specific_path = _PROMPTS_DIR / f"user_prompt_{tool.replace('-', '_')}{sufiks}.txt"
    generic_path = _PROMPTS_DIR / f"user_prompt{sufiks}.txt"

    system_prompt = system_path.read_text(encoding="utf-8").strip()

    if tool_specific_path.exists():
        user_template = tool_specific_path.read_text(encoding="utf-8").strip()
    else:
        user_template = generic_path.read_text(encoding="utf-8").strip()

    return system_prompt, user_template


def _resolve_java_path(
    finding: dict[str, Any], source_root: Path, sourcefile: str
) -> Path | None:
    """Odnajduje plik Javy, którego dotyczy ostrzeżenie, w katalogu źródeł."""
    source_path = finding.get("source_path")

    if source_path:
        direct = source_root / source_path
        if direct.exists():
            return direct

        suffix = source_path.replace("\\", "/")
        by_suffix = [
            path
            for path in source_root.rglob(sourcefile)
            if path.as_posix().endswith(suffix)
        ]
        if len(by_suffix) == 1:
            return by_suffix[0]
        if by_suffix:
            logger.warning(
                "Ambiguous source path '%s': %d files match, using %s.",
                source_path,
                len(by_suffix),
                by_suffix[0],
            )
            return by_suffix[0]

    matches = sorted(source_root.rglob(sourcefile))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Resolved '%s' by file name alone (%d candidates, reported path %r); "
            "using %s.",
            sourcefile,
            len(matches),
            source_path,
            matches[0],
        )
    return matches[0]


def _build_code_context(finding: dict[str, Any], source_root: Path) -> str:
    """Buduje wycinek kodu pokazywany modelowi razem z ostrzeżeniem."""
    sourcefile = finding.get("sourcefile")
    if not sourcefile:
        return "(source file unknown)"

    java_path = _resolve_java_path(finding, source_root, sourcefile)
    if java_path is None:
        return f"(source file '{sourcefile}' not found under {source_root})"
    try:
        all_lines = java_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return f"(could not read '{java_path}': {exc})"

    total = len(all_lines)

    specific = finding.get("tool_specific") or {}
    try:
        flag_start = int(finding.get("start_line") or 0)
        flag_end = int(finding.get("end_line") or flag_start)
    except (TypeError, ValueError):
        flag_start = flag_end = 0

    if flag_start <= 0:
        try:
            flag_start = int(specific.get("class_start_line") or 0)
            flag_end = int(specific.get("class_end_line") or flag_start)
        except (TypeError, ValueError):
            flag_start = flag_end = 0

    if flag_start <= 0 and total > MAX_FULL_FILE_LINES:
        return (
            f"(no line number reported for '{sourcefile}' "
            f"({total} lines); no context can be selected)"
        )
    if flag_start > total:
        return (
            f"(reported line {flag_start} is past the end of '{sourcefile}' "
            f"({total} lines))"
        )

    if total <= MAX_FULL_FILE_LINES:
        slice_start = 0
        slice_end = total
    else:
        slice_start = max(0, flag_start - 1 - CONTEXT_RADIUS_LINII_PRZED_I_PO)
        slice_end = min(total, flag_end + CONTEXT_RADIUS_LINII_PRZED_I_PO)

    lines_out: list[str] = []
    for idx in range(slice_start, slice_end):
        lineno = idx + 1
        marker = ">>>" if flag_start <= lineno <= flag_end else "   "
        lines_out.append(f"{lineno:5} |{marker} {all_lines[idx]}")

    if slice_end - slice_start < total:
        header = f"// {sourcefile}: showing lines {slice_start + 1}-{slice_end} of {total}"
    else:
        header = f"// {sourcefile}: complete file, {total} lines"
    return "\n".join([header, *lines_out])


def _flatten_tool_specific(finding: dict[str, Any]) -> dict[str, str]:
    """Wypłaszcza pola tool_specific do pojedynczych podstawień w szablonie."""
    flat: dict[str, str] = {}
    for key, value in (finding.get("tool_specific") or {}).items():
        if isinstance(value, list):
            flat[key] = ", ".join(str(v) for v in value) if value else ""
        else:
            flat[key] = "" if value is None else str(value)
    return flat


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    """Wydobywa obiekt JSON z odpowiedzi modelu."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    parsed = _loads_object(raw)
    if parsed is not None:
        return parsed

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        parsed = _loads_object(raw[start : end + 1])
        if parsed is not None:
            return parsed
    return _repair_object(raw)


def _repair_object(raw: str) -> dict[str, Any] | None:
    """Domyka obiekt, którego model nie zakończył."""
    start = raw.find("{")
    if start == -1:
        return None

    fragment = raw[start:]
    in_string = False
    escaped = False
    depth = 0
    for char in fragment:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

    if depth <= 0:
        return None
    repaired = _loads_object(fragment + ('"' if in_string else "") + "}" * depth)
    if repaired is None or not isinstance(repaired.get("is_true_positive"), bool):
        return None
    return repaired


class FatalAPIError(RuntimeError):
    """Błąd, który powtórzyłby się przy każdym kolejnym wywołaniu."""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _assert_same_prompts(
    existing_output: dict[str, Any], prompts: dict[str, Any], output_path: Path
) -> None:
    """Odmawia dopisania powtórzeń wykonanych innym promptem."""
    recorded = {key: existing_output.get(key) for key in prompts}
    if any(value is not None for value in recorded.values()) and recorded != prompts:
        raise ValueError(
            f"Refusing to append runs to {output_path}: the existing results were "
            f"produced with a different prompt ({recorded}) than the current one "
            f"({prompts})."
        )


def _load_partial(path: Path, prompts: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Wczytuje ostrzeżenia rozstrzygnięte przed przerwanym uruchomieniem."""
    items: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring unreadable line %d in %s (interrupted mid-write).",
                    number,
                    path.name,
                )
                continue
            if record.get("_header"):
                recorded = {key: record.get(key) for key in prompts}
                if recorded != prompts:
                    raise ValueError(
                        f"Refusing to resume {path}: it was produced with a different "
                        f"prompt ({recorded}) than the current one ({prompts}). "
                        f"Delete the file to start over."
                    )
                continue
            try:
                finding_id = int(record["finding_id"])
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring malformed record on line %d in %s.", number, path.name)
                continue
            runs = record.get("runs") or []
            if any(run.get("error") == "api_error" for run in runs):
                logger.warning(
                    "Finding %d in %s has a failed API call - it will be redone.",
                    finding_id,
                    path.name,
                )
                continue
            items[finding_id] = record
    return items


def _assert_context_usable(
    degenerate: list[str], total: int, findings_path: Path
) -> None:
    """Przerywa przed pierwszym płatnym wywołaniem, gdy katalog źródeł nie jest czytany."""
    if not degenerate or not total:
        return
    share = len(degenerate) / total
    listing = "\n  ".join(degenerate[:10])
    if share > MAX_DEGENERATE_CONTEXT_SHARE:
        raise RuntimeError(
            f"{len(degenerate)}/{total} findings in {findings_path} have no usable "
            f"source context ({share:.0%} > {MAX_DEGENERATE_CONTEXT_SHARE:.0%}). "
            f"Check --source-root and the report paths. First cases:\n  {listing}"
        )
    logger.warning(
        "%d/%d findings have no usable source context:\n  %s",
        len(degenerate),
        total,
        listing,
    )


def _resolve_temperature(env_value: str | None) -> float | None:
    """Zamienia wartość LLM_TEMPERATURE na liczbę albo brak ustawienia."""
    if not env_value or env_value.strip().upper() == "DEFAULT":
        return None
    try:
        return float(env_value)
    except ValueError:
        logger.warning("Invalid LLM_TEMPERATURE value '%s', using model default.", env_value)
        return None


class LLMAnalyzer(ABC):
    """Podstawa analizatorów: wspólny przebieg wywołania, ponowienia i telemetria."""

    def __init__(
        self,
        source_root: Path,
        num_runs: int = 3,
        temperature: float | None = None,
        concurrency: int = 1,
        bez_kodu: bool = False,
    ) -> None:
        self.source_root = Path(source_root)
        self.num_runs = num_runs
        self.bez_kodu = bez_kodu
        self.temperature = temperature
        self.concurrency = max(1, concurrency)
        self._meta_store = threading.local()

    @property
    def _last_meta(self) -> dict[str, Any]:
        return getattr(self._meta_store, "value", {})

    @_last_meta.setter
    def _last_meta(self, value: dict[str, Any]) -> None:
        self._meta_store.value = value


    def analyze_tool(
        self,
        findings_path: Path,
        output_path: Path,
        *,
        dry_run: bool = False,
        append_runs: bool = False,
        overwrite: bool = False,
        system_prompt_variant: str | None = None,
        limit: int | None = None,
    ) -> None:
        """Przetwarza wszystkie ostrzeżenia z pliku wejściowego i zapisuje wyniki."""
        with open(findings_path, encoding="utf-8") as f:
            data = json.load(f)

        tool = data["tool"]
        system_prompt, user_template = _load_template(tool, system_prompt_variant, self.bez_kodu)
        prompts = {
            "system_prompt_variant": system_prompt_variant,
            "system_prompt_sha256": _digest(system_prompt),
            "user_prompt_sha256": _digest(user_template),
        }

        existing_output: dict[str, Any] = {}
        existing_findings: dict[int, dict[str, Any]] = {}
        if output_path.exists() and not dry_run:
            if append_runs:
                with output_path.open(encoding="utf-8") as f:
                    existing_output = json.load(f)
                _assert_same_prompts(existing_output, prompts, output_path)
                existing_findings = {
                    int(item["finding_id"]): item
                    for item in existing_output.get("findings", [])
                    if str(item.get("finding_id", "")).isdigit()
                }
                logger.info("Appending runs to existing results: %s", output_path)
            elif not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing LLM results: {output_path}. "
                    "Use --append-runs to add more runs or --overwrite to replace them."
                )

        results: list[dict[str, Any]] = []

        all_findings = data.get("findings", [])
        if limit is not None:
            all_findings = all_findings[:limit]
            logger.info("Limiting to the first %d finding(s).", len(all_findings))

        prepared: list[tuple[int, dict[str, Any], str, str]] = []
        degenerate: list[str] = []
        for idx, finding in enumerate(all_findings):
            if self.bez_kodu:
                code_context = "(code fragment intentionally omitted in this run)"
            else:
                code_context = _build_code_context(finding, self.source_root)
            if not self.bez_kodu and code_context.startswith("("):
                degenerate.append(
                    f"#{idx} {finding.get('sourcefile')}:{finding.get('start_line')} "
                    f"→ {code_context}"
                )
            system_msg, user_msg = self.build_prompt(
                finding, code_context, system_prompt, user_template
            )

            if dry_run:
                self._print_dry_run(idx, finding, system_msg, user_msg)
                continue

            prepared.append((idx, finding, system_msg, user_msg))

        _assert_context_usable(degenerate, len(all_findings), findings_path)

        if dry_run:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(output_path.name + ".partial.jsonl")
        partial_lock = threading.Lock()

        completed: dict[int, dict[str, Any]] = {}
        if partial_path.exists():
            if overwrite:
                partial_path.unlink()
            else:
                completed = _load_partial(partial_path, prompts)
                prepared = [job for job in prepared if job[0] not in completed]
                logger.info(
                    "Resuming: %d finding(s) already done, %d left.",
                    len(completed),
                    len(prepared),
                )
        if not partial_path.exists():
            with partial_path.open("w", encoding="utf-8") as handle:
                header = {"_header": True, "model": self.model_id, "tool": tool, **prompts}
                handle.write(json.dumps(header, ensure_ascii=False) + "\n")

        def process(job: tuple[int, dict[str, Any], str, str]) -> dict[str, Any]:
            idx, finding, system_msg, user_msg = job
            existing_item = existing_findings.get(idx, {})
            existing_runs = list(existing_item.get("runs", []))
            start_run = _next_run_number(existing_runs)
            new_runs = self._run_finding(system_msg, user_msg, start_run=start_run)

            item = {
                "finding_id": idx,
                "type": finding.get("type"),
                "sourcefile": finding.get("sourcefile"),
                "start_line": finding.get("start_line"),
                "message": finding.get("message"),
                "tool_specific": {
                    "column": (finding.get("tool_specific") or {}).get("column")
                },
                "runs": existing_runs + new_runs,
            }
            if "original_finding_index" in finding:
                item["original_finding_index"] = finding["original_finding_index"]

            with partial_lock, partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return item

        done = 0
        if self.concurrency > 1:
            logger.info(
                "Processing %d finding(s) with %d concurrent workers.",
                len(prepared),
                self.concurrency,
            )
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                for item in pool.map(process, prepared):
                    results.append(item)
                    done += 1
                    logger.info("Finding %d/%d processed.", done, len(prepared))
        else:
            for job in prepared:
                results.append(process(job))
                done += 1
                logger.info("Finding %d/%d processed.", done, len(prepared))

        results.extend(completed.values())
        results.sort(key=lambda item: item["finding_id"])

        now = datetime.now(timezone.utc).isoformat()
        output = {
            "model": self.model_id,
            "tool": tool,
            "num_runs": max((len(item.get("runs", [])) for item in results), default=0),
            "runs_per_invocation": self.num_runs,
            "source_root": str(self.source_root),
            "call_parameters": self.call_parameters(),
            **prompts,
            "timestamp": existing_output.get("timestamp", now),
            "updated_at": now,
            "findings": results,
        }
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, output_path)
        partial_path.unlink(missing_ok=True)
        logger.info("Results written to %s", output_path)


    def call_parameters(self) -> dict[str, Any]:
        """Ustawienia samego wywołania API, zapisywane w nagłówku wyniku."""
        return {
            "temperature": self.temperature,
            "concurrency": self.concurrency,
            "response_format": "json_schema",
        }

    def build_prompt(
        self,
        finding: dict[str, Any],
        code_context: str,
        system_prompt: str,
        user_template: str,
    ) -> tuple[str, str]:
        """Buduje prompt systemowy i użytkownika dla jednego ostrzeżenia."""
        specific = finding.get("tool_specific") or {}
        start_line = finding.get("start_line") or specific.get("class_start_line") or ""
        end_line = finding.get("end_line") or specific.get("class_end_line") or ""

        placeholders: dict[str, str] = {
            "tool": finding.get("tool", ""),
            "type": finding.get("type") or "",
            "abbrev": finding.get("abbrev") or "",
            "message": finding.get("message") or "",
            "description": finding.get("description") or "",
            "sourcefile": finding.get("sourcefile") or "",
            "start_line": start_line,
            "end_line": end_line,
            "code_context": code_context,
        }
        placeholders.update(_flatten_tool_specific(finding))

        user_msg = user_template.format_map(_DefaultDict(placeholders))
        return system_prompt, user_msg


    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identyfikator modelu zapisywany w pliku wynikowym."""

    @abstractmethod
    def call_api(self, system_message: str, user_message: str) -> str:
        """Wysyła prompt do modelu i zwraca surową odpowiedź tekstową."""


    def is_fatal_error(self, exc: Exception) -> str | None:
        """Zwraca powód, gdy błąd powtórzyłby się przy każdym kolejnym wywołaniu."""
        return None

    def _run_finding(
        self, system_msg: str, user_msg: str, *, start_run: int = 1
    ) -> list[dict[str, Any]]:
        """Wykonuje zadaną liczbę wywołań dla jednego ostrzeżenia."""
        runs: list[dict[str, Any]] = []
        for run_num in range(start_run, start_run + self.num_runs):
            self._last_meta = {}
            started = time.perf_counter()
            try:
                raw = self.call_api(system_msg, user_msg)
                meta = self._last_meta
                meta["latency_s"] = round(time.perf_counter() - started, 3)
                parsed = _parse_json_response(raw)
                if parsed is not None:
                    parsed["run"] = run_num
                    parsed.update(meta)
                    runs.append(parsed)
                else:
                    logger.warning(
                        "Run %d: could not parse JSON response. Raw: %.200s",
                        run_num,
                        raw,
                    )
                    runs.append(
                        {"run": run_num, "error": "parse_failed", "raw": raw, **meta}
                    )
            except Exception as exc:  # noqa: BLE001
                fatal = self.is_fatal_error(exc)
                if fatal is not None:
                    raise FatalAPIError(fatal) from exc
                logger.warning("Run %d: API call failed: %s", run_num, exc)
                runs.append(
                    {
                        "run": run_num,
                        "error": "api_error",
                        "message": str(exc),
                        "latency_s": round(time.perf_counter() - started, 3),
                    }
                )
        return runs

    @staticmethod
    def _print_dry_run(
        idx: int,
        finding: dict[str, Any],
        system_msg: str,
        user_msg: str,
    ) -> None:
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"FINDING #{idx}  |  {finding.get('sourcefile')}:{finding.get('start_line')}  |  {finding.get('type')}")
        print(sep)
        print("── SYSTEM ──")
        print(system_msg)
        print("── USER ──")
        print(user_msg)


class _DefaultDict(dict):
    """Słownik zwracający pusty napis dla brakujących podstawień w format_map."""

    def __missing__(self, key: str) -> str:  # noqa: D105
        return ""


def _next_run_number(runs: list[dict[str, Any]]) -> int:
    numbers: list[int] = []
    for run in runs:
        try:
            numbers.append(int(run.get("run", 0)))
        except (TypeError, ValueError):
            continue
    return max(numbers, default=0) + 1
