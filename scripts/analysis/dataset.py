from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.runs import _aggregate_runs, _finding_key, _resolve_run_dir, modal
from analysis.constants import ALL_TOOLS, CATEGORIES, MODEL_DIRS, TOOL_FILES, TOOL_ORDER

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Warstwy Error Prone były regułami, więc reguła nie może tam służyć za klaster losowania.
RESAMPLING_SCHEME = {"spotbugs": "rule", "sonarqube": "rule", "error-prone": "stratum"}


@dataclass
class Decision:
    model: str
    verdict: bool | None
    votes: tuple[bool, ...]
    categories: tuple[str, ...]
    signatures: tuple[str, ...]
    explanations: tuple[str, ...]
    reasonings: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    # Anthropic raportuje odczyt z pamięci podręcznej rozłącznie z input_tokens, OpenAI jako ich podzbiór.
    cache_read_tokens: int
    cached_within_input: int
    cache_write_tokens: int
    # Koszt wprost od dostawcy podaje wyłącznie OpenRouter; u pozostałych 0,0.
    reported_cost_usd: float
    latency_s: float
    valid_runs: int
    total_runs: int

    @property
    def unanimous(self) -> bool | None:
        if len(self.votes) < 2:
            return None
        return len(set(self.votes)) == 1

    @property
    def category(self) -> str | None:
        return modal(self.categories)


@dataclass
class Finding:
    key: tuple[str, str, str, str, str]
    tool: str
    type: str
    sourcefile: str
    start_line: str
    message: str
    description: str
    label: bool
    stratum: str
    reference_category: str | None
    metadata: dict[str, Any]
    source_path: str | None
    raw: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, Decision] = field(default_factory=dict)

    @property
    def rule(self) -> str:
        return f"{self.tool}::{self.type}"


@dataclass
class Dataset:
    run_dir: Path
    findings: list[Finding]
    models: tuple[str, ...]
    model_ids: dict[str, str]
    source_root: Path | None
    partial_models: dict[str, int] = field(default_factory=dict)

    def tools(self) -> tuple[str, ...]:
        seen = [tool for tool in TOOL_ORDER if any(f.tool == tool for f in self.findings)]
        return tuple(seen)

    def by_tool(self, tool: str | None = None) -> list[Finding]:
        if tool is None:
            return list(self.findings)
        return [f for f in self.findings if f.tool == tool]


    def display_name(self, model: str) -> str:
        return self.model_ids.get(model, model)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _strata_index(sample_info: dict[str, Any]) -> dict[str, Any]:
    tools = sample_info.get("tools", {})
    named: dict[str, Any] = {}
    for tool_file, entry in tools.items():
        strata = entry.get("strata", [])
        if tool_file == "error_prone":
            named["error-prone"] = {row["type"] for row in strata if "type" in row}
        elif tool_file == "spotbugs":
            named["spotbugs"] = {row["category"] for row in strata if "category" in row}
        elif tool_file == "sonarqube":
            named["sonarqube"] = {(row["issue_type"], row["severity"]) for row in strata}
    return named


def _stratum_of(tool: str, finding_type: str, metadata: dict[str, Any], named: dict[str, Any]) -> str:
    if tool == "error-prone":
        known = named.get("error-prone", set())
        return finding_type if finding_type in known else "__remaining__"
    if tool == "spotbugs":
        return str(metadata.get("category") or "__unknown__")
    if tool == "sonarqube":
        return f"{metadata.get('issue_type')}/{metadata.get('severity')}"
    return "__all__"


def _line_number(value: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _decision_from_runs(model: str, runs: list[dict[str, Any]]) -> Decision:
    verdict, stats = _aggregate_runs(runs)
    usable = [
        run
        for run in runs
        if isinstance(run.get("is_true_positive"), bool)
        and str(run.get("stop_reason") or "") not in {"max_tokens", "length", "MAX_TOKENS"}
    ]
    def _total(pole: str) -> int:
        return sum(int((run.get("usage") or {}).get(pole) or 0) for run in runs)

    input_tokens = _total("input_tokens")
    output_tokens = _total("output_tokens")
    cache_read_tokens = _total("cache_read_input_tokens")
    cached_within_input = _total("cached_input_tokens")
    cache_write_tokens = _total("cache_creation_input_tokens")
    latency = sum(float(run.get("latency_s") or 0.0) for run in runs)
    reported_cost = sum(float((run.get("usage") or {}).get("cost_usd") or 0.0) for run in runs)
    return Decision(
        model=model,
        verdict=verdict,
        votes=tuple(bool(run["is_true_positive"]) for run in usable),
        categories=tuple(str(run.get("category")) for run in usable if run.get("category")),
        signatures=tuple(str(run.get("issue_signature") or "") for run in usable),
        explanations=tuple(str(run.get("developer_explanation") or "") for run in usable),
        reasonings=tuple(str(run.get("reasoning") or "") for run in usable),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cached_within_input=cached_within_input,
        cache_write_tokens=cache_write_tokens,
        reported_cost_usd=reported_cost,
        latency_s=latency,
        valid_runs=stats["valid_runs"],
        total_runs=stats["total_runs"],
    )


def load_dataset(
    run_dir: Path | str | None = None,
    models: tuple[str, ...] = MODEL_DIRS,
    project_root: Path = PROJECT_ROOT,
    include_partial: bool = False,
) -> Dataset:
    resolved = _resolve_run_dir(project_root, Path(str(run_dir)) if run_dir is not None else None)

    ground_truth = _load_json(resolved / "ground_truth.json")
    categories_path = resolved / "category_ground_truth.json"
    reference_categories: dict[str, str] = {}
    if categories_path.exists():
        payload = _load_json(categories_path)
        for rule_key, entry in payload.get("rules", {}).items():
            reference_categories[rule_key] = entry["category"]
        for rule_key, entry in payload.get("overrides", {}).items():
            reference_categories[rule_key] = entry["category"]

    sample_info_path = resolved / "sample_info.json"
    named_strata = _strata_index(_load_json(sample_info_path)) if sample_info_path.exists() else {}

    raw_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for tool_file in TOOL_FILES:
        path = resolved / "static_analysis" / "processed" / f"{tool_file}.json"
        if not path.exists():
            continue
        payload = _load_json(path)
        tool = str(payload["tool"])
        for entry in payload["findings"]:
            raw_by_key[_finding_key(tool, entry)] = entry

    findings: list[Finding] = []
    index: dict[tuple[str, str, str, str, str], Finding] = {}
    for entry in ground_truth["findings"]:
        tool = str(entry["tool"])
        key = _finding_key(tool, entry)
        raw = raw_by_key.get(key, {})
        metadata = dict(raw.get("tool_specific") or {})
        finding = Finding(
            key=key,
            tool=tool,
            type=str(entry["type"]),
            sourcefile=str(entry["sourcefile"]),
            start_line=str(entry["start_line"]),
            message=str(entry.get("message") or ""),
            description=str(raw.get("description") or ""),
            label=bool(entry["is_true_positive"]),
            stratum=_stratum_of(tool, str(entry["type"]), metadata, named_strata),
            reference_category=reference_categories.get(f"{tool}::{entry['type']}"),
            metadata=metadata,
            source_path=str(raw.get("source_path") or "") or None,
            raw=raw,
        )
        findings.append(finding)
        index[key] = finding

    model_ids: dict[str, str] = {}
    source_root: Path | None = None
    for model in models:
        for tool_file in TOOL_FILES:
            path = resolved / "llm" / model / f"{tool_file}.json"
            if not path.exists():
                continue
            payload = _load_json(path)
            tool = str(payload["tool"])
            model_ids[model] = str(payload.get("model") or model)
            if source_root is None and payload.get("source_root"):
                source_root = Path(str(payload["source_root"]))
            for entry in payload.get("findings", []):
                key = _finding_key(tool, entry)
                finding = index.get(key)
                if finding is None:
                    continue
                finding.decisions[model] = _decision_from_runs(model, entry.get("runs", []))

    if source_root is not None and not source_root.exists():
        base = os.environ.get("SOURCE_ROOT")
        candidate = Path(base) / source_root if base else None
        source_root = candidate if candidate is not None and candidate.exists() else None

    coverage = {
        model: sum(
            1
            for finding in findings
            if finding.decisions.get(model) and finding.decisions[model].verdict is not None
        )
        for model in model_ids
    }
    partial = {model: count for model, count in coverage.items() if count < len(findings)}
    if partial and not include_partial:
        for model in partial:
            model_ids.pop(model, None)
            for finding in findings:
                finding.decisions.pop(model, None)

    order = {tool: position for position, tool in enumerate(TOOL_ORDER)}
    findings.sort(key=lambda f: (order.get(f.tool, 99), f.sourcefile, _line_number(f.start_line), f.type))
    present = tuple(model for model in models if model in model_ids)
    return Dataset(
        run_dir=resolved,
        findings=findings,
        models=present,
        model_ids=model_ids,
        source_root=source_root,
        partial_models=partial,
    )
