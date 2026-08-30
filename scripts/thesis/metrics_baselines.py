#!/usr/bin/env python3
"""Metryki postprocessingu LLM z przedziałami ufności i punktami odniesienia B0-B2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Finding, load_dataset
from analysis.constants import ALL_TOOLS
from thesis._common import _confusion
from analysis.metrics import (
    Confusion,
    baseline_predictor,
    evaluate,
    metadata_candidate_report,
    model_predictor,
    wilson_interval,
)
from analysis.resampling import DEFAULT_ITERATIONS, DEFAULT_SEED, Resampler, icc_oneway

BASELINES = ("B0", "B1", "B2")


def _statistics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Callable[[np.ndarray], float | None]]:
    def make(attribute: str) -> Callable[[np.ndarray], float | None]:
        def statistic(indices: np.ndarray) -> float | None:
            return getattr(_confusion(labels, predictions, indices), attribute)

        return statistic

    return {name: make(name) for name in ("precision", "recall", "f1", "accuracy", "fp_reduction", "lost_tp_share")}


def _row(
    scope: str,
    approach: str,
    label: str,
    findings: Sequence[Finding],
    predictor: Callable[[Finding], bool],
    iterations: int,
    seed: int,
    cluster_by: dict[str, str] | str,
) -> dict[str, Any]:
    labels = np.array([finding.label for finding in findings], dtype=bool)
    predictions = np.array([predictor(finding) for finding in findings], dtype=bool)
    counts = evaluate(findings, predictor)

    resampler = Resampler(findings, scheme=cluster_by, seed=seed)
    intervals = resampler.intervals(_statistics(labels, predictions), iterations=iterations)

    row: dict[str, Any] = {
        "tool": scope,
        "approach": approach,
        "label": label,
        "n": len(findings),
        **counts.as_dict(),
    }
    for name, interval in intervals.items():
        row[f"{name}_ci_low"] = interval["ci_low"]
        row[f"{name}_ci_high"] = interval["ci_high"]

    precision_wilson = wilson_interval(counts.tp, counts.tp + counts.fp)
    recall_wilson = wilson_interval(counts.tp, counts.tp + counts.fn)
    row["precision_wilson_low"] = precision_wilson[0] if precision_wilson else None
    row["precision_wilson_high"] = precision_wilson[1] if precision_wilson else None
    row["recall_wilson_low"] = recall_wilson[0] if recall_wilson else None
    row["recall_wilson_high"] = recall_wilson[1] if recall_wilson else None
    return row


def _scopes(dataset: Any, split_by_context: bool) -> list[tuple[str, list[Finding]]]:
    scopes: list[tuple[str, list[Finding]]] = [(tool, dataset.by_tool(tool)) for tool in dataset.tools()]
    scopes.append((ALL_TOOLS, list(dataset.findings)))
    if split_by_context and dataset.source_root and dataset.source_root.exists():
        from analysis.grounding import context_kind

        grouped: dict[str, list[Finding]] = defaultdict(list)
        for finding in dataset.findings:
            grouped[context_kind(finding, dataset.source_root)].append(finding)
        for kind in ("pełny plik", "wycinek", "brak kontekstu"):
            if grouped.get(kind):
                scopes.append((kind, grouped[kind]))
    return scopes


def _per_rule_rows(dataset: Any) -> list[dict[str, Any]]:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in dataset.findings:
        grouped[finding.rule].append(finding)

    rows: list[dict[str, Any]] = []
    for rule in sorted(grouped):
        members = grouped[rule]
        base = {
            "rule": rule,
            "tool": members[0].tool,
            "type": members[0].type,
            "n": len(members),
            "tp_share": sum(1 for finding in members if finding.label) / len(members),
        }
        for model in dataset.models:
            scored = [finding for finding in members if finding.decisions[model].verdict is not None]
            counts = evaluate(scored, model_predictor(model))
            prefix = model
            base[f"{prefix}_n"] = len(scored)
            base[f"{prefix}_tp"] = counts.tp
            base[f"{prefix}_fp"] = counts.fp
            base[f"{prefix}_fn"] = counts.fn
            base[f"{prefix}_tn"] = counts.tn
            base[f"{prefix}_precision"] = counts.precision
            base[f"{prefix}_recall"] = counts.recall
            base[f"{prefix}_f1"] = counts.f1
        rows.append(base)
    return rows


def _cluster_report(dataset: Any) -> dict[str, Any]:
    report: dict[str, Any] = {}
    values = [1.0 if finding.label else 0.0 for finding in dataset.findings]
    for name, key in (("reguła", lambda f: f.rule), ("plik", lambda f: f"{f.tool}::{f.sourcefile}")):
        summary = icc_oneway(values, [key(finding) for finding in dataset.findings])
        report[name] = {
            "clusters": summary.clusters,
            "mean_size": summary.mean_size,
            "n0": summary.n0,
            "icc": summary.icc,
            "design_effect": summary.design_effect,
            "effective_n": summary.effective_n,
        }
    per_tool: dict[str, Any] = {}
    for tool in dataset.tools():
        subset = dataset.by_tool(tool)
        summary = icc_oneway(
            [1.0 if finding.label else 0.0 for finding in subset],
            [finding.rule for finding in subset],
        )
        per_tool[tool] = {
            "clusters": summary.clusters,
            "mean_size": summary.mean_size,
            "n0": summary.n0,
            "icc": summary.icc,
            "design_effect": summary.design_effect,
            "effective_n": summary.effective_n,
        }
    report["per_tool_rule"] = per_tool

    per_model: dict[str, Any] = {}
    for model in dataset.models:
        pairs = [
            (finding, finding.decisions[model])
            for finding in dataset.findings
            if finding.decisions.get(model) and finding.decisions[model].verdict is not None
        ]
        if not pairs:
            continue
        summary = icc_oneway(
            [1.0 if decision.verdict == finding.label else 0.0 for finding, decision in pairs],
            [finding.rule for finding, _ in pairs],
        )
        per_model[dataset.display_name(model)] = {
            "clusters": summary.clusters,
            "icc": summary.icc,
            "design_effect": summary.design_effect,
            "effective_n": summary.effective_n,
        }
    report["per_model_correctness_rule"] = per_model
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["narzędzie", "podejście", "n", "TP", "FP", "FN", "TN", "precision [CI]", "recall [CI]", "F1 [CI]", "red. FP", "utracone TP"]
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    row["tool"],
                    row["approach"],
                    str(row["n"]),
                    str(row["tp"]),
                    str(row["fp"]),
                    str(row["fn"]),
                    str(row["tn"]),
                    f"{_fmt(row['precision'])} [{_fmt(row['precision_ci_low'])};{_fmt(row['precision_ci_high'])}]",
                    f"{_fmt(row['recall'])} [{_fmt(row['recall_ci_low'])};{_fmt(row['recall_ci_high'])}]",
                    f"{_fmt(row['f1'])} [{_fmt(row['f1_ci_low'])};{_fmt(row['f1_ci_high'])}]",
                    _fmt(row["fp_reduction"]),
                    str(row["lost_tp"]),
                ]
            )
        )



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="3")
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="policz też modele bez kompletu werdyktów (domyślnie pomijane, bo zawężają wspólną próbę)",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cluster-by",
        choices=("plan", "rule", "file", "none"),
        default="plan",
        help="plan = klastrowanie po regule, dla Error Prone losowanie w warstwach",
    )
    parser.add_argument(
        "--no-context-split",
        action="store_true",
        help="pomiń wiersze w podziale na ostrzeżenia z pełnym plikiem i z wycinkiem",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(args.run_dir, include_partial=args.include_partial)
    if dataset.partial_models:
        for model, count in dataset.partial_models.items():
            stan = "policzony" if args.include_partial else "POMINIĘTY"
            print(f"UWAGA: {model} ma {count}/{len(dataset.findings)} werdyktów — {stan}.")
    scheme: dict[str, str] | str
    scheme = "rule" if args.cluster_by == "plan" else args.cluster_by
    if args.cluster_by == "plan":
        from analysis.dataset import RESAMPLING_SCHEME

        scheme = RESAMPLING_SCHEME

    rows: list[dict[str, Any]] = []
    baseline_labels: dict[str, str] = {}
    for scope, findings in _scopes(dataset, not args.no_context_split):
        for name in BASELINES:
            predictor, description = baseline_predictor(name, dataset.findings)
            baseline_labels[name] = description
            rows.append(
                _row(scope, name, description, findings, predictor, args.iterations, args.seed, scheme)
            )
        for model in dataset.models:
            scored = [finding for finding in findings if finding.decisions[model].verdict is not None]
            rows.append(
                _row(
                    scope,
                    dataset.display_name(model),
                    f"postprocessing LLM ({model})",
                    scored,
                    model_predictor(model),
                    args.iterations,
                    args.seed,
                    scheme,
                )
            )

    output_dir = args.output_dir or dataset.run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run": dataset.run_dir.name,
        "models": {model: dataset.display_name(model) for model in dataset.models},
        "resampling": {
            "scheme": args.cluster_by,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "baselines": baseline_labels,
        "b2_candidates": metadata_candidate_report(dataset.findings),
        "clustering": _cluster_report(dataset),
        "rows": rows,
        "per_rule": _per_rule_rows(dataset),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    _print_table(rows)
    print()
    for name, description in baseline_labels.items():
        print(f"{name}: {description}")
    print(f"\nZapisano: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
