from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from analysis.runs import modal
from analysis.dataset import CATEGORIES, Dataset
from analysis.tests import fleiss_kappa


TOOL_LABEL = {
    ("spotbugs", "CORRECTNESS"): "CORRECTNESS",
    ("spotbugs", "MT_CORRECTNESS"): "CONCURRENCY",
    ("spotbugs", "SECURITY"): "SECURITY",
    ("spotbugs", "PERFORMANCE"): "PERFORMANCE",
    ("spotbugs", "MALICIOUS_CODE"): "SECURITY",
    ("sonarqube", "BUG"): "CORRECTNESS",
    ("sonarqube", "VULNERABILITY"): "SECURITY",
    ("sonarqube", "CODE_SMELL"): "MAINTAINABILITY",
}


def tool_label_baseline(dataset: Dataset) -> dict[str, Any]:
    """Ile trafia mechaniczne przełożenie etykiety narzędzia na taksonomię odniesienia."""
    hit_items = covered = 0
    covered_findings = []
    benchmark_keys: set = set()
    for finding in dataset.findings:
        specific = (finding.raw or {}).get("tool_specific") or {}
        label = specific.get("category") if finding.tool == "spotbugs" else specific.get("issue_type")
        mapped = TOOL_LABEL.get((finding.tool, label))
        if mapped is None or finding.reference_category is None:
            continue
        covered += 1
        covered_findings.append(finding)
        if mapped == finding.reference_category:
            hit_items += 1
            benchmark_keys.add(finding.key)
    models_on_subset: dict[str, Any] = {}
    for model in dataset.models:
        evaluated = matched = 0
        matched_keys: set = set()
        for finding in covered_findings:
            decision = finding.decisions.get(model)
            if decision is None or not decision.categories:
                continue
            predicted = modal(decision.categories)
            if predicted is None:
                continue
            evaluated += 1
            if predicted == finding.reference_category:
                matched += 1
                matched_keys.add(finding.key)
        if evaluated:
            models_on_subset[dataset.display_name(model)] = {
                "scored": evaluated,
                "accuracy": matched / evaluated,
                "hits_shared_with_label": len(matched_keys & benchmark_keys),
                "hits_model_only": len(matched_keys - benchmark_keys),
                "hits_label_only": len(benchmark_keys - matched_keys),
            }
    return {
        "with_unambiguous_label": covered,
        "correct": hit_items,
        "accuracy": hit_items / covered if covered else None,
        "without_label": len(dataset.findings) - covered,
        "models_on_same_subset": models_on_subset,
    }


def category_report(dataset: Dataset) -> dict[str, Any]:
    scoped = [finding for finding in dataset.findings if finding.reference_category]
    report: dict[str, Any] = {
        "findings_with_reference": len(scoped),
        "models": {},
        "agreement_between_models": None,
    }

    for model in dataset.models:
        matrix: dict[str, Counter] = defaultdict(Counter)
        per_tool: dict[str, list[bool]] = defaultdict(list)
        correct = 0
        scored = 0
        run_stability: list[bool] = []
        for finding in scoped:
            decision = finding.decisions.get(model)
            if decision is None or not decision.categories:
                continue
            predicted = modal(decision.categories)
            if predicted is None:
                continue
            scored += 1
            hit = predicted == finding.reference_category
            correct += int(hit)
            matrix[finding.reference_category][predicted] += 1
            per_tool[finding.tool].append(hit)
            run_stability.append(len(set(decision.categories)) == 1)

        report["models"][dataset.display_name(model)] = {
            "scored": scored,
            "accuracy": correct / scored if scored else None,
            "accuracy_per_tool": {
                tool: sum(hits) / len(hits) for tool, hits in sorted(per_tool.items()) if hits
            },
            "unanimous_across_runs": (
                sum(run_stability) / len(run_stability) if run_stability else None
            ),
            "confusion": {
                reference: {predicted: matrix[reference][predicted] for predicted in CATEGORIES}
                for reference in CATEGORIES
            },
        }

    summed: dict[str, Counter] = {reference: Counter() for reference in CATEGORIES}
    for data in report["models"].values():
        for reference, row in data["confusion"].items():
            summed[reference].update(row)
    report["confusion_summed"] = {
        reference: {predicted: summed[reference][predicted] for predicted in CATEGORIES}
        for reference in CATEGORIES
    }
    report["errors_summed"] = sum(
        count for reference in CATEGORIES for predicted, count in
        report["confusion_summed"][reference].items() if predicted != reference
    )

    ratings: list[list[int]] = []
    for finding in scoped:
        row = [0] * len(CATEGORIES)
        raters = 0
        for model in dataset.models:
            decision = finding.decisions.get(model)
            if decision is None or not decision.categories:
                continue
            predicted = modal(decision.categories)
            if predicted in CATEGORIES:
                row[CATEGORIES.index(predicted)] += 1
                raters += 1
        if raters == len(dataset.models):
            ratings.append(row)
    if ratings:
        report["agreement_between_models"] = fleiss_kappa(ratings)

    distribution = {
        dataset.display_name(model): Counter(
            modal(finding.decisions[model].categories)
            for finding in scoped
            if finding.decisions.get(model) and finding.decisions[model].categories
        )
        for model in dataset.models
    }
    report["distribution"] = {
        model: {category: counts.get(category, 0) for category in CATEGORIES}
        for model, counts in distribution.items()
    }
    report["reference_distribution"] = {
        category: sum(1 for finding in scoped if finding.reference_category == category)
        for category in CATEGORIES
    }
    report["tool_label_baseline"] = tool_label_baseline(dataset)
    return report
