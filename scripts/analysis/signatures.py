from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from analysis.runs import modal
from analysis.dataset import Dataset

PUNCTUATION = re.compile(r"[^a-z0-9 ]+")
SPACES = re.compile(r"\s+")


def normalize(signature: str) -> str:
    lowered = (signature or "").lower()
    return SPACES.sub(" ", PUNCTUATION.sub(" ", lowered)).strip()


def _modal_signature(values: tuple[str, ...]) -> str | None:
    return modal([normalize(value) for value in values if normalize(value)])


def _pairwise_identity(values: list[str]) -> tuple[int, int]:
    pairs = list(combinations(values, 2))
    return sum(1 for left, right in pairs if left == right), len(pairs)


def _pairwise_similarity(values: list[str]) -> list[float]:
    return [SequenceMatcher(None, left, right).ratio() for left, right in combinations(values, 2)]


PROG_BLISKOSCI = 0.80


def _cross_tool_merges(dataset: Dataset, modal: dict[tuple, str]) -> dict[str, Any]:
    """Ostrzeżenia różnych narzędzi zebrane przez model pod jedną sygnaturą."""
    tools: dict[str, set[str]] = defaultdict(set)
    covered: dict[str, int] = defaultdict(int)
    for finding in dataset.findings:
        signature = modal.get(finding.key)
        if signature:
            tools[signature].add(finding.tool)
            covered[signature] += 1
    merged = {sig: tool_set for sig, tool_set in tools.items() if len(tool_set) > 1}

    parent = {sig: sig for sig in tools}

    def root(sig: str) -> str:
        while parent[sig] != sig:
            parent[sig] = parent[parent[sig]]
            sig = parent[sig]
        return sig

    for left, right in combinations(sorted(tools), 2):
        if SequenceMatcher(None, left, right).ratio() >= PROG_BLISKOSCI:
            a, b = root(left), root(right)
            if a != b:
                parent[a] = b

    clusters: dict[str, list[str]] = defaultdict(list)
    for sig in sorted(tools):
        clusters[root(sig)].append(sig)
    close_pairs = [
        members for members in clusters.values()
        if len(set().union(*(tools[sig] for sig in members))) > 1
    ]

    weakest = [
        min(SequenceMatcher(None, left, right).ratio() for left, right in combinations(members, 2))
        for members in close_pairs if len(members) > 1
    ]

    return {
        "cross_tool_merges": {
            "signatures": len(merged),
            "findings_covered": sum(covered[sig] for sig in merged),
            "groups_at_similarity_threshold": len(close_pairs),
            "groups_closed_transitively": sum(1 for r in weakest if r < PROG_BLISKOSCI),
            "lowest_similarity_in_group": min(weakest) if weakest else None,
            "findings_covered_at_threshold": sum(
                covered[sig] for members in close_pairs for sig in members
            ),
            "examples": sorted(
                ({"signature": sig, "tools": sorted(tools), "findings": covered[sig]}
                 for sig, tools in merged.items()),
                key=lambda row: -row["findings"],
            )[:5],
        }
    }


def signature_report(dataset: Dataset) -> dict[str, Any]:
    report: dict[str, Any] = {"models": {}}

    deterministic_rule = len({finding.rule for finding in dataset.findings})
    deterministic_rule_file = len({(finding.rule, finding.sourcefile) for finding in dataset.findings})
    report["reference_grouping"] = {
        "findings": len(dataset.findings),
        "groups_by_rule": deterministic_rule,
        "groups_by_rule_and_file": deterministic_rule_file,
    }

    for model in dataset.models:
        modal: dict[tuple, str] = {}
        run_agreement: list[bool] = []
        for finding in dataset.findings:
            decision = finding.decisions.get(model)
            if decision is None or not decision.signatures:
                continue
            normalized = [normalize(value) for value in decision.signatures if normalize(value)]
            if not normalized:
                continue
            if len(normalized) > 1:
                run_agreement.append(len(set(normalized)) == 1)
            signature = _modal_signature(decision.signatures)
            if signature:
                modal[finding.key] = signature

        by_rule: dict[str, list[str]] = defaultdict(list)
        for finding in dataset.findings:
            signature = modal.get(finding.key)
            if signature:
                by_rule[finding.rule].append(signature)

        identical = 0
        close_pairs = 0
        pairs = 0
        similarities: list[float] = []
        rule_rows: list[dict[str, Any]] = []
        for rule, values in sorted(by_rule.items()):
            if len(values) < 2:
                continue
            same, total = _pairwise_identity(values)
            identical += same
            pairs += total
            rule_similarity = _pairwise_similarity(values)
            similarities.extend(rule_similarity)
            close_pairs += sum(1 for r in rule_similarity if r >= PROG_BLISKOSCI)
            rule_rows.append(
                {
                    "rule": rule,
                    "findings": len(values),
                    "distinct_signatures": len(set(values)),
                    "identical_pair_share": same / total if total else None,
                    "mean_similarity": (
                        sum(rule_similarity) / len(rule_similarity) if rule_similarity else None
                    ),
                }
            )

        report["models"][dataset.display_name(model)] = {
            "findings_with_signature": len(modal),
            "distinct_signatures": len(set(modal.values())),
            "unanimous_across_runs": (
                sum(run_agreement) / len(run_agreement) if run_agreement else None
            ),
            "within_rule_identical_pair_share": identical / pairs if pairs else None,
            "within_rule_close_pair_share": (identical + close_pairs) / pairs if pairs else None,
            "within_rule_pairs": pairs,
            "within_rule_mean_similarity": (
                sum(similarities) / len(similarities) if similarities else None
            ),
            "per_rule": rule_rows,
            **_cross_tool_merges(dataset, modal),
        }
    return report
