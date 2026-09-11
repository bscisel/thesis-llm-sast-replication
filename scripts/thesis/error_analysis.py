#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.dataset import Dataset
from thesis._common import RESAMPLING_SCHEME, scored

DESCRIPTION = "Wrong verdicts: how many were unanimous, and which findings every model missed"


def error_unanimity(dataset: Dataset) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in dataset.models:
        unanimous = split = 0
        for finding in scored(dataset.findings, model):
            decision = finding.decisions[model]
            if decision.verdict == finding.label or len(decision.votes) < 2:
                continue
            if len(set(decision.votes)) == 1:
                unanimous += 1
            else:
                split += 1
        errors = unanimous + split
        out[dataset.display_name(model)] = {
            "errors": errors,
            "unanimous": unanimous,
            "split": split,
            "unanimous_share": unanimous / errors if errors else None,
        }
    return out


def unanimous_misses(dataset: Dataset) -> dict[str, Any]:
    rows = []
    for finding in dataset.findings:
        judged = [
            m
            for m in dataset.models
            if finding.decisions.get(m) and finding.decisions[m].verdict is not None
        ]
        if len(judged) < len(dataset.models):
            continue
        if all(finding.decisions[m].verdict != finding.label for m in judged):
            rows.append(
                {
                    "rule": finding.rule,
                    "tool": finding.tool,
                    "file": finding.sourcefile,
                    "line": finding.start_line,
                    "label": finding.label,
                }
            )
    return {
        "count": len(rows),
        "by_rule": dict(Counter(row["rule"] for row in rows).most_common()),
        "list": rows,
    }


def rejected_true_positives(dataset: Dataset, models: list[str]) -> dict[str, Any]:
    keys = {
        finding.key
        for finding in dataset.findings
        if finding.label
        for model in models
        if finding.decisions.get(model) and finding.decisions[model].verdict is False
    }
    return {"models": models, "count": len(keys)}


def main() -> None:
    import thesis._cli as cli

    parser = cli.parser(DESCRIPTION)
    parser.add_argument("--rejecting-models", nargs="*", default=["qwen", "oss20"])
    args = parser.parse_args()

    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["error_unanimity"] = error_unanimity(dataset)
    results["unanimous_misses"] = unanimous_misses(dataset)
    results["rejected_true_positives"] = rejected_true_positives(
        dataset, [m for m in args.rejecting_models if m in dataset.models]
    )
    cli.save(cli.output_dir(args, dataset), "error_analysis.json", results)


if __name__ == "__main__":
    main()
