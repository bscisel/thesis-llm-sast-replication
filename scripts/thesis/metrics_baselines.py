#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.resampling import icc_oneway
from analysis.dataset import Dataset
from thesis._common import RESAMPLING_SCHEME, confusion

DESCRIPTION = "Headline metrics for every model, the raw report and the naive classifier"


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    counts = confusion(labels, predictions, np.arange(len(labels)))
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "tn": counts.tn,
        "fn": counts.fn,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "accuracy": counts.accuracy,
    }


def overview(dataset: Dataset) -> dict[str, Any]:
    findings = dataset.findings
    labels = np.array([f.label for f in findings], dtype=bool)

    rows: dict[str, Any] = {}
    for model in dataset.models:
        predictions = np.array(
            [bool(f.decisions[model].verdict) for f in findings], dtype=bool
        )
        rows[dataset.display_name(model)] = _metrics(labels, predictions)
    rows["raw_report"] = _metrics(labels, np.ones_like(labels, dtype=bool))
    rows["naive_classifier"] = _metrics(labels, np.zeros_like(labels, dtype=bool))
    return rows


def clustering(dataset: Dataset) -> dict[str, Any]:
    summary = icc_oneway(
        [1.0 if f.label else 0.0 for f in dataset.findings],
        [f.rule for f in dataset.findings],
    )
    return {
        "clusters": summary.clusters,
        "mean_size": summary.mean_size,
        "icc": summary.icc,
    }


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(DESCRIPTION).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["findings"] = len(dataset.findings)
    results["true_positives"] = sum(1 for f in dataset.findings if f.label)
    results["overview"] = overview(dataset)
    results["clustering"] = clustering(dataset)
    cli.save(cli.output_dir(args, dataset), "metrics.json", results)


if __name__ == "__main__":
    main()
