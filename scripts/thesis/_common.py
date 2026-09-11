#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset, Finding
from analysis.metrics import Confusion
from analysis.resampling import Resampler
from analysis.tests import (
    discordance,
    fleiss_kappa,
    mcnemar_cluster_signflip,
    mcnemar_exact,
)

RESAMPLING_SCHEME = "rule"


def complete_findings(dataset: Dataset) -> list[Finding]:
    return [
        finding
        for finding in dataset.findings
        if all(
            finding.decisions.get(model) and finding.decisions[model].verdict is not None
            for model in dataset.models
        )
    ]


def scored(findings: Sequence[Finding], model: str) -> list[Finding]:
    return [
        finding
        for finding in findings
        if finding.decisions.get(model) and finding.decisions[model].verdict is not None
    ]


def confusion(labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray) -> Confusion:
    picked_labels, picked_predictions = labels[indices], predictions[indices]
    return Confusion(
        tp=int(np.count_nonzero(picked_labels & picked_predictions)),
        fp=int(np.count_nonzero(~picked_labels & picked_predictions)),
        fn=int(np.count_nonzero(picked_labels & ~picked_predictions)),
        tn=int(np.count_nonzero(~picked_labels & ~picked_predictions)),
    )


def arrays(findings: Sequence[Finding], model: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([finding.label for finding in findings], dtype=bool)
    predictions = np.array([finding.decisions[model].verdict for finding in findings], dtype=bool)
    return labels, predictions


def raw_report_f1(labels: np.ndarray, indices: np.ndarray | None = None) -> float | None:
    picked = labels if indices is None else labels[indices]
    return confusion(picked, np.ones_like(picked, dtype=bool), np.arange(len(picked))).f1


def pairs(models: Sequence[str]) -> list[tuple[str, str]]:
    return [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]


def kappa_from_counts(counts: np.ndarray) -> float | None:
    if counts.size == 0:
        return None
    return fleiss_kappa(counts.tolist())["kappa"]


def compare_paired(
    findings: Sequence[Finding],
    hits: np.ndarray,
    reference: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    resampler = Resampler(list(findings), seed=seed)
    interval = resampler.interval(
        lambda idx, a=hits, b=reference: float(a[idx].mean() - b[idx].mean()),
        iterations=iterations,
    )
    rules = [finding.rule for finding in findings]
    b, c = discordance(list(hits), list(reference))
    return {
        "difference": interval["point"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "mcnemar_cluster": mcnemar_cluster_signflip(list(hits), list(reference), rules, seed=seed),
        "mcnemar_exact": mcnemar_exact(b, c),
    }
