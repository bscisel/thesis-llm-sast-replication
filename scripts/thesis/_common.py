#!/usr/bin/env python3
"""Wspólne pomocniki skryptów liczących liczby do pracy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset, Finding
from analysis.constants import ALL_TOOLS
from analysis.metrics import Confusion
from analysis.tests import fleiss_kappa

def _scopes(dataset: Dataset) -> list[tuple[str, list[Finding]]]:
    return [(tool, dataset.by_tool(tool)) for tool in dataset.tools()] + [
        (ALL_TOOLS, list(dataset.findings))
    ]


def _arrays(findings: Sequence[Finding], model: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([finding.label for finding in findings], dtype=bool)
    predictions = np.array([finding.decisions[model].verdict for finding in findings], dtype=bool)
    return labels, predictions


def _confusion(labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray) -> Confusion:
    picked_labels, picked_predictions = labels[indices], predictions[indices]
    return Confusion(
        tp=int(np.count_nonzero(picked_labels & picked_predictions)),
        fp=int(np.count_nonzero(~picked_labels & picked_predictions)),
        fn=int(np.count_nonzero(picked_labels & ~picked_predictions)),
        tn=int(np.count_nonzero(~picked_labels & ~picked_predictions)),
    )


def _scored(findings: Sequence[Finding], model: str) -> list[Finding]:
    return [
        finding
        for finding in findings
        if finding.decisions.get(model) and finding.decisions[model].verdict is not None
    ]


def _safe_share(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _b0_f1(labels: np.ndarray, indices: np.ndarray | None = None) -> float | None:
    picked = labels if indices is None else labels[indices]
    return _confusion(picked, np.ones_like(picked, dtype=bool), np.arange(len(picked))).f1


def _delta_f1(labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray) -> float | None:
    model_f1 = _confusion(labels, predictions, indices).f1
    baseline_f1 = _b0_f1(labels, indices)
    if model_f1 is None or baseline_f1 is None:
        return None
    return model_f1 - baseline_f1


def _pairs(models: Sequence[str]) -> list[tuple[str, str]]:
    return [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]


def _accuracy_spread(findings: Sequence[Finding], model: str) -> dict[str, Any] | None:
    """Trafność liczona osobno z każdego przebiegu i różnica maksimum-minimum."""
    sizes = [len(finding.decisions[model].votes) for finding in findings]
    if not sizes:
        return None
    runs = max(sizes)
    if runs < 2:
        return None
    complete = [f for f in findings if len(f.decisions[model].votes) == runs]
    if not complete:
        return None
    labels = np.array([f.label for f in complete], dtype=bool)
    per_run = [
        float(np.mean(np.array([f.decisions[model].votes[i] for f in complete], dtype=bool) == labels))
        for i in range(runs)
    ]
    return {
        "findings": len(complete),
        "dropped_incomplete": len(findings) - len(complete),
        "runs": runs,
        "accuracy_per_run": per_run,
        "min": min(per_run),
        "max": max(per_run),
        "spread": max(per_run) - min(per_run),
    }


def _kappa_from_counts(counts: np.ndarray) -> float | None:
    if counts.size == 0:
        return None
    result = fleiss_kappa(counts.tolist())
    return result["kappa"]
