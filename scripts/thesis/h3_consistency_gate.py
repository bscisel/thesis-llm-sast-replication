#!/usr/bin/env python3
"""Bramka jednomyślności przebiegów i analiza tego, czy chwiejność jest cechą ostrzeżenia."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import RESAMPLING_SCHEME, Dataset, Finding, load_dataset
from analysis.metrics import Confusion, confusion
from analysis.resampling import DEFAULT_ITERATIONS, DEFAULT_SEED, Resampler, icc_oneway
from analysis.tests import fleiss_kappa


def _unanimous(finding: Finding, model: str) -> bool | None:
    decision = finding.decisions.get(model)
    if decision is None or decision.verdict is None:
        return None
    return decision.unanimous


def _scored(dataset: Dataset, model: str) -> list[Finding]:
    return [f for f in dataset.findings if _unanimous(f, model) is not None]


def _confusion_of(findings: Sequence[Finding], model: str) -> Confusion:
    return confusion((f.label, bool(f.decisions[model].verdict)) for f in findings)


def _gate_report(dataset: Dataset, model: str, iterations: int, seed: int) -> dict[str, Any]:
    scored = _scored(dataset, model)
    if not scored:
        return {"available": False}

    decided = [f for f in scored if f.decisions[model].unanimous]
    abstained = [f for f in scored if not f.decisions[model].unanimous]

    majority = _confusion_of(scored, model)
    gated = _confusion_of(decided, model) if decided else None

    resampler = Resampler(scored, scheme=RESAMPLING_SCHEME, seed=seed)

    def abstain_share(index: np.ndarray) -> float | None:
        if index.size == 0:
            return None
        picked = [scored[i] for i in index]
        return sum(1 for f in picked if not f.decisions[model].unanimous) / len(picked)

    def gated_metric(name: str):
        def statistic(index: np.ndarray) -> float | None:
            picked = [scored[i] for i in index if scored[i].decisions[model].unanimous]
            if not picked:
                return None
            return getattr(_confusion_of(picked, model), name)

        return statistic

    intervals = resampler.intervals(
        {
            "abstain_share": abstain_share,
            "precision": gated_metric("precision"),
            "recall": gated_metric("recall"),
            "f1": gated_metric("f1"),
            "accuracy": gated_metric("accuracy"),
        },
        iterations=iterations,
    )

    abstained_tp = sum(1 for f in abstained if f.label)
    return {
        "available": True,
        "scored": len(scored),
        "decided": len(decided),
        "abstained": len(abstained),
        "abstained_tp": abstained_tp,
        "abstained_fp": len(abstained) - abstained_tp,
        "tp_share_abstained": (abstained_tp / len(abstained)) if abstained else None,
        "tp_share_overall": sum(1 for f in scored if f.label) / len(scored),
        "majority": majority.as_dict(),
        "gated": gated.as_dict() if gated else None,
        "intervals": intervals,
    }


def _instability_matrix(dataset: Dataset) -> tuple[list[str], list[Finding], np.ndarray]:
    models = [m for m in dataset.models if any(_unanimous(f, m) is not None for f in dataset.findings)]
    findings = [f for f in dataset.findings if all(_unanimous(f, m) is not None for m in models)]
    matrix = np.array(
        [[0 if f.decisions[m].unanimous else 1 for m in models] for f in findings],
        dtype=int,
    )
    return models, findings, matrix


def _poisson_binomial(rates: Sequence[float]) -> np.ndarray:
    """Rozkład liczby chwiejnych modeli przy założeniu, że modele chwieją się niezależnie."""
    distribution = np.zeros(len(rates) + 1)
    distribution[0] = 1.0
    for rate in rates:
        shifted = np.zeros_like(distribution)
        shifted[1:] = distribution[:-1] * rate
        distribution = distribution * (1 - rate) + shifted
    return distribution


def _concentration(matrix: np.ndarray, iterations: int, seed: int) -> dict[str, Any]:
    counts = matrix.sum(axis=1)
    rates = matrix.mean(axis=0)
    expected = _poisson_binomial(rates) * len(matrix)
    observed = np.bincount(counts, minlength=matrix.shape[1] + 1).astype(float)

    rng = np.random.default_rng(seed)
    observed_var = float(np.var(counts))
    permuted = np.empty(iterations)
    shuffled = matrix.copy()
    for step in range(iterations):
        for column in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, column])
        permuted[step] = np.var(shuffled.sum(axis=1))
    p_value = float((np.sum(permuted >= observed_var) + 1) / (iterations + 1))

    return {
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "variance_observed": observed_var,
        "variance_expected_mean": float(np.mean(permuted)),
        "variance_ratio": observed_var / float(np.mean(permuted)) if np.mean(permuted) else None,
        "p_value": p_value,
        "permutations": iterations,
    }


def _hardness(models: Sequence[str], findings: Sequence[Finding], matrix: np.ndarray) -> list[dict[str, Any]]:
    counts = matrix.sum(axis=1)
    rows: list[dict[str, Any]] = []
    for k in range(matrix.shape[1] + 1):
        selected = [f for f, count in zip(findings, counts) if count == k]
        if not selected:
            continue
        correct = sum(
            1
            for f in selected
            for m in models
            if bool(f.decisions[m].verdict) == f.label
        )
        rows.append(
            {
                "unstable_models": k,
                "findings": len(selected),
                "tp_share": sum(1 for f in selected if f.label) / len(selected),
                "accuracy": correct / (len(selected) * len(models)),
            }
        )
    return rows


def _by_rule(findings: Sequence[Finding], matrix: np.ndarray) -> dict[str, Any]:
    counts = matrix.sum(axis=1) / matrix.shape[1]
    summary = icc_oneway(counts.tolist(), [f.rule for f in findings])
    per_rule: dict[str, list[int]] = {}
    for position, finding in enumerate(findings):
        per_rule.setdefault(finding.rule, []).append(position)
    ranked = [
        {
            "rule": rule,
            "findings": len(members),
            "mean_unstable_share": float(np.mean(counts[members])),
        }
        for rule, members in per_rule.items()
    ]
    ranked.sort(key=lambda row: (-row["mean_unstable_share"], -row["findings"]))
    return {
        "icc": summary.__dict__,
        "worst_rules": [row for row in ranked if row["findings"] >= 3][:8],
        "singleton_rules": sum(1 for row in ranked if row["findings"] < 3),
        "stable_rules": [row for row in ranked if row["mean_unstable_share"] == 0.0],
    }


def main() -> None:
    global RESAMPLING_SCHEME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument(
        "--cluster-by",
        choices=("plan", "rule", "stratum", "file", "none"),
        default="plan",
        help="bootstrap sampling unit; use 'rule' for the merged sample",
    )
    args = parser.parse_args()

    if args.cluster_by != "plan":
        RESAMPLING_SCHEME = args.cluster_by

    dataset = load_dataset(args.run_dir, include_partial=args.include_partial)

    gate = {model: _gate_report(dataset, model, args.iterations, args.seed) for model in dataset.models}
    models, findings, matrix = _instability_matrix(dataset)
    concentration = _concentration(matrix, args.permutations, args.seed)
    concentration["findings"] = len(findings)
    concentration["models"] = len(models)

    report = {
        "run": dataset.run_dir.name,
        "models": list(dataset.models),
        "settings": {"iterations": args.iterations, "permutations": args.permutations, "seed": args.seed,
                     "resampling": RESAMPLING_SCHEME},
        "gate": gate,
        "concentration": concentration,
        "fleiss": fleiss_kappa([[int(row.sum()), int(len(row) - row.sum())] for row in matrix]),
        "hardness": _hardness(models, findings, matrix),
        "by_rule": _by_rule(findings, matrix),
        "instability_rate": {m: float(matrix[:, i].mean()) for i, m in enumerate(models)},
    }

    out_dir = dataset.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "consistency.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nZapisano: {out_dir / 'consistency.json'}")


if __name__ == "__main__":
    main()
