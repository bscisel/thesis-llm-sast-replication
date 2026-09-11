#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset
from analysis.resampling import Resampler
from thesis._common import RESAMPLING_SCHEME, kappa_from_counts, pairs

DESCRIPTION = "H3 - verdict stability across the three repetitions"
RUNS_PER_FINDING = 3


def _per_model(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in dataset.models:
        usable = [
            f
            for f in dataset.findings
            if f.decisions.get(model) and len(f.decisions[model].votes) == RUNS_PER_FINDING
        ]
        if not usable:
            continue
        votes = [f.decisions[model].votes for f in usable]
        unanimity = np.array([len(set(vote)) == 1 for vote in votes], dtype=bool)
        counts = np.array([[v.count(False), v.count(True)] for v in votes], dtype=float)

        resampler = Resampler(usable, seed=seed)
        intervals = resampler.intervals(
            {
                "unanimity": lambda i: float(np.mean(unanimity[i])),
                "kappa": lambda i: kappa_from_counts(counts[i]),
            },
            iterations=iterations,
        )
        out[dataset.display_name(model)] = {
            "n": len(usable),
            "unanimity": float(np.mean(unanimity)),
            "unanimity_ci": [intervals["unanimity"]["ci_low"], intervals["unanimity"]["ci_high"]],
            "fleiss_kappa": kappa_from_counts(counts),
            "fleiss_kappa_ci": [intervals["kappa"]["ci_low"], intervals["kappa"]["ci_high"]],
        }
    return out


def _pairwise(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    findings = list(dataset.findings)
    complete, unanimous = {}, {}
    for model in dataset.models:
        decisions = [f.decisions.get(model) for f in findings]
        complete[model] = np.array(
            [bool(d and len(d.votes) == RUNS_PER_FINDING) for d in decisions]
        )
        unanimous[model] = np.array([bool(d and d.votes and len(set(d.votes)) == 1) for d in decisions])

    resampler = Resampler(findings, seed=seed)
    draws = [resampler.draw() for _ in range(iterations)]

    details, shown = {}, 0
    for first, second in pairs(list(dataset.models)):
        common = complete[first] & complete[second]
        if not common.any():
            continue
        differences = []
        for indexes in draws:
            selected = indexes[common[indexes]]
            if selected.size:
                differences.append(
                    float(unanimous[first][selected].mean() - unanimous[second][selected].mean())
                )
        if not differences:
            continue
        ordered = np.sort(np.array(differences))
        low, high = float(np.quantile(ordered, 0.025)), float(np.quantile(ordered, 0.975))
        proven = low > 0.0 or high < 0.0
        shown += proven
        details[f"{dataset.display_name(first)} vs {dataset.display_name(second)}"] = {
            "n": int(common.sum()),
            "difference": float(
                unanimous[first][common].mean() - unanimous[second][common].mean()
            ),
            "difference_ci": [low, high],
            "proven": bool(proven),
        }
    return {"pairs": len(details), "differences_shown": shown, "details": details}


def h3(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    return {
        "models": _per_model(dataset, iterations, seed),
        "pairwise": _pairwise(dataset, iterations, seed),
    }


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(DESCRIPTION).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["H3"] = h3(dataset, args.iterations, args.seed)
    cli.save(cli.output_dir(args, dataset), "h3.json", results)


if __name__ == "__main__":
    main()
