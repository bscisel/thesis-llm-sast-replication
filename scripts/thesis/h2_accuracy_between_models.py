#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.constants import ALL_TOOLS
from analysis.dataset import Dataset
from analysis.tests import cochran_q_cluster_permutation
from thesis._common import RESAMPLING_SCHEME, compare_paired, complete_findings

DESCRIPTION = "H2 - verdict accuracy between models and against the naive classifier"


def h2(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    findings = complete_findings(dataset)
    if len(dataset.models) < 2 or not findings:
        return {}

    correct = {
        model: np.array([f.decisions[model].verdict == f.label for f in findings], dtype=bool)
        for model in dataset.models
    }
    naive = np.array([not f.label for f in findings], dtype=bool)

    matrix = [[bool(correct[model][i]) for model in dataset.models] for i in range(len(findings))]
    return {
        ALL_TOOLS: {
            "n": len(findings),
            "accuracy": {
                dataset.display_name(model): float(correct[model].mean())
                for model in dataset.models
            },
            "cochran_q": cochran_q_cluster_permutation(
                matrix, [f.rule for f in findings], seed=seed
            ),
            "vs_naive": {
                dataset.display_name(model): {
                    "accuracy": float(correct[model].mean()),
                    "accuracy_naive": float(naive.mean()),
                    **compare_paired(findings, correct[model], naive, iterations, seed),
                }
                for model in dataset.models
            },
        }
    }


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(DESCRIPTION).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["H2"] = h2(dataset, args.iterations, args.seed)
    cli.save(cli.output_dir(args, dataset), "h2.json", results)


if __name__ == "__main__":
    main()
