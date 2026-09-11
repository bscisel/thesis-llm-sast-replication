#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset
from analysis.resampling import Resampler
from thesis._common import RESAMPLING_SCHEME, arrays, confusion, raw_report_f1, scored

DESCRIPTION = "H1 - false alarm reduction and the F1 balance against the raw report"


def h1(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for model in dataset.models:
        findings = scored(dataset.findings, model)
        labels, predictions = arrays(findings, model)
        counts = confusion(labels, predictions, np.arange(len(findings)))

        def delta_f1(indices: np.ndarray) -> float | None:
            model_f1 = confusion(labels, predictions, indices).f1
            baseline = raw_report_f1(labels, indices)
            return None if model_f1 is None or baseline is None else model_f1 - baseline

        resampler = Resampler(findings, seed=seed)
        intervals = resampler.intervals(
            {
                "fp_reduction": lambda i: confusion(labels, predictions, i).fp_reduction,
                "delta_f1": delta_f1,
            },
            iterations=iterations,
        )

        results[dataset.display_name(model)] = {
            "n": len(findings),
            "fp_reduction": counts.fp_reduction,
            "fp_reduction_ci": [
                intervals["fp_reduction"]["ci_low"],
                intervals["fp_reduction"]["ci_high"],
            ],
            "lost_tp": counts.fn,
            "lost_tp_share": counts.lost_tp_share,
            "f1": counts.f1,
            "f1_raw_report": raw_report_f1(labels),
            "delta_f1": intervals["delta_f1"]["point"],
            "delta_f1_ci": [intervals["delta_f1"]["ci_low"], intervals["delta_f1"]["ci_high"]],
            "supported": bool(
                intervals["delta_f1"]["ci_low"] is not None and intervals["delta_f1"]["ci_low"] > 0
            ),
        }
    return results


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(DESCRIPTION).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["H1"] = h1(dataset, args.iterations, args.seed)
    cli.save(cli.output_dir(args, dataset), "h1.json", results)


if __name__ == "__main__":
    main()
