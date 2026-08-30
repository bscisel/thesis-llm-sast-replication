#!/usr/bin/env python3
"""H1a i H1b — redukcja fałszywych alarmów i bilans F1 wobec punktu odniesienia B0."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import RESAMPLING_SCHEME, Dataset
from analysis.resampling import Resampler
from analysis.tests import cohens_h, discordance, mcnemar_cluster_signflip, mcnemar_exact
from thesis._common import _arrays, _b0_f1, _confusion, _delta_f1, _safe_share, _scopes, _scored


def h1(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    results: dict[str, Any] = {"H1a": {}, "H1b": {}}
    for scope, findings in _scopes(dataset):
        for model in dataset.models:
            scored = _scored(findings, model)
            labels, predictions = _arrays(scored, model)
            counts = _confusion(labels, predictions, np.arange(len(scored)))

            raw_alarms = [not finding.label for finding in scored]
            model_alarms = [
                bool(prediction) and not label for label, prediction in zip(labels, predictions)
            ]
            b, c = discordance(raw_alarms, model_alarms)
            exact = mcnemar_exact(b, c)
            clustered = mcnemar_cluster_signflip(
                raw_alarms, model_alarms, [finding.rule for finding in scored], seed=seed
            )

            resampler = Resampler(scored, scheme=RESAMPLING_SCHEME, seed=seed)
            intervals = resampler.intervals(
                {
                    "fp_reduction": lambda indices: _confusion(labels, predictions, indices).fp_reduction,
                    "delta_f1": lambda indices: _delta_f1(labels, predictions, indices),
                    "lost_tp_share": lambda indices: _confusion(labels, predictions, indices).lost_tp_share,
                },
                iterations=iterations,
            )

            name = dataset.display_name(model)
            results["H1a"].setdefault(scope, {})[name] = {
                "false_alarms_raw": counts.fp + counts.tn,
                "false_alarms_after": counts.fp,
                "removed": counts.tn,
                "fp_reduction": counts.fp_reduction,
                "fp_reduction_ci": [intervals["fp_reduction"]["ci_low"], intervals["fp_reduction"]["ci_high"]],
                "mcnemar_exact": exact,
                "mcnemar_cluster": clustered,
                "cohens_h": cohens_h(
                    _safe_share(counts.fp, counts.fp + counts.tn), 1.0 if counts.fp + counts.tn else None
                ),
            }
            results["H1b"].setdefault(scope, {})[name] = {
                "f1": counts.f1,
                "f1_baseline_b0": _b0_f1(labels),
                "delta_f1": intervals["delta_f1"]["point"],
                "delta_f1_ci": [intervals["delta_f1"]["ci_low"], intervals["delta_f1"]["ci_high"]],
                "lost_tp": counts.fn,
                "lost_tp_share": counts.lost_tp_share,
                "lost_tp_share_ci": [
                    intervals["lost_tp_share"]["ci_low"],
                    intervals["lost_tp_share"]["ci_high"],
                ],
                "supported": bool(
                    intervals["delta_f1"]["ci_low"] is not None and intervals["delta_f1"]["ci_low"] > 0
                ),
            }
    return results


def main() -> None:
    import thesis._cli as cli
    # sys.modules[__name__], bo uruchomiony skrypt jest modułem __main__ i import po nazwie łapie inną kopię.
    self_module = sys.modules[__name__]

    args = cli.parser(__doc__).parse_args()
    dataset = cli.dataset_from(args)
    scheme = cli.apply_scheme(args, [self_module])
    results = cli.header(dataset, args, scheme)
    results["H1"] = h1(dataset, args.iterations, args.seed)
    cli.save(cli.output_dir(args, dataset), "h1.json", results)


if __name__ == "__main__":
    main()
