#!/usr/bin/env python3
"""H2 — czy trafność klasyfikacji różni się między modelami (Q Cochrana)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from analysis.dataset import Dataset
from analysis.tests import (
    cochran_q,
    cochran_q_cluster_permutation,
    discordance,
    holm,
    mcnemar_cluster_signflip,
    mcnemar_exact,
    signflip_mde,
)
from thesis._common import _pairs, _scopes


def h2(dataset: Dataset, seed: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for scope, findings in _scopes(dataset):
        common = [
            finding
            for finding in findings
            if all(
                finding.decisions.get(model) and finding.decisions[model].verdict is not None
                for model in dataset.models
            )
        ]
        if len(dataset.models) < 2 or not common:
            continue
        correctness = {
            model: [finding.decisions[model].verdict == finding.label for finding in common]
            for model in dataset.models
        }
        matrix = [[correctness[model][index] for model in dataset.models] for index in range(len(common))]
        omnibus = cochran_q_cluster_permutation(
            matrix, [finding.rule for finding in common], seed=seed
        )

        pairwise: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        for first, second in _pairs(dataset.models):
            b, c = discordance(correctness[first], correctness[second])
            exact = mcnemar_exact(b, c)
            clustered = mcnemar_cluster_signflip(
                correctness[first], correctness[second], [finding.rule for finding in common], seed=seed
            )
            key = f"{dataset.display_name(first)} vs {dataset.display_name(second)}"
            pairwise[key] = {
                "accuracy_first": sum(correctness[first]) / len(common),
                "accuracy_second": sum(correctness[second]) / len(common),
                "mcnemar_exact": exact,
                "mcnemar_cluster": clustered,
                "mde": signflip_mde(
                    correctness[first],
                    correctness[second],
                    [finding.rule for finding in common],
                    total=len(common),
                    seed=seed,
                ),
            }
            raw_p[key] = clustered["p_value"]
        adjusted = holm(raw_p)
        for key, value in adjusted.items():
            pairwise[key]["p_value_holm"] = value

        results[scope] = {
            "n": len(common),
            "accuracy": {
                dataset.display_name(model): sum(correctness[model]) / len(common)
                for model in dataset.models
            },
            "cochran_q": omnibus,
            "pairwise": pairwise,
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
    results["H2"] = h2(dataset, args.seed)
    cli.save(cli.output_dir(args, dataset), "h2.json", results)


if __name__ == "__main__":
    main()
