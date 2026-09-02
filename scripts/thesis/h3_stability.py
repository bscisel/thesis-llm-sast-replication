#!/usr/bin/env python3
"""H3 — stabilność odpowiedzi między powtórzeniami (jednomyślność, kappa Fleissa)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import RESAMPLING_SCHEME, Dataset

RUNS_PER_FINDING = 3
from analysis.resampling import Resampler
from analysis.tests import fleiss_kappa
from thesis._common import _accuracy_spread, _kappa_from_counts, _pairs, _scopes


def h3(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for scope, findings in _scopes(dataset):
        scope_result: dict[str, Any] = {}
        for model in dataset.models:
            evaluated = [f for f in findings if f.decisions.get(model) and f.decisions[model].votes]
            usable = [f for f in evaluated if len(f.decisions[model].votes) == RUNS_PER_FINDING]
            if not usable:
                continue
            incomplete = len(evaluated) - len(usable)
            votes = [finding.decisions[model].votes for finding in usable]
            unanimity = np.array([len(set(vote)) == 1 for vote in votes], dtype=bool)
            counts = np.array([[vote.count(False), vote.count(True)] for vote in votes], dtype=float)

            resampler = Resampler(usable, scheme=RESAMPLING_SCHEME, seed=seed)
            intervals = resampler.intervals(
                {
                    "unanimity": lambda indices: float(np.mean(unanimity[indices])),
                    "kappa": lambda indices: _kappa_from_counts(counts[indices]),
                },
                iterations=iterations,
            )
            scope_result[dataset.display_name(model)] = {
                "findings": len(usable),
                "findings_dropped_incomplete": incomplete,
                "runs_per_finding": {
                    str(size): int(np.count_nonzero(np.array([len(vote) for vote in votes]) == size))
                    for size in sorted({len(vote) for vote in votes})
                },
                "unanimity": float(np.mean(unanimity)),
                "unanimity_ci": [intervals["unanimity"]["ci_low"], intervals["unanimity"]["ci_high"]],
                "fleiss_kappa": _kappa_from_counts(counts),
                "fleiss_kappa_ci": [intervals["kappa"]["ci_low"], intervals["kappa"]["ci_high"]],
                "accuracy_spread": _accuracy_spread(usable, model),
            }
        scope_result["pary"] = _pairs_on_difference(
            dataset, findings, scope_result, iterations, seed
        )
        results[scope] = scope_result
    return results


def _pairs_on_difference(
    dataset: Dataset,
    findings: Sequence[Any],
    per_model: dict[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Kryterium H3: przedział ufności różnicy jednomyślności nie obejmuje zera."""
    KRYTERIUM = "przedzial ufnosci roznicy jednomyslnosci nie obejmuje zera"
    models = [m for m in dataset.models if dataset.display_name(m) in per_model]
    if len(models) < 2:
        return {"kryterium": KRYTERIUM, "par": 0, "roznic_wykazanych": 0, "szczegoly": {}}

    unanimous_items, complete_set = {}, {}
    for model in models:
        decision_list = [f.decisions.get(model) for f in findings]
        complete_set[model] = np.array(
            [bool(d and d.votes and len(d.votes) == RUNS_PER_FINDING) for d in decision_list]
        )
        unanimous_items[model] = np.array(
            [bool(d and d.votes and len(set(d.votes)) == 1) for d in decision_list]
        )

    resampler = Resampler(list(findings), scheme=RESAMPLING_SCHEME, seed=seed)
    draws = [resampler.draw() for _ in range(iterations)]

    pairs, shown_items = {}, 0
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            common = complete_set[a] & complete_set[b]
            if not common.any():
                continue
            differences = []
            for indexes in draws:
                selected = indexes[common[indexes]]
                if selected.size:
                    differences.append(
                        float(unanimous_items[a][selected].mean() - unanimous_items[b][selected].mean())
                    )
            if not differences:
                continue
            ordered_items = np.sort(np.array(differences))
            lo = float(np.quantile(ordered_items, 0.025))
            hi = float(np.quantile(ordered_items, 0.975))
            point = float(
                unanimous_items[a][common].mean() - unanimous_items[b][common].mean()
            )
            shown_value = lo > 0.0 or hi < 0.0
            shown_items += shown_value
            pairs[f"{dataset.display_name(a)} vs {dataset.display_name(b)}"] = {
                "ostrzezen_wspolnych": int(common.sum()),
                "roznica": point,
                "roznica_ci": [lo, hi],
                "roznica_wykazana": bool(shown_value),
            }
    return {
        "kryterium": KRYTERIUM,
        "par": len(pairs),
        "roznic_wykazanych": shown_items,
        "szczegoly": pairs,
    }


def main() -> None:
    import thesis._cli as cli
    # sys.modules[__name__], bo uruchomiony skrypt jest modułem __main__ i import po nazwie łapie inną kopię.
    self_module = sys.modules[__name__]

    args = cli.parser(__doc__).parse_args()
    dataset = cli.dataset_from(args)
    scheme = cli.apply_scheme(args, [self_module])
    results = cli.header(dataset, args, scheme)
    results["H3"] = h3(dataset, args.iterations, args.seed)
    cli.save(cli.output_dir(args, dataset), "h3.json", results)


if __name__ == "__main__":
    main()
