#!/usr/bin/env python3
"""H4 — osadzenie uzasadnień w kodzie wobec komunikatu narzędzia i różnice między modelami."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset
from analysis.grounding import Grounding, build_grounding, reference_precision, score_text
from analysis.tests import (
    discordance,
    friedman,
    holm,
    mcnemar_cluster_signflip,
    mcnemar_exact,
    signflip_mde,
    wilcoxon_signed_rank,
)


def h4(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    if dataset.source_root is None or not dataset.source_root.exists():
        return {"available": False, "reason": f"no source directory: {dataset.source_root}"}

    groundings: dict[tuple, Grounding] = {
        finding.key: build_grounding(finding, dataset.source_root) for finding in dataset.findings
    }
    usable = [finding for finding in dataset.findings if groundings[finding.key].context_available]

    h4a: dict[str, Any] = {}
    per_model_counts: dict[str, list[float]] = {}
    for model in dataset.models:
        explanation_grounded: list[bool] = []
        reasoning_grounded: list[bool] = []
        message_grounded: list[bool] = []
        beyond: list[bool] = []
        near: list[bool] = []
        hits: list[float] = []
        clusters: list[str] = []
        precisions: list[float] = []
        unsupported_any: list[bool] = []
        for finding in usable:
            decision = finding.decisions.get(model)
            if decision is None or not decision.reasonings or not decision.explanations:
                continue
            grounding = groundings[finding.key]
            text = decision.reasonings[0]
            scored = score_text(text, grounding)
            precision = reference_precision(text, grounding)
            if precision.get("precision") is not None:
                precisions.append(precision["precision"])
                unsupported_any.append(precision["unsupported"] > 0)
            message_score = score_text(f"{finding.message} {finding.description}", grounding)
            reasoning_grounded.append(scored["grounded"])
            explanation_grounded.append(
                score_text(decision.explanations[0], grounding)["grounded"]
            )
            message_grounded.append(message_score["grounded"])
            beyond.append(scored["grounded_beyond_message"])
            near.append(scored["grounded_near_flagged_line"])
            hits.append(float(scored["hits_beyond_message"]))
            clusters.append(finding.rule)

        if not explanation_grounded:
            continue
        b, c = discordance(explanation_grounded, message_grounded)
        clustered = mcnemar_cluster_signflip(explanation_grounded, message_grounded, clusters, seed=seed)
        b_r, c_r = discordance(reasoning_grounded, message_grounded)
        h4a[dataset.display_name(model)] = {
            "n": len(explanation_grounded),
            "explanation_grounded": sum(explanation_grounded) / len(explanation_grounded),
            "reasoning_grounded": sum(reasoning_grounded) / len(reasoning_grounded),
            "message_grounded": sum(message_grounded) / len(message_grounded),
            "grounded_beyond_message": sum(beyond) / len(beyond),
            "grounded_near_flagged_line": sum(near) / len(near),
            "mean_identifiers_beyond_message": float(np.mean(hits)),
            "reference_precision": float(np.mean(precisions)) if precisions else None,
            "unsupported_reference_share": (sum(unsupported_any) / len(unsupported_any)) if unsupported_any else None,
            "scored_for_precision": len(precisions),
            "mcnemar_exact": mcnemar_exact(b, c),
            "mcnemar_cluster": clustered,
            "reasoning_mcnemar_exact": mcnemar_exact(b_r, c_r),
            "reasoning_mcnemar_cluster": mcnemar_cluster_signflip(
                reasoning_grounded, message_grounded, clusters, seed=seed
            ),
        }
    # Sama zgodność długości kolumn nie gwarantuje tego samego zbioru ostrzeżeń.
    common = [
        finding
        for finding in usable
        if all(
            finding.decisions.get(model) and finding.decisions[model].reasonings
            for model in dataset.models
        )
    ]
    for model in dataset.models:
        per_model_counts[dataset.display_name(model)] = [
            score_text(finding.decisions[model].reasonings[0], groundings[finding.key])[
                "hits_beyond_message"
            ]
            for finding in common
        ]

    rq7: dict[str, Any] = {"paired_findings": len(common), "findings_with_context": len(usable)}
    names = list(per_model_counts)
    if len(names) >= 2 and common:
        rq7["friedman"] = friedman([per_model_counts[name] for name in names])
        raw_p: dict[str, float] = {}
        pairwise: dict[str, Any] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]} vs {names[j]}"
                test = wilcoxon_signed_rank(per_model_counts[names[i]], per_model_counts[names[j]])
                pairwise[key] = test
                raw_p[key] = test["p_value"]
        for key, value in holm(raw_p).items():
            pairwise[key]["p_value_holm"] = value
        rq7["pairwise"] = pairwise
        rq7["mean_identifiers_beyond_message"] = {
            name: float(np.mean(values)) for name, values in per_model_counts.items()
        }

    full_file = sum(1 for finding in usable if groundings[finding.key].shown_full_file)
    return {
        "available": True,
        "findings_with_context": len(usable),
        "findings_with_full_file": full_file,
        "findings_with_window": len(usable) - full_file,
        "H4": h4a,
        "RQ7": rq7,
    }


def main() -> None:
    import thesis._cli as cli
    # sys.modules[__name__], bo uruchomiony skrypt jest modułem __main__ i import po nazwie łapie inną kopię.
    self_module = sys.modules[__name__]

    args = cli.parser(__doc__).parse_args()
    dataset = cli.dataset_from(args)
    scheme = cli.apply_scheme(args, [self_module])
    results = cli.header(dataset, args, scheme)
    results["H4"] = h4(dataset, args.iterations, args.seed)
    cli.save(cli.output_dir(args, dataset), "h4.json", results)


if __name__ == "__main__":
    main()
