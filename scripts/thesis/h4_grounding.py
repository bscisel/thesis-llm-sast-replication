#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset, Finding
from analysis.grounding import Grounding, build_grounding, score_text, tool_roots
from thesis._common import RESAMPLING_SCHEME, compare_paired

DESCRIPTION = "H4 - do model reasonings refer to the shown code more often than the tool message"


def _grounded_reasoning(
    findings: Sequence[Finding], model: str, groundings: dict
) -> tuple[list[Finding], np.ndarray]:
    used, grounded = [], []
    for finding in findings:
        decision = finding.decisions.get(model)
        if decision is None or not decision.reasonings:
            continue
        used.append(finding)
        grounded.append(score_text(decision.reasonings[0], groundings[finding.key])["grounded"])
    return used, np.array(grounded, dtype=bool)


def _grounded_message(findings: Sequence[Finding], groundings: dict) -> np.ndarray:
    return np.array(
        [
            score_text(f"{f.message} {f.description}", groundings[f.key])["grounded"]
            for f in findings
        ],
        dtype=bool,
    )


def h4(dataset: Dataset, iterations: int, seed: int) -> dict[str, Any]:
    if dataset.source_root is None or not dataset.source_root.exists():
        return {"available": False, "reason": f"no source directory: {dataset.source_root}"}

    roots = tool_roots(dataset.findings, dataset.source_root)
    groundings: dict[tuple, Grounding] = {
        f.key: build_grounding(f, dataset.source_root, roots[f.tool]) for f in dataset.findings
    }
    usable = [f for f in dataset.findings if groundings[f.key].context_available]

    models: dict[str, Any] = {}
    per_model_rates: dict[str, dict[str, float]] = {}
    for model in dataset.models:
        used, reasoning = _grounded_reasoning(usable, model, groundings)
        if not used:
            continue
        message = _grounded_message(used, groundings)
        models[dataset.display_name(model)] = {
            "n": len(used),
            "reasoning": float(reasoning.mean()),
            "message": float(message.mean()),
            **compare_paired(used, reasoning, message, iterations, seed),
        }
        per_model_rates[model] = {
            tool: float(
                np.mean([r for f, r in zip(used, reasoning) if f.tool == tool])
            )
            for tool in {f.tool for f in used}
        }

    per_tool: dict[str, Any] = {}
    for tool in sorted({f.tool for f in usable}):
        subset = [f for f in usable if f.tool == tool]
        rates = [rates[tool] for rates in per_model_rates.values() if tool in rates]
        per_tool[tool] = {
            "n": len(subset),
            "message": float(_grounded_message(subset, groundings).mean()),
            "model_min": min(rates) if rates else None,
            "model_max": max(rates) if rates else None,
        }

    return {
        "available": True,
        "findings_with_context": len(usable),
        "models": models,
        "per_tool": per_tool,
    }


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(DESCRIPTION).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["H4"] = h4(dataset, args.iterations, args.seed)
    cli.save(cli.output_dir(args, dataset), "h4.json", results)


if __name__ == "__main__":
    main()
