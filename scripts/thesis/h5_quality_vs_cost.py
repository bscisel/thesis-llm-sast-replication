#!/usr/bin/env python3
"""H5 — jakość wobec kosztu: iloraz F1 i koszt na ostrzeżenie z rzeczywistego zużycia tokenów."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import RESAMPLING_SCHEME, Dataset
from analysis.resampling import Resampler
from thesis._common import _arrays, _confusion


def h5(dataset: Dataset, iterations: int, seed: int, pricing: dict[str, Any] | None) -> dict[str, Any]:
    reference = "claude"
    if reference not in dataset.models:
        reference = dataset.models[0]

    usage: dict[str, Any] = {}
    for model in dataset.models:
        decisions = [
            finding.decisions[model]
            for finding in dataset.findings
            if finding.decisions.get(model)
        ]
        runs = sum(decision.valid_runs for decision in decisions)
        input_tokens = sum(decision.input_tokens for decision in decisions)
        output_tokens = sum(decision.output_tokens for decision in decisions)
        cache_read = sum(decision.cache_read_tokens for decision in decisions)
        cached_within = sum(decision.cached_within_input for decision in decisions)
        cache_write = sum(decision.cache_write_tokens for decision in decisions)
        latency = sum(decision.latency_s for decision in decisions)
        reported = sum(decision.reported_cost_usd for decision in decisions)
        entry: dict[str, Any] = {
            "findings": len(decisions),
            "runs": runs,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_tokens_per_run": input_tokens / runs if runs else None,
            "output_tokens_per_run": output_tokens / runs if runs else None,
            "wall_clock_s_per_run": latency / runs if runs else None,
        }
        entry["cache_read_tokens"] = cache_read + cached_within
        entry["cache_write_tokens"] = cache_write
        entry["reported_cost_usd"] = reported or None
        prices = (pricing or {}).get(model)
        if prices:
            # Odczyt z pamięci podręcznej: u Anthropic rozłączny z wejściem, u OpenAI jego podzbiór, więc odejmowany.
            pelne = input_tokens - cached_within
            cost = (
                pelne / 1e6 * prices["input"]
                + (cache_read + cached_within) / 1e6 * prices["input"]
                * prices.get("cache_read_multiplier", 0.0)
                + cache_write / 1e6 * prices["input"]
                * prices.get("cache_write_multiplier", 0.0)
                + output_tokens / 1e6 * prices["output"]
            )
            entry["billable_input_tokens"] = pelne
            entry["pricing_usd_per_million"] = prices
            entry["cost_usd_total"] = cost
            entry["cost_usd_per_finding"] = cost / len(decisions) if decisions else None
            entry["cost_usd_per_run"] = cost / runs if runs else None
        usage[dataset.display_name(model)] = entry

    ratios: dict[str, Any] = {}
    common = [
        finding
        for finding in dataset.findings
        if all(
            finding.decisions.get(model) and finding.decisions[model].verdict is not None
            for model in dataset.models
        )
    ]
    reference_labels, reference_predictions = _arrays(common, reference)
    resampler = Resampler(common, scheme=RESAMPLING_SCHEME, seed=seed)
    for model in dataset.models:
        if model == reference:
            continue
        _, predictions = _arrays(common, model)

        def ratio(indices: np.ndarray, predictions: np.ndarray = predictions) -> float | None:
            numerator = _confusion(reference_labels, predictions, indices).f1
            denominator = _confusion(reference_labels, reference_predictions, indices).f1
            if numerator is None or denominator in (None, 0):
                return None
            return numerator / denominator

        interval = resampler.intervals({"ratio": ratio}, iterations=iterations)["ratio"]
        per_tool: dict[str, Any] = {}
        for tool in dataset.tools():
            subset = [finding for finding in common if finding.tool == tool]
            tool_labels, tool_reference = _arrays(subset, reference)
            _, tool_predictions = _arrays(subset, model)
            full = np.arange(len(subset))
            numerator = _confusion(tool_labels, tool_predictions, full).f1
            denominator = _confusion(tool_labels, tool_reference, full).f1
            per_tool[tool] = None if not denominator else numerator / denominator
        ratios[dataset.display_name(model)] = {
            "reference": dataset.display_name(reference),
            "f1_ratio": interval["point"],
            "f1_ratio_ci": [interval["ci_low"], interval["ci_high"]],
            "threshold_met": bool(interval["ci_low"] is not None and interval["ci_low"] >= 0.80),
            "threshold_met_point": bool(interval["point"] is not None and interval["point"] >= 0.80),
            "f1_ratio_per_tool": per_tool,
        }

    return {"usage": usage, "ratios": ratios, "pricing_supplied": bool(pricing)}


def main() -> None:
    import thesis._cli as cli
    # sys.modules[__name__], bo uruchomiony skrypt jest modułem __main__ i import po nazwie łapie inną kopię.
    self_module = sys.modules[__name__]

    args = cli.parser(__doc__).parse_args()
    dataset = cli.dataset_from(args)
    scheme = cli.apply_scheme(args, [self_module])
    results = cli.header(dataset, args, scheme)
    results["H5"] = h5(dataset, args.iterations, args.seed, cli.pricing_from(args))
    cli.save(cli.output_dir(args, dataset), "h5.json", results)


if __name__ == "__main__":
    main()
