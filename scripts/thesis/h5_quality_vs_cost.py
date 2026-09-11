#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset
from analysis.resampling import Resampler
from thesis._common import RESAMPLING_SCHEME, arrays, complete_findings, confusion

DESCRIPTION = "H5 - F1 of the locally run open-weight model against the most expensive one, and cost"
THRESHOLD = 0.80


def _cost(dataset: Dataset, pricing: dict[str, Any] | None) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for model in dataset.models:
        decisions = [f.decisions[model] for f in dataset.findings if f.decisions.get(model)]
        input_tokens = sum(d.input_tokens for d in decisions)
        output_tokens = sum(d.output_tokens for d in decisions)
        cache_read = sum(d.cache_read_tokens for d in decisions)
        cached_within = sum(d.cached_within_input for d in decisions)
        cache_write = sum(d.cache_write_tokens for d in decisions)

        entry: dict[str, Any] = {"findings": len(decisions)}
        prices = (pricing or {}).get(model)
        if prices:
            billable = input_tokens - cached_within
            total = (
                billable / 1e6 * prices["input"]
                + (cache_read + cached_within)
                / 1e6
                * prices["input"]
                * prices.get("cache_read_multiplier", 0.0)
                + cache_write / 1e6 * prices["input"] * prices.get("cache_write_multiplier", 0.0)
                + output_tokens / 1e6 * prices["output"]
            )
            entry["cost_usd_total"] = total
            entry["cost_usd_per_finding"] = total / len(decisions) if decisions else None
        usage[dataset.display_name(model)] = entry
    return usage


def h5(dataset: Dataset, iterations: int, seed: int, pricing: dict[str, Any] | None) -> dict[str, Any]:
    reference = "claude" if "claude" in dataset.models else dataset.models[0]
    common = complete_findings(dataset)
    labels, reference_predictions = arrays(common, reference)
    resampler = Resampler(common, seed=seed)

    ratios: dict[str, Any] = {}
    for model in dataset.models:
        if model == reference:
            continue
        _, predictions = arrays(common, model)

        def ratio(indices: np.ndarray, p: np.ndarray = predictions) -> float | None:
            numerator = confusion(labels, p, indices).f1
            denominator = confusion(labels, reference_predictions, indices).f1
            if numerator is None or denominator in (None, 0):
                return None
            return numerator / denominator

        interval = resampler.intervals({"ratio": ratio}, iterations=iterations)["ratio"]
        ratios[dataset.display_name(model)] = {
            "reference": dataset.display_name(reference),
            "f1_ratio": interval["point"],
            "f1_ratio_ci": [interval["ci_low"], interval["ci_high"]],
            "threshold_met": bool(interval["ci_low"] is not None and interval["ci_low"] >= THRESHOLD),
        }

    return {"usage": _cost(dataset, pricing), "ratios": ratios, "pricing_supplied": bool(pricing)}


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(DESCRIPTION).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["H5"] = h5(dataset, args.iterations, args.seed, cli.pricing_from(args))
    cli.save(cli.output_dir(args, dataset), "h5.json", results)


if __name__ == "__main__":
    main()
