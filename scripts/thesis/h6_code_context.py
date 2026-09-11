#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.dataset import Dataset
from thesis._common import RESAMPLING_SCHEME, compare_paired

DESCRIPTION = "H6 - how much accuracy comes from the code fragment shown to the model"


def verdicts(dataset: Dataset, model: str, repetition: int) -> dict:
    out = {}
    for finding in dataset.findings:
        decision = finding.decisions.get(model)
        if decision is None or len(decision.votes) <= repetition:
            continue
        out[finding.key] = decision.votes[repetition]
    return out


def h6(main: Dataset, ablation: Dataset, repetition: int, iterations: int, seed: int) -> dict[str, Any]:
    shared = sorted(set(main.models) & set(ablation.models))
    result: dict[str, Any] = {
        "main_repetition": repetition,
        "missing_models": sorted(set(main.models) - set(ablation.models)),
        "models": {},
    }
    by_key = {finding.key: finding for finding in main.findings}

    for model in shared:
        with_code = verdicts(main, model, repetition)
        without_code = verdicts(ablation, model, 0)
        keys = sorted(set(with_code) & set(without_code), key=str)
        if not keys:
            continue
        findings = [by_key[key] for key in keys]
        labels = np.array([finding.label for finding in findings], dtype=bool)
        hits_with = np.array([with_code[k] for k in keys], dtype=bool) == labels
        hits_without = np.array([without_code[k] for k in keys], dtype=bool) == labels
        result["models"][main.display_name(model)] = {
            "n": len(keys),
            "with_code": float(hits_with.mean()),
            "without_code": float(hits_without.mean()),
            "kept_with_code": float(np.mean([with_code[k] for k in keys])),
            "kept_without_code": float(np.mean([without_code[k] for k in keys])),
            **compare_paired(findings, hits_with, hits_without, iterations, seed),
        }
    return result


def main() -> None:
    import thesis._cli as cli
    from analysis.dataset import load_dataset
    from analysis.runs import find_run_dir

    parser = cli.parser(DESCRIPTION)
    parser.add_argument("--ablation-run", type=int, default=7)
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset(args.run_dir, include_partial=True)
    ablation = load_dataset(find_run_dir(args.ablation_run), include_partial=True)

    results = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["H6"] = h6(dataset, ablation, args.repetition, args.iterations, args.seed)
    results["H6"]["ablation_run"] = Path(find_run_dir(args.ablation_run)).name
    if not results["H6"]["models"]:
        raise SystemExit("no models in common - did the run without code finish?")
    cli.save(cli.output_dir(args, dataset), "h6.json", results)


if __name__ == "__main__":
    main()
