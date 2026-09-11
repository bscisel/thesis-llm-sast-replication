#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.categorization import category_report
from analysis.dataset import load_dataset
from analysis.runs import find_run_dir
from analysis.signatures import signature_report
from analysis.tests import holm

import thesis._cli as cli
from thesis._common import RESAMPLING_SCHEME
from thesis._report import family_p_values
from thesis.h1_false_alarm_reduction import h1
from thesis.h2_accuracy_between_models import h2
from thesis.h3_stability import h3
from thesis.h4_grounding import h4
from thesis.h5_quality_vs_cost import h5
from thesis.h6_code_context import h6

DESCRIPTION = "Computes every number the thesis reports, in one pass"


def main() -> None:
    parser = cli.parser(DESCRIPTION)
    parser.add_argument("--ablation-run", type=int, default=7)
    parser.add_argument("--main-repetition", type=int, default=0)
    args = parser.parse_args()

    dataset = cli.dataset_from(args)
    pricing = cli.pricing_from(args)

    results: dict[str, Any] = cli.header(dataset, args, RESAMPLING_SCHEME)
    results["H1"] = h1(dataset, args.iterations, args.seed)
    results["H2"] = h2(dataset, args.iterations, args.seed)
    results["H3"] = h3(dataset, args.iterations, args.seed)
    results["H4"] = h4(dataset, args.iterations, args.seed)
    results["H5"] = h5(dataset, args.iterations, args.seed, pricing)
    results["RQ8"] = category_report(dataset)
    results["RQ9"] = signature_report(dataset)

    if args.ablation_run:
        ablation = load_dataset(find_run_dir(args.ablation_run), include_partial=True)
        with_partial = load_dataset(args.run_dir, include_partial=True)
        results["H6"] = h6(
            with_partial, ablation, args.main_repetition, args.iterations, args.seed
        )
        results["H6"]["ablation_run"] = Path(find_run_dir(args.ablation_run)).name

    raw = family_p_values(results)
    results["holm_family"] = {"raw": raw, "adjusted": holm(raw), "size": len(raw)}

    cli.save(cli.output_dir(args, dataset), "stats_tests.json", results)


if __name__ == "__main__":
    main()
