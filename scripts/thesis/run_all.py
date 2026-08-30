#!/usr/bin/env python3
"""Przelicza komplet liczb do pracy jednym poleceniem."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.categorization import category_report
from analysis.signatures import signature_report
from analysis.tests import holm

import thesis._cli as cli
import thesis.h1_false_alarm_reduction as m_h1
import thesis.h2_accuracy_between_models as m_h2
import thesis.h3_stability as m_h3
import thesis.h4_grounding as m_grounding
import thesis.h5_quality_vs_cost as m_h5
from thesis._report import _family_p_values
from analysis.runs import find_run_dir
from thesis.h6_code_context import h6 as _h6


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--ablation-run", type=int, default=None,
                    help="numer runu z wariantem B3; bez niego rodzina Holma nie obejmuje H6")
    ap.add_argument("--main-repetition", type=int, default=0,
                    help="który powtórzenie przebiegu głównego wchodzi do porównania H6 (0-2)")
    args = ap.parse_args()
    dataset = cli.dataset_from(args)
    scheme = cli.apply_scheme(args, [m_h1, m_h2, m_h3, m_grounding, m_h5])
    pricing = cli.pricing_from(args)

    results: dict[str, Any] = cli.header(dataset, args, scheme)
    results["H1"] = m_h1.h1(dataset, args.iterations, args.seed)
    results["H2"] = m_h2.h2(dataset, args.seed)
    results["H3"] = m_h3.h3(dataset, args.iterations, args.seed)
    results["H4"] = m_grounding.h4(dataset, args.iterations, args.seed)
    results["H5"] = m_h5.h5(dataset, args.iterations, args.seed, pricing)
    results["RQ8"] = category_report(dataset)
    results["RQ9"] = signature_report(dataset)

    if args.ablation_run:
        from analysis.dataset import load_dataset
        ablation = load_dataset(find_run_dir(args.ablation_run), include_partial=True)
        results["H6"] = _h6(dataset, ablation, args.main_repetition, args.seed)
        results["H6"]["ablacja"] = Path(find_run_dir(args.ablation_run)).name

    raw_family = _family_p_values(results, dataset)
    results["holm_family"] = {
        "raw": raw_family,
        "adjusted": holm(raw_family),
        "size": len(raw_family),
        "complete": args.ablation_run is not None and results.get("H4", {}).get("available", False),
        "missing": [name for name, present in
                    (("H6", args.ablation_run is not None),
                     ("H4", results.get("H4", {}).get("available", False)))
                    if not present],
    }

    target = cli.output_dir(args, dataset)
    cli.save(target, "stats_tests.json", results)



if __name__ == "__main__":
    main()
