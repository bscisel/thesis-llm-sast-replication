#!/usr/bin/env python3
"""Wspólna obsługa wiersza poleceń: te same flagi w każdym skrypcie liczącym."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.dataset import Dataset, load_dataset
from analysis.resampling import DEFAULT_ITERATIONS, DEFAULT_SEED


def parser(doc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--run-dir", default="6")
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="policz też modele bez kompletu werdyktów (domyślnie pomijane, bo zawężają wspólną próbę)",
    )
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--pricing",
        type=Path,
        help='JSON z cenami katalogowymi: {"claude": {"input": 15.0, "output": 75.0}, ...} USD za 1 mln tokenów',
    )
    p.add_argument(
        "--cluster-by",
        choices=("plan", "rule", "stratum", "file", "none"),
        default="plan",
        help=(
            "jednostka losowania bootstrapu; 'plan' bierze schemat per narzędzie z dataset.RESAMPLING_SCHEME. "
            "Dla próby łączonej użyj 'rule' — warstwy Error Prone nie przenoszą się między etapami doboru."
        ),
    )
    p.add_argument("--output-dir", type=Path)
    return p


def dataset_from(args: argparse.Namespace) -> Dataset:
    dataset = load_dataset(args.run_dir, include_partial=args.include_partial)
    for model, count in (dataset.partial_models or {}).items():
        stan = "policzony" if args.include_partial else "POMINIĘTY"
        print(f"UWAGA: {model} ma {count}/{len(dataset.findings)} werdyktów — {stan}.")
    return dataset


def apply_scheme(args: argparse.Namespace, modules: Sequence[ModuleType]) -> str:
    """Rozsyła wybraną jednostkę losowania do modułów."""
    if args.cluster_by == "plan":
        from analysis.dataset import RESAMPLING_SCHEME

        return RESAMPLING_SCHEME
    for module in modules:
        if hasattr(module, "RESAMPLING_SCHEME"):
            module.RESAMPLING_SCHEME = args.cluster_by
    return args.cluster_by


def pricing_from(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.pricing:
        return None
    with args.pricing.open(encoding="utf-8") as handle:
        return json.load(handle)


def output_dir(args: argparse.Namespace, dataset: Dataset) -> Path:
    target = args.output_dir or dataset.run_dir / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


def save(target: Path, name: str, payload: Any) -> Path:
    path = target / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Zapisano: {path}")
    return path


def header(dataset: Dataset, args: argparse.Namespace, scheme: str) -> dict[str, Any]:
    return {
        "run": dataset.run_dir.name,
        "models": {model: dataset.display_name(model) for model in dataset.models},
        "settings": {
            "iterations": args.iterations,
            "seed": args.seed,
            "alpha": 0.05,
            "resampling": scheme,
        },
    }
