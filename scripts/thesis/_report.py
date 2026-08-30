#!/usr/bin/env python3
"""Składanie raportu: rodzina p-wartości, korekta Holma, podsumowanie w Markdownie."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from analysis.dataset import Dataset
from thesis._common import ALL_TOOLS


def _family_p_values(results: dict[str, Any], dataset: Dataset) -> dict[str, float]:
    family: dict[str, float] = {}
    for model, entry in results["H1"]["H1a"].get(ALL_TOOLS, {}).items():
        family[f"H1a/{model}"] = entry["mcnemar_cluster"]["p_value"]
    cochran = results["H2"].get(ALL_TOOLS, {}).get("cochran_q", {})
    omnibus = cochran.get("p_value_cluster", cochran.get("p_value"))
    if omnibus is not None:
        family["H2"] = omnibus
    grounding = results.get("H4", {})
    if grounding.get("available"):
        for model, entry in grounding["H4"].items():
            family[f"H4/{model}"] = entry["reasoning_mcnemar_cluster"]["p_value"]
    ablation = results.get("H6", {})
    for model, entry in ablation.get("modele", {}).items():
        p_value = entry.get("mcnemar_cluster", {}).get("p_value")
        if p_value is not None:
            family[f"H6/{model}"] = p_value
    return family
