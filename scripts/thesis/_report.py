#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.constants import ALL_TOOLS


def family_p_values(results: dict[str, Any]) -> dict[str, float]:
    family: dict[str, float] = {}

    scope = results.get("H2", {}).get(ALL_TOOLS, {})
    cochran = scope.get("cochran_q", {})
    omnibus = cochran.get("p_value_cluster", cochran.get("p_value"))
    if omnibus is not None:
        family["H2/cochran"] = omnibus
    for model, entry in scope.get("vs_naive", {}).items():
        family[f"H2/naive/{model}"] = entry["mcnemar_cluster"]["p_value"]

    grounding = results.get("H4", {})
    if grounding.get("available"):
        for model, entry in grounding["models"].items():
            family[f"H4/{model}"] = entry["mcnemar_cluster"]["p_value"]

    for model, entry in results.get("H6", {}).get("models", {}).items():
        family[f"H6/{model}"] = entry["mcnemar_cluster"]["p_value"]

    return family
