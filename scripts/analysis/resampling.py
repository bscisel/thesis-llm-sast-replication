from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from analysis.dataset import RESAMPLING_SCHEME, Finding

DEFAULT_SEED = 20260803
DEFAULT_ITERATIONS = 10000


def _group_key(finding: Finding, scheme: str) -> str:
    if scheme == "rule":
        return finding.rule
    if scheme == "stratum":
        return f"{finding.tool}::{finding.stratum}"
    if scheme == "file":
        return f"{finding.tool}::{finding.sourcefile}"
    if scheme == "none":
        return f"{finding.tool}"
    raise ValueError(f"unknown resampling scheme: {scheme}")


class Resampler:
    def __init__(
        self,
        findings: Sequence[Finding],
        scheme: dict[str, str] | str = RESAMPLING_SCHEME,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.findings = list(findings)
        self.scheme = scheme
        self.rng = np.random.default_rng(seed)
        self._clusters: list[list[np.ndarray]] = []
        self._units: list[np.ndarray] = []
        self._build()

    def _scheme_for(self, tool: str) -> str:
        if isinstance(self.scheme, str):
            return self.scheme
        return self.scheme.get(tool, "rule")

    def _build(self) -> None:
        by_tool: dict[str, list[int]] = defaultdict(list)
        for position, finding in enumerate(self.findings):
            by_tool[finding.tool].append(position)

        for tool in sorted(by_tool):
            scheme = self._scheme_for(tool)
            groups: dict[str, list[int]] = defaultdict(list)
            for position in by_tool[tool]:
                groups[_group_key(self.findings[position], scheme)].append(position)
            arrays = [np.array(sorted(members)) for _, members in sorted(groups.items())]
            if scheme == "stratum":
                self._assert_strata_match_rules(tool, by_tool[tool], groups)
                self._units.extend(arrays)
            else:
                self._clusters.append(arrays)

    def _assert_strata_match_rules(self, tool, positions, groups) -> None:
        rules = defaultdict(list)
        for position in positions:
            rules[self.findings[position].rule].append(position)
        if sorted(sorted(v) for v in groups.values()) == sorted(sorted(v) for v in rules.values()):
            return
        raise ValueError(
            f"{tool}: the 'stratum' scheme assumes the strata coincide with the rules, "
            f"but there are {len(groups)} strata against {len(rules)} rules. Use --cluster-by rule."
        )

    def draw(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for clusters in self._clusters:
            picks = self.rng.integers(0, len(clusters), size=len(clusters))
            parts.extend(clusters[index] for index in picks)
        for unit in self._units:
            parts.append(self.rng.choice(unit, size=len(unit), replace=True))
        if not parts:
            return np.array([], dtype=int)
        return np.concatenate(parts)


    def interval(
        self,
        statistic: Callable[[np.ndarray], float | None],
        iterations: int = DEFAULT_ITERATIONS,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        point = statistic(np.arange(len(self.findings)))
        samples: list[float] = []
        for _ in range(iterations):
            value = statistic(self.draw())
            if value is not None:
                samples.append(value)
        if not samples:
            return {"point": point, "ci_low": None, "ci_high": None, "iterations": 0}
        ordered = np.sort(np.array(samples))
        low = float(np.quantile(ordered, alpha / 2))
        high = float(np.quantile(ordered, 1 - alpha / 2))
        return {
            "point": point,
            "ci_low": low,
            "ci_high": high,
            "iterations": len(samples),
            "se": float(np.std(ordered, ddof=1)) if len(ordered) > 1 else None,
        }

    def intervals(
        self,
        statistics: dict[str, Callable[[np.ndarray], float | None]],
        iterations: int = DEFAULT_ITERATIONS,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, Any]]:
        full = np.arange(len(self.findings))
        points = {name: statistic(full) for name, statistic in statistics.items()}
        samples: dict[str, list[float]] = {name: [] for name in statistics}
        for _ in range(iterations):
            indices = self.draw()
            for name, statistic in statistics.items():
                value = statistic(indices)
                if value is not None:
                    samples[name].append(value)

        results: dict[str, dict[str, Any]] = {}
        for name in statistics:
            drawn = samples[name]
            if not drawn:
                results[name] = {"point": points[name], "ci_low": None, "ci_high": None, "iterations": 0}
                continue
            ordered = np.sort(np.array(drawn))
            results[name] = {
                "point": points[name],
                "ci_low": float(np.quantile(ordered, alpha / 2)),
                "ci_high": float(np.quantile(ordered, 1 - alpha / 2)),
                "iterations": len(drawn),
                "se": float(np.std(ordered, ddof=1)) if len(ordered) > 1 else None,
            }
        return results


@dataclass(frozen=True)
class ClusterSummary:
    clusters: int
    mean_size: float
    n0: float
    icc: float
    design_effect: float
    effective_n: float


def icc_oneway(values: Sequence[float], groups: Sequence[Any]) -> ClusterSummary:
    buckets: dict[Any, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(float(value))
    sizes = np.array([len(members) for members in buckets.values()], dtype=float)
    total = float(sizes.sum())
    k = len(sizes)
    if k < 2 or total <= k:
        return ClusterSummary(k, total / max(k, 1), total / max(k, 1), 0.0, 1.0, total)

    grand_mean = float(np.mean([value for members in buckets.values() for value in members]))
    group_means = np.array([np.mean(members) for members in buckets.values()])
    ss_between = float(np.sum(sizes * (group_means - grand_mean) ** 2))
    ss_within = float(
        sum(sum((value - np.mean(members)) ** 2 for value in members) for members in buckets.values())
    )
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (total - k)
    n0 = (total - float(np.sum(sizes**2)) / total) / (k - 1)
    denominator = ms_between + (n0 - 1) * ms_within
    icc = 0.0 if denominator == 0 else (ms_between - ms_within) / denominator
    icc = max(0.0, min(1.0, icc))
    mean_size = total / k
    design_effect = 1 + (n0 - 1) * icc
    return ClusterSummary(
        clusters=k,
        mean_size=mean_size,
        n0=n0,
        icc=icc,
        design_effect=design_effect,
        effective_n=total / design_effect if design_effect else total,
    )
