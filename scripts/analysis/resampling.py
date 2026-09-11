from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from analysis.dataset import Finding

DEFAULT_SEED = 20260803
DEFAULT_ITERATIONS = 10000


class Resampler:
    def __init__(
        self,
        findings: Sequence[Finding],
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.findings = list(findings)
        self.rng = np.random.default_rng(seed)
        self._clusters: list[list[np.ndarray]] = []
        self._build()

    def _build(self) -> None:
        by_tool: dict[str, list[int]] = defaultdict(list)
        for position, finding in enumerate(self.findings):
            by_tool[finding.tool].append(position)

        for tool in sorted(by_tool):
            groups: dict[str, list[int]] = defaultdict(list)
            for position in by_tool[tool]:
                groups[self.findings[position].rule].append(position)
            self._clusters.append([np.array(sorted(m)) for _, m in sorted(groups.items())])

    def draw(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for clusters in self._clusters:
            picks = self.rng.integers(0, len(clusters), size=len(clusters))
            parts.extend(clusters[index] for index in picks)
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
    icc: float


def icc_oneway(values: Sequence[float], groups: Sequence[Any]) -> ClusterSummary:
    buckets: dict[Any, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(float(value))
    sizes = np.array([len(members) for members in buckets.values()], dtype=float)
    total = float(sizes.sum())
    k = len(sizes)
    if k < 2 or total <= k:
        return ClusterSummary(k, total / max(k, 1), 0.0)

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
    return ClusterSummary(clusters=k, mean_size=total / k, icc=icc)
