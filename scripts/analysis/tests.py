from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats

DEFAULT_SEED = 20260803
DEFAULT_PERMUTATIONS = 20000


def mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    n = b + c
    if n == 0:
        p_value: float | None = 1.0
    else:
        p_value = min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    odds_ratio = None if c == 0 else b / c
    return {
        "b": b,
        "c": c,
        "discordant": n,
        "p_value": float(p_value) if p_value is not None else None,
        "odds_ratio": odds_ratio,
        "test": "exact McNemar test (two-sided, binomial distribution)",
    }


def discordance(first: Sequence[bool], second: Sequence[bool]) -> tuple[int, int]:
    b = sum(1 for x, y in zip(first, second) if x and not y)
    c = sum(1 for x, y in zip(first, second) if y and not x)
    return b, c


def _cluster_contributions(
    first: Sequence[bool],
    second: Sequence[bool],
    clusters: Sequence[Any],
) -> dict[Any, float]:
    """Przewaga netto pierwszego nad drugim, zsumowana wewnątrz każdego klastra."""
    contributions: dict[Any, float] = {}
    for x, y, cluster in zip(first, second, clusters):
        if x == y:
            continue
        contributions[cluster] = contributions.get(cluster, 0.0) + (1.0 if x and not y else -1.0)
    return contributions


def signflip_mde(
    first: Sequence[bool],
    second: Sequence[bool],
    clusters: Sequence[Any],
    total: int,
    power: float = 0.8,
    alpha: float = 0.05,
    permutations: int = DEFAULT_PERMUTATIONS,
    simulations: int = 4000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Najmniejsza różnica trafności, jaką ten plan wykrywa z zadaną mocą."""
    contributions = _cluster_contributions(first, second, clusters)
    values = np.array([abs(value) for value in contributions.values() if value != 0.0])
    if values.size == 0 or total == 0:
        return {"available": False}

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, values.size))
    critical = float(np.quantile(np.abs(signs @ values), 1 - alpha))

    mass = float(values.sum())
    for step in range(1, 101):
        q = 0.5 + step * 0.005
        if q > 1.0:
            break
        draws = np.where(rng.random((simulations, values.size)) < q, 1.0, -1.0)
        attained = float(np.mean(np.abs(draws @ values) >= critical))
        if attained >= power:
            return {
                "available": True,
                "mde_accuracy": (2 * q - 1) * mass / total,
                "attained_power": attained,
                "critical_value": critical,
                "clusters_with_discordance": int(values.size),
                "net_discordance": int(mass),
                "power_target": power,
                "alpha": alpha,
            }
    return {
        "available": True,
        "mde_accuracy": None,
        "critical_value": critical,
        "clusters_with_discordance": int(values.size),
        "net_discordance": int(mass),
        "power_target": power,
        "alpha": alpha,
        "note": "even with fully one-sided discordance the power stays below the threshold",
    }


def mcnemar_cluster_signflip(
    first: Sequence[bool],
    second: Sequence[bool],
    clusters: Sequence[Any],
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    contributions = _cluster_contributions(first, second, clusters)
    values = np.array([value for value in contributions.values() if value != 0.0])
    b, c = discordance(first, second)
    if values.size == 0:
        return {
            "b": b,
            "c": c,
            "clusters_with_discordance": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "test": "cluster-level sign-flip (SAST rule)",
        }

    observed = float(abs(values.sum()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, values.size))
    null = np.abs(signs @ values)
    p_value = float((np.count_nonzero(null >= observed - 1e-12) + 1) / (permutations + 1))
    return {
        "b": b,
        "c": c,
        "clusters_with_discordance": int(values.size),
        "statistic": observed,
        "p_value": p_value,
        "permutations": permutations,
        "test": "cluster-level sign-flip (SAST rule)",
    }


def cochran_q(matrix: Sequence[Sequence[bool]]) -> dict[str, Any]:
    data = np.array(matrix, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("cochran_q wymaga macierzy n x k, gdzie k >= 2")
    k = data.shape[1]
    column_sums = data.sum(axis=0)
    row_sums = data.sum(axis=1)
    numerator = (k - 1) * (k * float(np.sum(column_sums**2)) - float(column_sums.sum()) ** 2)
    denominator = k * float(row_sums.sum()) - float(np.sum(row_sums**2))
    if denominator == 0:
        return {"q": None, "df": k - 1, "p_value": 1.0, "test": "Q Cochrana"}
    q = numerator / denominator
    return {
        "q": q,
        "df": k - 1,
        "p_value": float(stats.chi2.sf(q, k - 1)),
        "test": "Q Cochrana",
    }


def cochran_q_cluster_permutation(
    matrix: Sequence[Sequence[bool]],
    clusters: Sequence[Any],
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    data = np.array(matrix, dtype=float)
    observed = cochran_q(matrix)
    if observed["q"] is None:
        return {**observed, "p_value_cluster": 1.0, "permutations": 0}

    labels = np.array([str(cluster) for cluster in clusters])
    groups = [np.flatnonzero(labels == name) for name in np.unique(labels)]
    k = data.shape[1]
    rng = np.random.default_rng(seed)

    def _q(values: np.ndarray) -> float:
        column_sums = values.sum(axis=0)
        row_sums = values.sum(axis=1)
        denominator = k * float(row_sums.sum()) - float(np.sum(row_sums**2))
        if denominator == 0:
            return 0.0
        return (k - 1) * (k * float(np.sum(column_sums**2)) - float(column_sums.sum()) ** 2) / denominator

    exceed = 0
    for _ in range(permutations):
        shuffled = np.empty_like(data)
        for rows in groups:
            # Permutacja etykiet raz na klaster, nie na ostrzeżenie — inaczej test wytwarza dowód z zależności.
            shuffled[rows] = data[rows][:, rng.permutation(k)]
        if _q(shuffled) >= observed["q"] - 1e-12:
            exceed += 1
    return {
        **observed,
        "p_value_cluster": (exceed + 1) / (permutations + 1),
        "permutations": permutations,
        "clusters": len(groups),
    }


def fleiss_kappa(ratings: Sequence[Sequence[int]]) -> dict[str, Any]:
    data = np.array(ratings, dtype=float)
    if data.size == 0:
        return {"kappa": None, "subjects": 0, "raters": 0}
    n_per_subject = data.sum(axis=1)
    if not np.all(n_per_subject == n_per_subject[0]):
        usable = n_per_subject == n_per_subject.max()
        data = data[usable]
        n_per_subject = data.sum(axis=1)
    n = int(n_per_subject[0])
    subjects = data.shape[0]
    if n < 2 or subjects == 0:
        return {"kappa": None, "subjects": subjects, "raters": n}
    agreement = (np.sum(data**2, axis=1) - n) / (n * (n - 1))
    p_bar = float(np.mean(agreement))
    proportions = data.sum(axis=0) / (subjects * n)
    pe = float(np.sum(proportions**2))
    kappa = None if pe == 1 else (p_bar - pe) / (1 - pe)
    return {
        "kappa": kappa,
        "observed_agreement": p_bar,
        "expected_agreement": pe,
        "subjects": subjects,
        "raters": n,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted(p_values.items(), key=lambda item: item[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for position, (name, value) in enumerate(items):
        candidate = (m - position) * value
        running = max(running, min(1.0, candidate))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def friedman(columns: Sequence[Sequence[float]]) -> dict[str, Any]:
    arrays = [np.array(column, dtype=float) for column in columns]
    statistic, p_value = stats.friedmanchisquare(*arrays)
    return {
        "statistic": float(statistic),
        "df": len(arrays) - 1,
        "p_value": float(p_value),
        "test": "test Friedmana",
    }


def wilcoxon_signed_rank(first: Sequence[float], second: Sequence[float]) -> dict[str, Any]:
    differences = np.array(first, dtype=float) - np.array(second, dtype=float)
    if np.all(differences == 0):
        return {"statistic": None, "p_value": 1.0, "test": "test Wilcoxona dla par", "pairs": len(differences)}
    result = stats.wilcoxon(first, second, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "pairs": int(np.count_nonzero(differences)),
        "test": "test Wilcoxona dla par",
    }


def cohens_h(p1: float | None, p2: float | None) -> float | None:
    if p1 is None or p2 is None:
        return None
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
