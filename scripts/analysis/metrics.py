from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from analysis.dataset import Finding

SEVERITY_ORDER = {"BLOCKER": 0, "CRITICAL": 1, "MAJOR": 2, "MINOR": 3, "INFO": 4}


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def accuracy(self) -> float | None:
        return _ratio(self.tp + self.tn, self.total)

    @property
    def fp_reduction(self) -> float | None:
        return _ratio(self.tn, self.fp + self.tn)

    @property
    def lost_tp(self) -> int:
        return self.fn

    @property
    def lost_tp_share(self) -> float | None:
        return _ratio(self.fn, self.tp + self.fn)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "evaluated": self.total,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "fp_reduction": self.fp_reduction,
            "lost_tp": self.lost_tp,
            "lost_tp_share": self.lost_tp_share,
        }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def confusion(pairs: Iterable[tuple[bool, bool]]) -> Confusion:
    tp = fp = fn = tn = 0
    for label, predicted in pairs:
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = z * ((proportion * (1 - proportion) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


Predictor = Callable[[Finding], bool]


def model_predictor(model: str) -> Predictor:
    def predict(finding: Finding) -> bool:
        decision = finding.decisions.get(model)
        if decision is None or decision.verdict is None:
            raise KeyError(f"no verdict for {finding.key} / {model}")
        return decision.verdict

    return predict


def baseline_predictor(name: str, findings: Sequence[Finding]) -> tuple[Predictor, str]:
    if name == "B0":
        return (lambda finding: True), "wszystkie ostrzeżenia zachowane"
    if name == "B1":
        return (lambda finding: False), "wszystkie ostrzeżenia odrzucone"
    if name == "B2":
        return _fit_metadata_heuristic(findings)
    raise ValueError(f"unknown baseline: {name}")


def _metadata_candidates(tool: str) -> list[tuple[str, Predictor]]:
    if tool == "spotbugs":
        candidates: list[tuple[str, Predictor]] = []
        for threshold in (4, 6, 8, 10, 12, 14, 16, 18, 20):
            candidates.append(
                (f"rank <= {threshold}", lambda f, t=threshold: _as_int(f.metadata.get("rank"), 99) <= t)
            )
        candidates.append(("priority == 1", lambda f: _as_int(f.metadata.get("priority"), 99) <= 1))
        candidates.append(
            ("kategoria w {CORRECTNESS, MT_CORRECTNESS, SECURITY}",
             lambda f: str(f.metadata.get("category")) in {"CORRECTNESS", "MT_CORRECTNESS", "SECURITY"})
        )
        candidates.append(("cweid ustawiony", lambda f: bool(f.metadata.get("cweid"))))
        return candidates
    if tool == "sonarqube":
        candidates = []
        for severity in ("BLOCKER", "CRITICAL", "MAJOR"):
            candidates.append(
                (
                    f"severity <= {severity}",
                    lambda f, s=severity: SEVERITY_ORDER.get(str(f.metadata.get("severity")), 9)
                    <= SEVERITY_ORDER[s],
                )
            )
        candidates.append(
            ("typ w {BUG, VULNERABILITY}", lambda f: str(f.metadata.get("issue_type")) in {"BUG", "VULNERABILITY"})
        )
        candidates.append(
            (
                "typ w {BUG, VULNERABILITY} i severity <= BLOCKER",
                lambda f: str(f.metadata.get("issue_type")) in {"BUG", "VULNERABILITY"}
                and SEVERITY_ORDER.get(str(f.metadata.get("severity")), 9) <= 0,
            )
        )
        return candidates
    if tool == "error-prone":
        return [("narzędzie podaje sugerowaną poprawkę", lambda f: bool(f.metadata.get("suggestion")))]
    return []


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _fit_metadata_heuristic(findings: Sequence[Finding]) -> tuple[Predictor, str]:
    tools = {finding.tool for finding in findings}
    chosen: dict[str, Predictor] = {}
    labels: list[str] = []
    for tool in sorted(tools):
        subset = [finding for finding in findings if finding.tool == tool]
        candidates = _metadata_candidates(tool)
        best_predictor: Predictor = lambda finding: True
        best_label = "brak użytecznych metadanych (degeneruje do B0)"
        best_f1 = confusion((f.label, True) for f in subset).f1 or 0.0
        for label, predictor in candidates:
            score = confusion((f.label, predictor(f)) for f in subset).f1
            if score is not None and score > best_f1:
                best_f1, best_predictor, best_label = score, predictor, label
        chosen[tool] = best_predictor
        labels.append(f"{tool}: {best_label}")

    def predict(finding: Finding) -> bool:
        predictor = chosen.get(finding.tool)
        return True if predictor is None else predictor(finding)

    return predict, "; ".join(labels)


def metadata_candidate_report(findings: Sequence[Finding]) -> dict[str, list[dict[str, Any]]]:
    report: dict[str, list[dict[str, Any]]] = {}
    for tool in sorted({finding.tool for finding in findings}):
        subset = [finding for finding in findings if finding.tool == tool]
        rows = [{"predicate": "B0 (wszystko zachowane)", **confusion((f.label, True) for f in subset).as_dict()}]
        for label, predictor in _metadata_candidates(tool):
            rows.append({"predicate": label, **confusion((f.label, predictor(f)) for f in subset).as_dict()})
        report[tool] = rows
    return report


def evaluate(findings: Sequence[Finding], predictor: Predictor) -> Confusion:
    return confusion((finding.label, predictor(finding)) for finding in findings)
