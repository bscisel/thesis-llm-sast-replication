#!/usr/bin/env python3
"""Writes one CSV per table in the thesis, straight from the stored results."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results"

from analysis.constants import ALL_TOOLS, CATEGORIES
from analysis.runs import find_run_dir as _find_run_dir


@lru_cache(maxsize=None)
def find_run_dir(spec):
    return _find_run_dir(spec)
def _pricing_path() -> Path:
    for candidate in (ROOT / "docs" / "methodology" / "model-pricing.json", ROOT / "model-pricing.json"):
        if candidate.exists():
            return candidate
    return ROOT / "model-pricing.json"


OUTPUT = ROOT / "tables"

MODELS = [
    ("claude-opus-5", "Claude Opus 5", "claude"),
    ("claude-sonnet-5", "Claude Sonnet 5", "sonnet"),
    ("gpt-5.6-terra", "GPT-5.6 Terra", "gpt"),
    ("gpt-5.6-luna", "GPT-5.6 Luna", "luna"),
    ("openai/gpt-oss-120b", "gpt-oss-120b", "oss"),
    ("openai/gpt-oss-20b", "gpt-oss-20b", "oss20"),
    ("qwen3.5:9b", "Qwen3.5 9B", "qwen"),
]
TOOLS = [("spotbugs", "SpotBugs"), ("error-prone", "Error Prone"), ("sonarqube", "SonarQube")]




@lru_cache(maxsize=None)
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def num(value, places: int = 3) -> str:
    """Number in Polish notation, matching the thesis; empty when undefined."""
    if value is None:
        return ""
    return f"{value:.{places}f}".replace(".", ",")


def write(name: str, header: list[str], rows: list[list]) -> None:
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {name}  ({len(rows)} rows)")


def table_6_1() -> None:
    pricing = read_json(_pricing_path())
    described = {
        "claude": ("Anthropic", "closed", "high", "vendor API"),
        "sonnet": ("Anthropic", "closed", "medium", "vendor API"),
        "gpt": ("OpenAI", "closed", "medium", "vendor API"),
        "luna": ("OpenAI", "closed", "low", "vendor API"),
        "oss": ("OpenAI", "open", "116.8B", "OpenRouter"),
        "oss20": ("OpenAI", "open", "20.9B", "OpenRouter"),
        "qwen": ("Alibaba", "open", "9B", "own GPU"),
    }
    rows = []
    for _, label, key in MODELS:
        vendor, weights, tier, access = described[key]
        price = pricing[key]
        local = key == "qwen"
        rows.append([
            label, vendor, weights, tier,
            "" if local else f"{price['input']:g}".replace(".", ","),
            "" if local else f"{price['output']:g}".replace(".", ","),
            access,
        ])
    write("table-6-1-models.csv",
          ["model", "vendor", "weights", "tier_or_size",
           "price_input_usd_per_million", "price_output_usd_per_million", "access"], rows)


def table_7_01() -> None:
    metrics = read_json(find_run_dir("006") / "analysis" / "metrics.json")
    labels = dict(TOOLS + [(ALL_TOOLS, "all")])
    rows = []
    for baseline in ("B0", "B1", "B2"):
        for tool in ("spotbugs", "error-prone", "sonarqube", ALL_TOOLS):
            row = next(r for r in metrics["rows"] if r["approach"] == baseline and r["tool"] == tool)
            rows.append([baseline, labels[tool], num(row["precision"]), num(row["recall"]),
                         num(row["f1"]), num(row["accuracy"])])
    write("table-7-01-baselines.csv",
          ["baseline", "tool", "precision", "recall", "f1", "accuracy"], rows)


def table_7_02() -> None:
    stats = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")
    reduction, balance = stats["H1"]["H1a"][ALL_TOOLS], stats["H1"]["H1b"][ALL_TOOLS]
    rows = []
    for key, label, _ in MODELS:
        a, b = reduction[key], balance[key]
        rows.append([
            label, num(a["fp_reduction"]), num(a["fp_reduction_ci"][0]), num(a["fp_reduction_ci"][1]),
            b["lost_tp"], num(100 * b["lost_tp_share"], 1),
            num(b["delta_f1"]), num(b["delta_f1_ci"][0]), num(b["delta_f1_ci"][1]),
        ])
    write("table-7-02-reduction-and-balance.csv",
          ["model", "false_alarm_reduction", "reduction_ci_low", "reduction_ci_high",
           "lost_true_positives", "lost_share_percent",
           "delta_f1", "delta_f1_ci_low", "delta_f1_ci_high"], rows)


def table_7_03() -> None:
    metrics = read_json(find_run_dir("006") / "analysis" / "metrics.json")
    rows = []
    for key, label, _ in MODELS:
        row = next(r for r in metrics["rows"] if r["approach"] == key and r["tool"] == ALL_TOOLS)
        rows.append([label, num(row["accuracy"]), num(row["accuracy_ci_low"]), num(row["accuracy_ci_high"])])
    rows.sort(key=lambda r: r[1], reverse=True)
    write("table-7-03-accuracy.csv", ["model", "accuracy", "ci_low", "ci_high"], rows)


def table_7_04() -> None:
    stability = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")["H3"][ALL_TOOLS]
    rows = []
    for key, label, _ in MODELS:
        entry = stability[key]
        rows.append([
            label, num(entry["unanimity"]), num(entry["unanimity_ci"][0]), num(entry["unanimity_ci"][1]),
            num(entry["fleiss_kappa"]), num(entry["fleiss_kappa_ci"][0]), num(entry["fleiss_kappa_ci"][1]),
            num(entry["accuracy_spread"]["spread"]),
        ])
    rows.sort(key=lambda r: r[1], reverse=True)
    write("table-7-04-stability.csv",
          ["model", "unanimity", "unanimity_ci_low", "unanimity_ci_high",
           "fleiss_kappa", "kappa_ci_low", "kappa_ci_high", "accuracy_spread"], rows)


def table_7_05() -> None:
    """Grounding per tool; recomputed here because no result file stores it."""
    from analysis.dataset import load_dataset
    from analysis.grounding import (build_grounding, tool_roots, score_text,
                                    _looks_like_identifier, _tokens)

    dataset = load_dataset(str(find_run_dir("006")))
    if dataset.source_root is None or not dataset.source_root.exists():
        print("  table-7-05: skipped, Jetty sources not available")
        return
    roots = tool_roots(dataset.findings, dataset.source_root)
    grounding = {f.key: build_grounding(f, dataset.source_root, roots[f.tool])
                 for f in dataset.findings}
    rows = []
    for tool, label in TOOLS:
        findings = [f for f in dataset.findings
                    if f.tool == tool and grounding[f.key].context_available]
        texts = [f"{f.message} {f.description}" for f in findings]
        scored_texts = [score_text(text, grounding[f.key])
                        for text, f in zip(texts, findings)]
        tool_text = [s["grounded"] for s in scored_texts]
        names_written = [sum(1 for token in _tokens(text) if _looks_like_identifier(token))
                         for text in texts]
        names_from_excerpt = [s["hits"] for s in scored_texts]
        model_scores, beyond_scores = [], []
        for _, _, short in MODELS:
            hits, beyond = [], []
            for finding in findings:
                decision = finding.decisions.get(short)
                if decision is None or not decision.reasonings:
                    continue
                scored = score_text(decision.reasonings[0], grounding[finding.key])
                hits.append(scored["grounded"])
                beyond.append(scored["grounded_beyond_message"])
            if hits:
                model_scores.append(sum(hits) / len(hits))
                beyond_scores.append(sum(beyond) / len(beyond))
        rows.append([
            label, num(sum(tool_text) / len(tool_text)),
            num(sum(names_written) / len(names_written)),
            num(sum(names_from_excerpt) / len(names_from_excerpt)),
            num(min(model_scores)), num(max(model_scores)),
            num(min(beyond_scores)), num(max(beyond_scores)),
        ])
    write("table-7-05-grounding-per-tool.csv",
          ["tool", "tool_text", "names_written_by_tool", "names_from_excerpt",
           "models_min", "models_max",
           "beyond_tool_text_min", "beyond_tool_text_max"], rows)


def table_7_06() -> None:
    diagnostics = read_json(find_run_dir("006") / "analysis" / "grounding_diagnostics.json")
    ordering = diagnostics["czy_miara_porzadkuje_modele"]
    rows = []
    for _, label, short in MODELS:
        entry = ordering[short]
        rows.append([label, num(entry["nazw_na_tekst"]), num(entry["nazw_na_100_znakow"]),
                     f"{round(entry['srednia_dlugosc_znakow'])}"])
    rows.sort(key=lambda r: r[1], reverse=True)
    write("table-7-06-grounding-measure-variants.csv",
          ["model", "names_per_text", "names_per_100_chars", "mean_length_chars"], rows)


def table_7_07() -> None:
    stats = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")
    usage, ratios = stats["H5"]["usage"], stats["H5"]["ratios"]
    balance = stats["H1"]["H1b"][ALL_TOOLS]
    rows = []
    for key, label, _ in MODELS:
        ratio = ratios.get(key)
        rows.append([
            label, num(usage[key]["cost_usd_per_finding"], 4),
            num(ratio["f1_ratio"]) if ratio else "",
            num(ratio["f1_ratio_ci"][0]) if ratio else "",
            num(ratio["f1_ratio_ci"][1]) if ratio else "",
            balance[key]["lost_tp"],
        ])
    write("table-7-07-cost.csv",
          ["model", "cost_usd_per_finding", "f1_ratio_vs_opus",
           "ratio_ci_low", "ratio_ci_high", "lost_true_positives"], rows)


def table_7_08() -> None:
    ablation = read_json(find_run_dir("007") / "analysis" / "h6.json")["modele"]
    rows = []
    for _, label, short in MODELS:
        entry = ablation[short]
        rows.append([label, num(entry["z_kodem"]), num(entry["bez_kodu"]), num(entry["roznica"]),
                     num(entry["zachowanych_z_kodem"]), num(entry["zachowanych_bez_kodu"])])
    rows.sort(key=lambda r: r[3])
    write("table-7-08-code-context.csv",
          ["model", "accuracy_with_code", "accuracy_without_code", "accuracy_drop",
           "kept_as_true_with_code", "kept_as_true_without_code"], rows)


def table_7_09() -> None:
    rq8 = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")["RQ8"]
    subset = rq8["tool_label_baseline"]["modele_na_tym_samym_podzbiorze"]
    rows = []
    for key, label, _ in MODELS:
        rows.append([label, num(rq8["models"][key]["accuracy"]), num(subset[key]["trafnosc"])])
    rows.sort(key=lambda r: r[1], reverse=True)
    rows.append(["tool label mapped mechanically", "", num(rq8["tool_label_baseline"]["trafnosc"])])
    write("table-7-09-categorisation.csv",
          ["model", "accuracy_360_findings", "accuracy_198_mapped"], rows)


def table_7_10() -> None:
    confusion = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")["RQ8"]["confusion_summed"]
    rows = []
    for reference in CATEGORIES:
        row = confusion.get(reference, {})
        if not any(row.get(c) for c in CATEGORIES):
            continue
        rows.append([reference] + [row.get(c, 0) for c in CATEGORIES])
    write("table-7-10-confusion-matrix.csv",
          ["reference_category"] + [f"model_said_{c}" for c in CATEGORIES], rows)


def table_7_11() -> None:
    rq9 = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")["RQ9"]["models"]
    rows = []
    for key, label, _ in MODELS:
        entry = rq9[key]
        merges = entry["scalenia_miedzynarzedziowe"]
        rows.append([
            label, num(entry["within_rule_identical_pair_share"]),
            num(entry["within_rule_close_pair_share"]),
            merges["sygnatur"], merges["objetych_ostrzezen"],
            merges["grup_przy_progu_bliskosci"], merges["objetych_ostrzezen_przy_progu_bliskosci"],
        ])
    rows.sort(key=lambda r: r[1], reverse=True)
    write("table-7-11-signatures.csv",
          ["model", "convergence_exact", "convergence_at_080",
           "merge_groups_exact", "merge_findings_exact",
           "merge_groups_at_080", "merge_findings_at_080"], rows)


def table_7_12() -> None:
    clustering = read_json(find_run_dir("006") / "analysis" / "metrics.json")["clustering"]["per_tool_rule"]
    rows = []
    for tool, label in [("sonarqube", "SonarQube"), ("spotbugs", "SpotBugs"), ("error-prone", "Error Prone")]:
        entry = clustering[tool]
        rows.append([label, entry["clusters"], num(entry["icc"]), num(entry["design_effect"]),
                     f"{round(entry['effective_n'])}"])
    write("table-7-12-clustering-per-tool.csv",
          ["tool", "rules", "intraclass_correlation", "design_effect", "effective_sample_size"], rows)


def table_7_13(threshold: int = 5) -> None:
    raw = read_json(find_run_dir("006") / "analysis" / "metrics.json")["per_rule"]
    tool_labels = dict(TOOLS)
    rows = []
    for entry in raw:
        count = int(entry["n"])
        if count < threshold:
            continue
        accuracies = []
        for _, _, short in MODELS:
            true_pos, true_neg = entry.get(f"{short}_tp"), entry.get(f"{short}_tn")
            if true_pos in (None, ""):
                continue
            accuracies.append((int(true_pos) + int(true_neg)) / count)
        rows.append([tool_labels.get(entry["tool"], entry["tool"]), entry["type"], count,
                     num(float(entry["tp_share"]), 2), num(min(accuracies)), num(max(accuracies))])
    order = {"Error Prone": 0, "SonarQube": 1, "SpotBugs": 2}
    rows.sort(key=lambda r: (order.get(r[0], 9), -r[2], r[1]))
    write("table-7-13-rules.csv",
          ["tool", "rule", "findings", "true_positive_share",
           "accuracy_lowest", "accuracy_highest"], rows)


def table_7_14() -> None:
    stats = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")
    benchmark = read_json(find_run_dir("004") / "analysis" / "metrics.json")
    balance = stats["H1"]["H1b"][ALL_TOOLS]
    reduction = stats["H1"]["H1a"][ALL_TOOLS]
    benchmark_positives = sum(
        1 for f in read_json(find_run_dir("004") / "ground_truth.json")["findings"] if f["is_true_positive"])
    rows = []
    for key, label, _ in MODELS:
        row = next((r for r in benchmark["rows"] if r["approach"] == key and r["tool"] == ALL_TOOLS), None)
        lost = row["lost_tp"] if row else None
        rows.append([
            label,
            lost if lost is not None else "",
            num(100 * lost / benchmark_positives, 1) if lost is not None else "",
            num(reduction[key]["fp_reduction"]),
            balance[key]["lost_tp"], num(100 * balance[key]["lost_tp_share"], 1),
            num(balance[key]["delta_f1"]),
        ])
    rows.sort(key=lambda r: (r[1] if r[1] != "" else 999))
    write("table-7-14-benchmark-vs-jetty.csv",
          ["model", "benchmark_lost_true_positives", "benchmark_lost_share_percent",
           "jetty_false_alarm_reduction", "jetty_lost_true_positives",
           "jetty_lost_share_percent", "jetty_delta_f1"], rows)


def table_gate() -> None:
    """Unanimity gate; backs prose figures in section 7.3, not a numbered table."""
    gate = read_json(find_run_dir("006") / "analysis" / "consistency.json")["gate"]
    rows = []
    for _, label, short in MODELS:
        entry = gate[short]
        if not entry.get("available"):
            continue
        rows.append([
            label, entry["abstained"], entry["scored"],
            num(entry["abstained"] / entry["scored"]),
            num(entry["tp_share_abstained"]),
            num(entry["gated"]["f1"]), num(entry["majority"]["f1"]),
            num(entry["gated"]["precision"]), num(entry["majority"]["precision"]),
        ])
    rows.sort(key=lambda r: r[3])
    write("extra-unanimity-gate.csv",
          ["model", "handed_to_human", "scored", "handed_share", "true_positive_share_handed",
           "f1_gated", "f1_majority", "precision_gated", "precision_majority"], rows)


def table_instability() -> None:
    """Where instability concentrates; backs prose figures in section 7.5."""
    data = read_json(find_run_dir("006") / "analysis" / "consistency.json")
    hardness = data["hardness"]
    rows = [[entry["unstable_models"], entry["findings"], num(entry["tp_share"]), num(entry["accuracy"])]
            for entry in hardness]
    write("extra-instability-vs-accuracy.csv",
          ["unstable_models", "findings", "true_positive_share", "mean_accuracy"], rows)

    concentration = data["concentration"]
    observed, expected = concentration["observed"], concentration["expected"]
    rows = [[i, int(o), num(e, 1)] for i, (o, e) in enumerate(zip(observed, expected))]
    write("extra-instability-distribution.csv",
          ["unstable_models", "observed_findings", "expected_if_independent"], rows)


def table_grounding_per_model() -> None:
    """Grounding figures quoted in section 7.6 prose."""
    h4 = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")["H4"]["H4"]
    rows = []
    for key, label, _ in MODELS:
        entry = h4[key]
        rows.append([
            label, num(entry["reasoning_grounded"]), num(entry["message_grounded"]),
            num(entry["explanation_grounded"]), num(entry["grounded_beyond_message"]),
            num(entry["grounded_near_flagged_line"]),
            num(entry["reasoning_mcnemar_cluster"]["p_value"], 5),
        ])
    rows.sort(key=lambda r: r[1], reverse=True)
    write("extra-grounding-per-model.csv",
          ["model", "reasoning_grounded", "tool_text_grounded", "developer_explanation_grounded",
           "beyond_tool_text", "near_flagged_line", "p_signflip"], rows)


def table_h2_pairwise() -> None:
    """Pairwise comparisons and detectable differences quoted in section 7.4."""
    pairwise = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")["H2"][ALL_TOOLS]["pairwise"]
    rows = []
    for pair, entry in pairwise.items():
        rows.append([
            pair.replace(" vs ", " | "),
            num(abs(entry["accuracy_first"] - entry["accuracy_second"])),
            num(entry["mcnemar_cluster"]["p_value"], 4),
            num(entry["p_value_holm"], 4),
            num(entry["mde"]["mde_accuracy"]),
        ])
    rows.sort(key=lambda r: r[4])
    write("extra-h2-pairwise.csv",
          ["model_pair", "accuracy_difference", "p_signflip", "p_holm",
           "minimum_detectable_difference"], rows)


def table_clustering() -> None:
    """Intraclass correlations quoted in sections 7.1 and 7.4."""
    clustering = read_json(find_run_dir("006") / "analysis" / "metrics.json")["clustering"]
    rows = [["reference label", "rule", clustering["rule"]["clusters"] if "rule" in clustering
             else clustering["reguła"]["clusters"],
             num(clustering.get("rule", clustering.get("reguła"))["icc"]),
             num(clustering.get("rule", clustering.get("reguła"))["design_effect"]),
             f"{round(clustering.get('rule', clustering.get('reguła'))['effective_n'])}"]]
    for key, label, _ in MODELS:
        entry = clustering["per_model_correctness_rule"][key]
        rows.append([f"accuracy of {label}", "rule", entry["clusters"], num(entry["icc"]),
                     num(entry["design_effect"]), f"{round(entry['effective_n'])}"])
    write("extra-clustering.csv",
          ["quantity", "cluster_unit", "clusters", "intraclass_correlation",
           "design_effect", "effective_sample_size"], rows)


def table_code_context_per_tool() -> None:
    """Accuracy drop per tool, quoted in section 7.8."""
    ablation = read_json(find_run_dir("007") / "analysis" / "h6.json")["modele"]
    rows = []
    for _, label, short in MODELS:
        for tool, tool_label in TOOLS:
            entry = ablation[short]["per_tool"][tool]
            rows.append([label, tool_label, num(entry["z_kodem"]), num(entry["bez_kodu"]),
                         num(entry["z_kodem"] - entry["bez_kodu"])])
    for tool, tool_label in TOOLS:
        drops = [ablation[short]["per_tool"][tool]["z_kodem"] - ablation[short]["per_tool"][tool]["bez_kodu"]
                 for _, _, short in MODELS]
        rows.append(["mean over models", tool_label, "", "", num(sum(drops) / len(drops))])
    write("extra-code-context-per-tool.csv",
          ["model", "tool", "accuracy_with_code", "accuracy_without_code", "accuracy_drop"], rows)


def table_accuracy_per_stage() -> None:
    """Accuracy on each sampling stage separately, quoted in section 7.4."""
    rows = []
    for prefix, stage in (("003", "stage 1"), ("005", "stage 2"), ("006", "combined")):
        metrics = read_json(find_run_dir(prefix) / "analysis" / "metrics.json")
        for key, label, _ in MODELS:
            row = next((r for r in metrics["rows"]
                        if r["approach"] == key and r["tool"] == ALL_TOOLS), None)
            if row:
                rows.append([stage, label, num(row["accuracy"])])
    write("extra-accuracy-per-stage.csv", ["sample_stage", "model", "accuracy"], rows)


def table_test_summary() -> None:
    """Headline test statistics quoted in prose."""
    stats = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")
    cochran = stats["H2"][ALL_TOOLS]["cochran_q"]
    pairwise = stats["H2"][ALL_TOOLS]["pairwise"].values()
    mdes = sorted(e["mde"]["mde_accuracy"] for e in pairwise if e["mde"].get("available"))
    accuracies = [e["accuracy_first"] for e in pairwise] + [e["accuracy_second"] for e in pairwise]
    rows = [
        ["Cochran Q statistic", num(cochran["q"])],
        ["Cochran Q degrees of freedom", cochran["df"]],
        ["Cochran Q p, no clustering correction", num(cochran["p_value"], 4)],
        ["Cochran Q p, rule-level permutation", num(cochran["p_value_cluster"], 4)],
        ["lowest pairwise p", num(min(e["mcnemar_cluster"]["p_value"] for e in pairwise), 3)],
        ["median minimum detectable difference", num(mdes[len(mdes) // 2])],
        ["largest observed accuracy difference", num(max(accuracies) - min(accuracies))],
        ["Holm family size", stats["holm_family"]["size"]],
    ]
    write("extra-test-summary.csv", ["quantity", "value"], rows)


def table_repetition_unanimity() -> None:
    """How often three repetitions agree, per output field."""
    stats = read_json(find_run_dir("006") / "analysis" / "stats_tests.json")
    consistency = read_json(find_run_dir("006") / "analysis" / "consistency.json")
    rows = []
    for key, label, short in MODELS:
        rows.append([
            label,
            num(stats["H3"][ALL_TOOLS][key]["unanimity"]),
            num(stats["RQ8"]["models"][key]["unanimous_across_runs"]),
            num(stats["RQ9"]["models"][key]["unanimous_across_runs"]),
            num(consistency["instability_rate"][short]),
        ])
    rows.sort(key=lambda r: r[1], reverse=True)
    write("extra-repetition-unanimity.csv",
          ["model", "verdict_unanimity", "category_unanimity", "signature_unanimity",
           "instability_rate"], rows)

    concentration = consistency["concentration"]
    summary = [
        ["variance of unstable-model count, observed", num(concentration["variance_observed"])],
        ["variance expected under independence", num(concentration["variance_expected_mean"])],
        ["permutation p", num(concentration["p_value"], 5)],
        ["Fleiss kappa of the instability indicator", num(consistency["fleiss"]["kappa"])],
        ["weakest similarity inside a merge group",
         num(min(m["scalenia_miedzynarzedziowe"]["najnizsze_podobienstwo_w_grupie"]
                 for m in stats["RQ9"]["models"].values()
                 if m["scalenia_miedzynarzedziowe"]["najnizsze_podobienstwo_w_grupie"] is not None))],
    ]
    write("extra-instability-summary.csv", ["quantity", "value"], summary)


def table_categorisation_per_rule() -> None:
    """Categorisation accuracy per rule, summed over models."""
    from analysis.dataset import load_dataset

    dataset = load_dataset(str(find_run_dir("006")))
    per_rule: dict[str, list[bool]] = {}
    counts: dict[str, int] = {}
    for finding in dataset.findings:
        if finding.reference_category is None:
            continue
        counts[finding.rule] = counts.get(finding.rule, 0) + 1
        for _, _, short in MODELS:
            decision = finding.decisions.get(short)
            if decision is None or not decision.categories:
                continue
            chosen = decision.category
            per_rule.setdefault(finding.rule, []).append(chosen == finding.reference_category)
    rows = [[rule, counts[rule], num(sum(hits) / len(hits))]
            for rule, hits in sorted(per_rule.items()) if counts[rule] >= 5]
    rows.sort(key=lambda r: r[2])
    write("extra-categorisation-per-rule.csv",
          ["rule", "findings", "accuracy_over_models"], rows)


def main() -> None:
    print("Thesis tables:")
    for table in (table_6_1, table_7_01, table_7_02, table_7_03, table_7_04, table_7_05,
                  table_7_06, table_7_07, table_7_08, table_7_09, table_7_10, table_7_11,
                  table_7_12, table_7_13, table_7_14,
                  table_gate, table_instability, table_grounding_per_model,
                  table_h2_pairwise, table_clustering, table_code_context_per_tool,
                  table_accuracy_per_stage, table_test_summary,
                  table_repetition_unanimity, table_categorisation_per_rule):
        table()


if __name__ == "__main__":
    main()
