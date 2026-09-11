#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.constants import ALL_TOOLS
from analysis.runs import find_run_dir as _find_run_dir

OUTPUT = ROOT / "tables"
MAIN_RUN, BENCHMARK_RUN = 6, 4

MODELS = [
    ("Claude Opus 5", "claude", "claude-opus-5"),
    ("Claude Sonnet 5", "sonnet", "claude-sonnet-5"),
    ("GPT-5.6 Terra", "gpt", "gpt-5.6-terra"),
    ("GPT-5.6 Luna", "luna", "gpt-5.6-luna"),
    ("gpt-oss-120b", "oss", "openai/gpt-oss-120b"),
    ("gpt-oss-20b", "oss20", "openai/gpt-oss-20b"),
    ("Qwen3.5 9B", "qwen", "qwen3.5:9b"),
]
VENDORS = {
    "claude": ("Anthropic", "closed", "high", "vendor API"),
    "sonnet": ("Anthropic", "closed", "medium", "vendor API"),
    "gpt": ("OpenAI", "closed", "medium", "vendor API"),
    "luna": ("OpenAI", "closed", "low", "vendor API"),
    "oss": ("OpenAI", "open", "116.8B", "OpenRouter"),
    "oss20": ("OpenAI", "open", "20.9B", "OpenRouter"),
    "qwen": ("Alibaba", "open", "9B", "own GPU"),
}


@lru_cache(maxsize=None)
def find_run_dir(spec):
    return _find_run_dir(spec)


@lru_cache(maxsize=None)
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def analysis(run: int, name: str):
    return read_json(Path(find_run_dir(run)) / "analysis" / name)


def num(value, places: int = 3) -> str:
    return "" if value is None else f"{value:.{places}f}".replace(".", ",")


def write(name: str, header: list[str], rows: list[list]) -> None:
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {name}  ({len(rows)} rows)")


def by_model(source: dict):
    for label, _, name in MODELS:
        entry = source.get(name)
        if entry is not None:
            yield label, entry


def table_4_1_models() -> None:
    pricing = read_json(ROOT / "docs" / "methodology" / "model-pricing.json")
    rows = []
    for label, short, _ in MODELS:
        vendor, weights, tier, access = VENDORS[short]
        price = pricing[short]
        local = short == "qwen"
        rows.append([
            label, vendor, weights, tier,
            "" if local else f"{price['input']:g}".replace(".", ","),
            "" if local else f"{price['output']:g}".replace(".", ","),
            access,
        ])
    write("table-4-1-models.csv",
          ["model", "vendor", "weights", "tier_or_size",
           "price_input_usd_per_million", "price_output_usd_per_million", "access"], rows)


def table_6_1_overview() -> None:
    data = analysis(MAIN_RUN, "metrics.json")["overview"]
    rows = []
    for label, entry in by_model(data):
        rows.append([label, num(entry["precision"]), num(entry["recall"]),
                     num(entry["f1"]), num(entry["accuracy"])])
    for label, key in (("raw report", "raw_report"), ("naive classifier", "naive_classifier")):
        entry = data[key]
        rows.append([label, num(entry["precision"]), num(entry["recall"]),
                     num(entry["f1"]), num(entry["accuracy"])])
    write("table-6-1-overview.csv", ["model", "precision", "recall", "f1", "accuracy"], rows)


def table_6_2_h1_balance() -> None:
    data = analysis(MAIN_RUN, "stats_tests.json")["H1"]
    rows = [
        [label, num(e["fp_reduction"]), num(e["fp_reduction_ci"][0]), num(e["fp_reduction_ci"][1]),
         e["lost_tp"], num(e["lost_tp_share"]),
         num(e["delta_f1"]), num(e["delta_f1_ci"][0]), num(e["delta_f1_ci"][1])]
        for label, e in by_model(data)
    ]
    write("table-6-2-h1-balance.csv",
          ["model", "fp_reduction", "ci_low", "ci_high", "lost_tp", "lost_tp_share",
           "delta_f1", "delta_f1_ci_low", "delta_f1_ci_high"], rows)


def table_6_3_h2_accuracy() -> None:
    stats = analysis(MAIN_RUN, "stats_tests.json")
    scope = stats["H2"][ALL_TOOLS]
    holm = stats["holm_family"]["adjusted"]
    rows = []
    for label, _, name in MODELS:
        entry = scope["vs_naive"].get(name)
        if entry is None:
            continue
        rows.append([label, num(entry["accuracy"]), num(entry["difference"]),
                     num(entry["ci_low"]), num(entry["ci_high"]),
                     num(holm[f"H2/naive/{name}"], 5)])
    write("table-6-3-h2-accuracy.csv",
          ["model", "accuracy", "difference_vs_naive", "ci_low", "ci_high", "p_holm"], rows)


def table_6_4_h3_stability() -> None:
    data = analysis(MAIN_RUN, "stats_tests.json")["H3"]["models"]
    rows = [
        [label, num(e["unanimity"]), num(e["unanimity_ci"][0]), num(e["unanimity_ci"][1]),
         num(e["fleiss_kappa"]), num(e["fleiss_kappa_ci"][0]), num(e["fleiss_kappa_ci"][1])]
        for label, e in by_model(data)
    ]
    write("table-6-4-h3-stability.csv",
          ["model", "unanimity", "ci_low", "ci_high", "fleiss_kappa", "kappa_ci_low", "kappa_ci_high"],
          rows)


def table_6_5_h4_grounding() -> None:
    stats = analysis(MAIN_RUN, "stats_tests.json")
    holm = stats["holm_family"]["adjusted"]
    rows = []
    for label, _, name in MODELS:
        entry = stats["H4"]["models"].get(name)
        if entry is None:
            continue
        rows.append([label, num(entry["reasoning"]), num(entry["message"]), num(entry["difference"]),
                     num(entry["ci_low"]), num(entry["ci_high"]), num(holm[f"H4/{name}"], 5)])
    write("table-6-5-h4-grounding.csv",
          ["model", "reasoning", "tool_message", "difference", "ci_low", "ci_high", "p_holm"], rows)


def table_6_6_h5_cost() -> None:
    data = analysis(MAIN_RUN, "stats_tests.json")["H5"]
    rows = []
    for label, short, name in MODELS:
        usage = data["usage"].get(name, {})
        ratio = data["ratios"].get(name)
        rows.append([
            label,
            "" if short == "qwen" else num(usage.get("cost_usd_per_finding"), 4),
            num(ratio["f1_ratio"]) if ratio else num(1.0),
            num(ratio["f1_ratio_ci"][0]) if ratio else "",
            num(ratio["f1_ratio_ci"][1]) if ratio else "",
        ])
    write("table-6-6-h5-cost.csv",
          ["model", "cost_usd_per_finding", "f1_ratio", "ci_low", "ci_high"], rows)


def table_6_7_h6_code_context() -> None:
    stats = analysis(MAIN_RUN, "stats_tests.json")
    holm = stats["holm_family"]["adjusted"]
    rows = []
    for label, _, name in MODELS:
        entry = stats["H6"]["models"].get(name)
        if entry is None:
            continue
        rows.append([label, num(entry["with_code"]), num(entry["without_code"]),
                     num(entry["difference"]), num(entry["ci_low"]), num(entry["ci_high"]),
                     num(holm[f"H6/{name}"], 5),
                     num(entry["kept_with_code"]), num(entry["kept_without_code"])])
    write("table-6-7-h6-code-context.csv",
          ["model", "with_code", "without_code", "difference", "ci_low", "ci_high",
           "p_holm", "kept_with_code", "kept_without_code"], rows)


def table_6_8_categorisation() -> None:
    data = analysis(MAIN_RUN, "stats_tests.json")["RQ8"]
    baseline = data["tool_label_baseline"]
    mapped = baseline["models_on_same_subset"]
    rows = []
    for label, _, name in MODELS:
        entry = data["models"].get(name)
        if entry is None:
            continue
        rows.append([label, num(entry["accuracy"]), num(mapped.get(name, {}).get("accuracy"))])
    rows.append(["tool label mapped", "", num(baseline["accuracy"])])
    write("table-6-8-categorisation.csv", ["model", "accuracy_all", "accuracy_mapped_subset"], rows)


def table_6_9_signatures() -> None:
    data = analysis(MAIN_RUN, "stats_tests.json")["RQ9"]
    rows = []
    for label, entry in by_model(data["models"]):
        merges = entry["cross_tool_merges"]
        rows.append([label,
                     num(entry["within_rule_identical_pair_share"]),
                     num(entry["within_rule_close_pair_share"]),
                     merges["signatures"], merges["groups_at_similarity_threshold"]])
    write("table-6-9-signatures.csv",
          ["model", "within_rule_exact", "within_rule_similar",
           "cross_tool_groups_exact", "cross_tool_groups_similar"], rows)


def table_6_10_benchmark() -> None:
    data = analysis(BENCHMARK_RUN, "metrics.json")["overview"]
    total = analysis(BENCHMARK_RUN, "metrics.json")["true_positives"]
    rows = []
    for label, entry in by_model(data):
        rows.append([label, entry["fn"], num(entry["fn"] / total if total else None),
                     num(entry["recall"])])
    write("table-6-10-benchmark.csv", ["model", "lost_tp", "lost_tp_share", "recall"], rows)


def table_7_1_error_unanimity() -> None:
    data = analysis(MAIN_RUN, "error_analysis.json")["error_unanimity"]
    rows = [
        [label, e["errors"], e["unanimous"], e["split"], num(e["unanimous_share"])]
        for label, e in by_model(data)
    ]
    write("table-7-1-error-unanimity.csv",
          ["model", "errors", "unanimous", "split", "unanimous_share"], rows)


def table_7_2_reasoning_quality() -> None:
    data = analysis(MAIN_RUN, "rq7_reading.json")["criteria"]
    rows = []
    for criterion, distribution in data.items():
        yes = distribution.get("tak", 0) + distribution.get("konkretny", 0)
        partial = (distribution.get("częściowo", 0) + distribution.get("niejednoznaczne", 0))
        no = distribution.get("nie", 0) + distribution.get("dekoracyjny", 0)
        rows.append([criterion, yes, partial, no])
    write("table-7-2-reasoning-quality.csv", ["criterion", "yes", "partial", "no"], rows)


TABLES = [
    table_4_1_models,
    table_6_1_overview,
    table_6_2_h1_balance,
    table_6_3_h2_accuracy,
    table_6_4_h3_stability,
    table_6_5_h4_grounding,
    table_6_6_h5_cost,
    table_6_7_h6_code_context,
    table_6_8_categorisation,
    table_6_9_signatures,
    table_6_10_benchmark,
    table_7_1_error_unanimity,
    table_7_2_reasoning_quality,
]


def main() -> None:
    print(f"Writing {len(TABLES)} tables to {OUTPUT}")
    for table in TABLES:
        try:
            table()
        except (FileNotFoundError, KeyError, StopIteration) as error:
            print(f"  {table.__name__}: skipped ({type(error).__name__}: {error})")


if __name__ == "__main__":
    main()
