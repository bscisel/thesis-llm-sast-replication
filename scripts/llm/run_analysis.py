#!/usr/bin/env python3
"""Uruchamia postprocessing raportów statycznej analizy wybranym modelem."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from llm.base import FatalAPIError, LLMAnalyzer, _resolve_temperature
from llm.claude_analyzer import ClaudeAnalyzer
from llm.gpt_analyzer import GPTAnalyzer
from llm.ollama_analyzer import OllamaAnalyzer
from llm.oss_analyzer import OpenRouterAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOOLS = ["spotbugs", "error_prone", "sonarqube"]

MODELS = ["claude", "gpt", "qwen", "oss", "luna", "oss20", "sonnet"]

TOOL_FILENAMES: dict[str, str] = {
    "spotbugs": "spotbugs.json",
    "error_prone": "error_prone.json",
    "sonarqube": "sonarqube.json",
}


def _load_env() -> None:
    """Wczytuje .env z korzenia projektu."""
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        logger.warning("No .env found in %s - using the environment variables.", env_file)


def _resolve_num_runs(cli_value: int | None) -> int:
    """Ustala liczbę powtórzeń: wiersz poleceń, potem LLM_NUM_RUNS, potem 3."""
    if cli_value is not None:
        return cli_value
    env_value = os.getenv("LLM_NUM_RUNS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            logger.warning("Invalid LLM_NUM_RUNS='%s', using default 3.", env_value)
    return 3


DEFAULT_CONCURRENCY: dict[str, int] = {
    "claude": 3,
    "gpt": 3,
    "qwen": 1,
    "oss": 3,
    "oss20": 30,
    "luna": 30,
    "sonnet": 3,
}


def _resolve_concurrency(cli_value: int | None, model: str) -> int:
    """Ustala zrównoleglenie: wiersz poleceń, potem zmienna środowiskowa, potem wartość dla modelu."""
    if cli_value is not None:
        return max(1, cli_value)
    env_value = os.getenv(f"LLM_CONCURRENCY_{model.upper()}")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            logger.warning(
                "Invalid LLM_CONCURRENCY_%s='%s', using default.", model.upper(), env_value
            )
    return DEFAULT_CONCURRENCY.get(model, 1)


def _load_run_info(run_dir: Path) -> dict:
    path = run_dir / "run_info.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Could not read the run metadata: %s", path)
        return {}


SYSTEM_PROMPT_BY_TARGET: dict[str, str] = {
    "benchmark": "benchmark",
    "jetty": "jetty",
}


def _resolve_system_prompt(cli_value: str | None, run_dir: Path) -> str:
    if cli_value:
        return cli_value

    target = _load_run_info(run_dir).get("target_project") or {}
    key = str(target.get("key") or "").lower()
    matches = {
        variant for needle, variant in SYSTEM_PROMPT_BY_TARGET.items() if needle in key
    }
    if len(matches) == 1:
        variant = matches.pop()
        logger.info("Wariant promptu systemowego '%s' wyprowadzony z projektu '%s'.", variant, key)
        return variant

    logger.error(
        "Cannot derive the system prompt variant from target_project.key=%r in %s. "
        "Pass --system-prompt explicitly.",
        target.get("key"),
        run_dir / "run_info.json",
    )
    sys.exit(1)


def _resolve_source_root(cli_value: str | None, run_dir: Path) -> Path:
    """Ustala katalog źródeł: wiersz poleceń, potem run_info, potem SOURCE_ROOT."""
    value = cli_value
    if not value:
        run_info = _load_run_info(run_dir)
        value = ((run_info.get("target_project") or {}).get("directory") or "").strip()
    if not value:
        value = os.getenv("SOURCE_ROOT")
    if not value:
        logger.error(
            "No source directory set. Pass --source-root, set SOURCE_ROOT, or use a run whose run_info.json carries target_project.directory."
        )
        sys.exit(1)
    path = Path(value)
    if not path.exists():
        logger.error("Source directory does not exist: %s", path)
        sys.exit(1)
    return path


def _build_analyzer(model: str, shared_kwargs: dict, dry_run: bool = False) -> LLMAnalyzer:
    """Tworzy analizator właściwy dla wskazanego modelu."""
    if model in ("claude", "sonnet"):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if model == "claude":
            claude_model = os.getenv("ANTHROPIC_MODEL", ClaudeAnalyzer.DEFAULT_MODEL)
        else:
            claude_model = os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-5")
        base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
        if not api_key and not base_url and not dry_run:
            logger.error("Set ANTHROPIC_API_KEY, or ANTHROPIC_BASE_URL for a custom endpoint.")
            sys.exit(1)
        return ClaudeAnalyzer(
            api_key=api_key or "unused",
            model=claude_model,
            base_url=base_url,
            **shared_kwargs,
        )

    if model in ("gpt", "luna"):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if model == "gpt":
            openai_model = os.getenv("OPENAI_MODEL", GPTAnalyzer.DEFAULT_MODEL)
        else:
            openai_model = os.getenv("OPENAI_MODEL_LUNA", "gpt-5.6-luna")
        if not api_key and not dry_run:
            logger.error("Nie ustawiono OPENAI_API_KEY.")
            sys.exit(1)
        return GPTAnalyzer(api_key=api_key or "dry-run", model=openai_model, **shared_kwargs)

    if model in ("oss", "oss20"):
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if model == "oss":
            oss_model = os.getenv("OPENROUTER_MODEL", OpenRouterAnalyzer.DEFAULT_MODEL)
            provider = os.getenv("OPENROUTER_PROVIDER", "").strip() or OpenRouterAnalyzer.DEFAULT_PROVIDER
        else:
            oss_model = os.getenv("OPENROUTER_MODEL_SMALL", "openai/gpt-oss-20b")
            provider = os.getenv("OPENROUTER_PROVIDER_SMALL", "").strip() or OpenRouterAnalyzer.DEFAULT_PROVIDER
        cache_control = os.getenv("OPENROUTER_CACHE_CONTROL", "").strip().lower() in ("1", "true", "yes")
        if not api_key and not dry_run:
            logger.error("Nie ustawiono OPENROUTER_API_KEY.")
            sys.exit(1)
        return OpenRouterAnalyzer(
            api_key=api_key or "dry-run",
            model=oss_model,
            provider=provider,
            cache_control=cache_control,
            **shared_kwargs,
        )

    if model == "qwen":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", OllamaAnalyzer.DEFAULT_MODEL)
        return OllamaAnalyzer(base_url=base_url, model=ollama_model, **shared_kwargs)

    raise ValueError(f"Nieznany model: {model}")


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _latest_run_dir(project_root: Path) -> Path:
    pattern = re.compile(r"^[0-9]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    runs = [path for path in (project_root / "results").iterdir() if path.is_dir() and pattern.match(path.name)]
    if not runs:
        logger.error("No run directories found in %s.", project_root / "results")
        sys.exit(1)
    return sorted(runs, key=lambda path: int(path.name.split("_", 1)[0]))[-1]


def _resolve_run_dir(project_root: Path, cli_value: str | None) -> Path:
    value = cli_value or os.getenv("RUN_DIR")
    if value:
        path = Path(value)
        if not path.is_absolute():
            if value.isdigit():
                run_id = f"{int(value):03d}"
                matches = sorted(
                    (project_root / "results").glob(f"{run_id}_*"),
                    key=lambda item: int(item.name.split("_", 1)[0]),
                )
                if not matches:
                    logger.error("Nie znaleziono przebiegu o numerze: %s", value)
                    sys.exit(1)
                path = matches[-1]
            else:
                path = project_root / value if value.startswith("results/") else project_root / "results" / value
    else:
        path = _latest_run_dir(project_root)

    if not path.exists():
        logger.error("Katalog przebiegu nie istnieje: %s", path)
        sys.exit(1)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs post-processing of static analysis reports with the chosen model."
    )
    parser.add_argument(
        "--model",
        choices=MODELS + ["all"],
        required=True,
        help="Model to use ('all' = all seven).",
    )
    parser.add_argument(
        "--tool",
        choices=TOOLS + ["all"],
        required=True,
        help="Which tool report to process ('all' = all three).",
    )
    parser.add_argument(
        "--source-root",
        metavar="PATH",
        help="Java source directory; takes precedence over run_info and SOURCE_ROOT.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        metavar="N",
        help="Repetitions per warning; takes precedence over LLM_NUM_RUNS, default 3.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        metavar="N",
        help="How many warnings to process in parallel. Model-dependent by default; "
             "model lokalny dzieli jedną kartę, więc dostaje mniej. Ma pierwszeństwo "
             "przed LLM_CONCURRENCY_<MODEL>.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Process only the first N warnings of a tool (for trials).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompts and make no API call at all.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--append-runs",
        action="store_true",
        help="Append repetitions to the existing files instead of replacing them.",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing LLM output files.",
    )
    parser.add_argument(
        "--run-dir",
        metavar="PATH",
        help="Run number or directory. Defaults to RUN_DIR or the latest run.",
    )
    parser.add_argument(
        "--system-prompt",
        choices=["benchmark", "jetty"],
        help="System prompt variant; derived from the run project when omitted.",
    )
    parser.add_argument(
        "--bez-kodu",
        action="store_true",
        help="Wariant ablacyjny B3 (H6): ten sam prompt z pominietym fragmentem zrodla. "
             "Model dostaje wylacznie komunikat narzedzia i metadane. Wyniki zapisuja sie "
             "w osobnym katalogu runu, nie nadpisuja przebiegu glownego.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.append_runs:
        parser.error(
            "--limit cannot be combined with --append-runs: the output is rebuilt "
            "from the processed findings, so every finding outside the limit would "
            "be dropped from the file together with the runs already paid for."
        )

    _load_env()

    models = MODELS if args.model == "all" else [args.model]
    tools = TOOLS if args.tool == "all" else [args.tool]
    num_runs = _resolve_num_runs(args.num_runs)
    temperature = _resolve_temperature(os.getenv("LLM_TEMPERATURE"))

    project_root = _project_root()
    run_dir = _resolve_run_dir(project_root, args.run_dir)
    source_root = _resolve_source_root(args.source_root, run_dir)
    system_prompt_variant = _resolve_system_prompt(args.system_prompt, run_dir)
    processed_dir = run_dir / "static_analysis" / "processed"
    llm_dir = run_dir / "llm"

    shared_kwargs = {
        "source_root": source_root,
        "num_runs": num_runs,
        "temperature": temperature,
        "bez_kodu": args.bez_kodu,
    }

    if args.bez_kodu:
        logger.warning(
            "B3 VARIANT: the source excerpt will NOT be inserted into the prompt. "
            "Metryki z tego przebiegu nie sa porownywalne z glownym inaczej "
            "niz jako ablacja (H6)."
        )

    if args.dry_run:
        logger.info("DRY RUN - no API call will be made.")
    logger.info("Katalog przebiegu: %s", run_dir)

    if not args.dry_run and not args.append_runs and not args.overwrite:
        existing_outputs = [
            llm_dir / model / f"{tool}.json"
            for model in models
            for tool in tools
            if (llm_dir / model / f"{tool}.json").exists()
        ]
        if existing_outputs:
            logger.error(
                "Refusing to start: %d output files already exist. "
                "Use --append-runs to add more runs or --overwrite to replace them.",
                len(existing_outputs),
            )
            for path in existing_outputs:
                logger.error("Existing output: %s", path)
            sys.exit(1)

    for model in models:
        concurrency = _resolve_concurrency(args.concurrency, model)
        analyzer = _build_analyzer(
            model, {**shared_kwargs, "concurrency": concurrency}, dry_run=args.dry_run
        )
        for tool in tools:
            findings_path = processed_dir / TOOL_FILENAMES[tool]
            if not findings_path.exists():
                logger.warning("No findings file, skipping: %s", findings_path)
                continue

            output_path = llm_dir / model / f"{tool}.json"
            logger.info(
                "Model: %s | Tool: %s | Repetitions: %s | Parallel: %d | Sources: %s",
                model,
                tool,
                num_runs if not args.dry_run else "dry-run",
                concurrency,
                source_root,
            )
            try:
                analyzer.analyze_tool(
                    findings_path,
                    output_path,
                    dry_run=args.dry_run,
                    append_runs=args.append_runs,
                    overwrite=args.overwrite,
                    system_prompt_variant=system_prompt_variant,
                    limit=args.limit,
                )
            except FatalAPIError as exc:
                logger.error("Przebieg przerwany — %s", exc)
                logger.error(
                    "Policzone findingi zostaly zapisane. Po usunieciu przyczyny "
                    "wznow tym samym poleceniem; nieudane findingi zostana doliczone."
                )
                sys.exit(1)
            except (FileExistsError, ValueError, RuntimeError) as exc:
                logger.error("%s", exc)
                sys.exit(1)

    if not args.dry_run:
        logger.info("Done.")


if __name__ == "__main__":
    main()
