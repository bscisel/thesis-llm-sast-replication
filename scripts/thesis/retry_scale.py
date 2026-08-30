"""Skala powtórek nieudanych wywołań: ile przebiegów trzeba było powtórzyć i u których modeli."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.runs import find_run_dir


TOOLS = {"spotbugs", "error_prone", "sonarqube"}


def policz(directory: Path) -> dict:
    per_model = collections.defaultdict(lambda: {"przebiegi": 0, "powtorzone": 0})
    powody = collections.Counter()

    for path in sorted(directory.glob("llm/*/*.json")):
        # Migawki sprzed przebiegu powtórkowego zawyżyłyby i mianownik, i liczbę ponowień.
        if path.stem not in TOOLS:
            continue
        model = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8"))
        for finding in data.get("findings", []):
            for repetition in finding.get("runs", []):
                per_model[model]["przebiegi"] += 1
                if not repetition.get("retry_of_truncated"):
                    continue
                per_model[model]["powtorzone"] += 1
                pierwotny = repetition.get("superseded") or {}
                powody[pierwotny.get("stop_reason") or "nieznany"] += 1

    repetitions = sum(m["przebiegi"] for m in per_model.values())
    powtorzone = sum(m["powtorzone"] for m in per_model.values())
    return {
        "run": directory.name,
        "przebiegi": repetitions,
        "powtorzone": powtorzone,
        "udzial": powtorzone / repetitions if repetitions else 0.0,
        "powody_pierwotnego_niepowodzenia": dict(powody),
        "per_model": {k: dict(v) for k, v in sorted(per_model.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="4,6,7", help="numery katalogów runów po przecinku")
    ap.add_argument("--target-run", type=int, default=6, help="run, w którego analysis/ zapisać wynik")
    args = ap.parse_args()

    numery = [int(n) for n in args.runs.split(",")]
    results = [policz(find_run_dir(n)) for n in numery]

    repetitions = sum(entry["przebiegi"] for entry in results)
    powtorzone = sum(entry["powtorzone"] for entry in results)
    powody = collections.Counter()
    for entry in results:
        powody.update(entry["powody_pierwotnego_niepowodzenia"])

    raport = {
        "runs": results,
        "lacznie": {
            "przebiegi": repetitions,
            "powtorzone": powtorzone,
            "udzial": powtorzone / repetitions if repetitions else 0.0,
            "powody_pierwotnego_niepowodzenia": dict(powody),
        },
    }

    directory = find_run_dir(args.target_run) / "analysis"
    directory.mkdir(exist_ok=True)
    cel = directory / "retry_scale.json"
    cel.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding="utf-8")

    print(cel)
    for entry in results:
        print(f"\n{entry['run']}: {entry['powtorzone']} powtórek na {entry['przebiegi']} przebiegów "
              f"({entry['udzial']:.3%})")
        for model, stats in entry["per_model"].items():
            if stats["powtorzone"]:
                print(f"   {model:<10} {stats['powtorzone']:>3} z {stats['przebiegi']}")
    print(f"\nŁĄCZNIE: {powtorzone} powtórek na {repetitions} przebiegów "
          f"({powtorzone / repetitions:.3%})")
    print(f"powody pierwotnego niepowodzenia: {dict(powody)}")


if __name__ == "__main__":
    main()
