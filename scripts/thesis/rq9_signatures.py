#!/usr/bin/env python3
"""RQ9 — czy modele nadają ostrzeżeniom tego samego rodzaju spójne sygnatury problemu."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.signatures import signature_report


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(__doc__).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, cli.apply_scheme(args, []))
    results["RQ9"] = signature_report(dataset)
    cli.save(cli.output_dir(args, dataset), "rq9.json", results)


if __name__ == "__main__":
    main()
