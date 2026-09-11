#!/usr/bin/env python3
"""RQ8 — czy modele poprawnie kategoryzują problemy wobec kategorii odniesienia."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.categorization import category_report


def main() -> None:
    import thesis._cli as cli

    args = cli.parser(__doc__).parse_args()
    dataset = cli.dataset_from(args)
    results = cli.header(dataset, args, "rule")
    results["RQ8"] = category_report(dataset)
    cli.save(cli.output_dir(args, dataset), "rq8.json", results)


if __name__ == "__main__":
    main()
