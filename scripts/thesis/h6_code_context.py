#!/usr/bin/env python3
"""H6 (RQ6) — ile skuteczności modelu pochodzi z pokazanego kodu."""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.runs import find_run_dir


def verdicts(dataset, model, repetition):
    """Werdykt z jednego przebiegu, kluczowany findingiem. Brak przebiegu → pominięty."""
    out = {}
    for f in dataset.findings:
        d = f.decisions.get(model)
        if d is None or len(d.votes) <= repetition:
            continue
        out[f.key] = (d.votes[repetition], f.label, f.tool, f.rule)
    return out


def h6(dataset_main, dataset_ablation, repetition: int = 0, seed: int = 20260803) -> dict:
    """Porównanie przebiegu głównego z wariantem bez pokazanego kodu."""
    from analysis.tests import mcnemar_exact, discordance, mcnemar_cluster_signflip

    wspolne = sorted(set(dataset_main.models) & set(dataset_ablation.models))
    result = {"przebieg_glownego": repetition, "modele": {},
             "brakujace_modele": sorted(set(dataset_main.models) - set(dataset_ablation.models))}
    for m in wspolne:
        a, b = verdicts(dataset_main, m, repetition), verdicts(dataset_ablation, m, 0)
        keys = sorted(set(a) & set(b), key=str)
        if not keys:
            continue
        traf_a = [a[k][0] == a[k][1] for k in keys]
        traf_b = [b[k][0] == b[k][1] for k in keys]
        nb, nc = discordance(traf_a, traf_b)
        test_klastrowy = mcnemar_cluster_signflip(
            traf_a, traf_b, [a[k][3] for k in keys], seed=seed
        )
        ta, tb = sum(traf_a) / len(keys), sum(traf_b) / len(keys)
        per_tool = {}
        for tool in sorted({a[k][2] for k in keys}):
            kt = [k for k in keys if a[k][2] == tool]
            per_tool[tool] = {
                "n": len(kt),
                "z_kodem": sum(a[k][0] == a[k][1] for k in kt) / len(kt),
                "bez_kodu": sum(b[k][0] == b[k][1] for k in kt) / len(kt),
            }
        result["modele"][m] = {
            "n": len(keys), "z_kodem": ta, "bez_kodu": tb, "roznica": ta - tb,
            "mcnemar": mcnemar_exact(nb, nc), "mcnemar_cluster": test_klastrowy,
            "per_tool": per_tool,
            "zachowanych_z_kodem": sum(a[k][0] for k in keys) / len(keys),
            "zachowanych_bez_kodu": sum(b[k][0] for k in keys) / len(keys),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-run", type=int, default=6, help="run holding the main pass (default 6)")
    ap.add_argument("--ablation-run", type=int, default=7, help="run holding the B3 variant (default 7)")
    ap.add_argument("--repetition", type=int, default=0, help="which repetition of the main pass (0-2)")
    ap.add_argument("--json", type=Path, help="also write the result as JSON")
    ap.add_argument("--seed", type=int, default=20260803, help="ziarno permutacji sign-flip")
    args = ap.parse_args()

    from analysis.dataset import load_dataset
    from analysis.tests import mcnemar_exact, discordance, mcnemar_cluster_signflip

    dataset_main = load_dataset(find_run_dir(args.main_run), include_partial=True)
    dataset_ablation = load_dataset(find_run_dir(args.ablation_run), include_partial=True)

    result = h6(dataset_main, dataset_ablation, args.repetition, args.seed)
    result["glowny"] = Path(find_run_dir(args.main_run)).name
    result["ablacja"] = Path(find_run_dir(args.ablation_run)).name
    if result["brakujace_modele"]:
        print(f"WARNING: models missing from the B3 variant: {', '.join(result['brakujace_modele'])}\n")
    if not result["modele"]:
        raise SystemExit("no models in common - did the run without code finish?")

    print(f"H6 - contribution of the code context   (repetition {args.repetition + 1} of the main pass "
          f"wobec jedynego przebiegu B3)\n")
    print(f"{'model':<22}{'with code':>11}{'no code':>9}{'difference':>12}"
          f"{'p (exact)':>11}{'p (cluster)':>12}{'n':>6}")
    for m, v in result["modele"].items():
        print(f"{m:<22}{v['z_kodem']:>9.3f}{v['bez_kodu']:>10.3f}{v['roznica']:>+9.3f}"
              f"{v['mcnemar']['p_value']:>11.4g}{v['mcnemar_cluster']['p_value']:>12.4g}{v['n']:>6}")

    print(f"\n{'model':<22}" + "".join(f"{t[:12]:>13}" for t in ("error-prone", "sonarqube", "spotbugs")))
    for m, v in result["modele"].items():
        kom = []
        for t in ("error-prone", "sonarqube", "spotbugs"):
            d = v["per_tool"].get(t)
            kom.append(f"{d['z_kodem'] - d['bez_kodu']:>+13.3f}" if d else f"{'—':>13}")
        print(f"{m:<22}" + "".join(kom))
    print("\nPositive values: the code helps. Negative: the model does better without it.")

    print(f"\n{'model':<22}{'zachowanych z kodem':>21}{'bez kodu':>11}")
    for m, v in result["modele"].items():
        print(f"{m:<22}{v['zachowanych_z_kodem']:>21.3f}{v['zachowanych_bez_kodu']:>11.3f}")

    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nZapisano {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
