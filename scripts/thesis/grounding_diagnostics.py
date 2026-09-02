#!/usr/bin/env python3
"""Liczy wielkości pomocnicze miary osadzenia uzasadnień w kodzie jednym poleceniem."""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.runs import find_run_dir

def model_text(decision):
    return decision.reasonings[0] if decision and decision.reasonings else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=int, default=6)
    args = ap.parse_args()

    from analysis.dataset import load_dataset
    from analysis.grounding import (_build_code_context, _strip_line_prefixes, _strip_comments,
                                    _as_raw, build_grounding, tool_roots, score_text,
                                    IDENTIFIER, _looks_like_identifier)

    ds = load_dataset(find_run_dir(args.run_dir))
    roots = tool_roots(ds.findings, ds.source_root)
    gr = {f.key: build_grounding(f, ds.source_root, roots[f.tool]) for f in ds.findings}
    usable_items = [f for f in ds.findings if gr[f.key].context_available]

    def model_share(dataset, score_one):
        result_of = {}
        for m in ds.models:
            t = c = 0
            for f in dataset:
                d = f.decisions.get(m)
                if d is None or not d.reasonings:
                    continue
                c += 1
                t += bool(score_one(model_text(d), f))
            result_of[m] = t / c if c else float("nan")
        return result_of

    def message_share(dataset, score_one):
        return sum(bool(score_one(f"{f.message} {f.description}", f)) for f in dataset) / len(dataset)

    home = lambda txt, f: score_text(txt, gr[f.key])["grounded"]

    print("== HOW MANY CODE NAMES THE TOOL SUPPLIES BY ITSELF ==")
    print("The tool text goes into the prompt, so the model gets these names without reading the code.")
    print(f"{'tool':<14}{'names per warning':>22}{'n':>6}")
    for tool in sorted({f.tool for f in usable_items}):
        sub = [f for f in usable_items if f.tool == tool]
        mean = sum(len(gr[f.key].message_identifiers) for f in sub) / len(sub)
        print(f"{tool:<14}{mean:>22.2f}{len(sub):>6}")
    print()
    print("== H4 BROKEN DOWN BY TOOL ==")
    print("The aggregate hides the difference: with SpotBugs no model beats the baseline.")
    print(f"{'tool':<14}{'message':>10}{'models min':>12}{'models max':>13}{'n':>6}")
    for tool in sorted({f.tool for f in usable_items}):
        sub = [f for f in usable_items if f.tool == tool]
        u = model_share(sub, home)
        print(f"{tool:<14}{message_share(sub, home):>10.3f}"
              f"{min(u.values()):>12.3f}{max(u.values()):>13.3f}{len(sub):>6}")

    def tokens(txt, minlen, keep):
        return {t for t in IDENTIFIER.findall(txt or "")
                if len(t) >= minlen and (not keep or _looks_like_identifier(t))}

    window = {}
    for f in usable_items:
        ctx = _build_code_context(f.raw or _as_raw(f), ds.source_root)
        raw_item = "\n".join(k for _, k in _strip_line_prefixes(ctx))
        window[f.key] = (raw_item, _strip_comments(raw_item))

    def gap(strip_comments, keep, minlen=4):
        mod = comment_text = lm = lk = 0
        for f in usable_items:
            code = window[f.key][1 if strip_comments else 0]
            kt = tokens(code, minlen, keep)
            comment_text += bool(tokens(f"{f.message} {f.description}", minlen, keep) & kt)
            lk += 1
            for m in ds.models:
                d = f.decisions.get(m)
                if d and d.reasonings:
                    mod += bool(tokens(model_text(d), minlen, keep) & kt)
                    lm += 1
        return mod / lm, comment_text / lk

    print("\n== TOKENISATION VARIANTS (threshold 4 characters) ==")
    print(f"{'variant':<40}{'models':>9}{'message':>11}{'gap':>10}")
    for name, (uk, fl) in {
        "adopted (no comments, with filter)": (True, True),
        "no comments, no filter": (True, False),
        "with comments, with filter": (False, True),
        "every word (neither)": (False, False),
    }.items():
        a, b = gap(uk, fl)
        print(f"{name:<40}{a:>9.3f}{b:>11.3f}{a - b:>10.3f}")

    print("\n== SENSITIVITY TO THE LENGTH THRESHOLD ==")
    print(f"{'threshold':>10}{'models':>9}{'message':>11}{'gap':>10}")
    for minlen in (2, 3, 4, 5):
        a, b = gap(True, True, minlen)
        print(f"{minlen:>6}{a:>9.3f}{b:>11.3f}{a - b:>10.3f}")

    # Blok komentarza otwarty przed początkiem wycinka zostaje niewykryty — w wycinku widać samo domknięcie.
    truncated = 0
    for f in usable_items:
        code = window[f.key][0]
        truncated += code.count("*/") > code.count("/*")
    print("\n== EXCERPTS WITH A COMMENT BLOCK OPENED BEFORE THEIR START ==")
    print(f"{truncated} of {len(usable_items)} - the stripping does not detect these")

    def ranks(x):
        s = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for position, i in enumerate(s):
            r[i] = position + 1
        return r

    def spearman(a, b):
        ra, rb = ranks(a), ranks(b)
        n = len(a)
        sa, sb = sum(ra) / n, sum(rb) / n
        counts = sum((x - sa) * (y - sb) for x, y in zip(ra, rb))
        mia = (sum((x - sa) ** 2 for x in ra) * sum((y - sb) ** 2 for y in rb)) ** 0.5
        return counts / mia if mia else float("nan")

    code_tokens, comment_tokens = {}, {}
    for f in usable_items:
        kt = tokens(window[f.key][1], 4, True)
        km = tokens(f"{f.message} {f.description}", 4, True)
        code_tokens[f.key] = kt
        comment_tokens[f.key] = km

    def hits(field):
        result_of = {}
        for m in ds.models:
            totals = []
            for f in usable_items:
                d = f.decisions.get(m)
                texts = getattr(d, field, None) if d else None
                if not texts:
                    continue
                kt = code_tokens[f.key]
                km = comment_tokens[f.key]
                scored = tokens(texts[0], 4, True)
                totals.append(len((scored & kt) - km))
            result_of[m] = sum(totals) / len(totals) if totals else float("nan")
        return result_of

    print("\n== DOES THE MEASURE RANK THE MODELS ==")
    reasoning_hits = hits("reasonings")
    explanation_hits = hits("explanations")
    length = {}
    for m in ds.models:
        t = [len(f.decisions[m].reasonings[0]) for f in usable_items
             if f.decisions.get(m) and f.decisions[m].reasonings]
        length[m] = sum(t) / len(t) if t else float("nan")
    na100 = {m: reasoning_hits[m] / length[m] * 100 for m in ds.models}
    position = {column: {m: i + 1 for i, m in enumerate(sorted(ds.models, key=lambda m: -d[m]))}
           for column, d in (("przyjeta", reasoning_hits), ("na100", na100))}
    print(f"{'model':22}{'adopted':>10}{'per 100 ch.':>13}{'explanation':>13}{'characters':>12}")
    for m in sorted(ds.models, key=lambda m: -reasoning_hits[m]):
        print(f"{m:22}{reasoning_hits[m]:>10.3f}{na100[m]:>12.4f}{explanation_hits[m]:>13.3f}{length[m]:>12.0f}")
    print("\n  rank (adopted -> per 100 characters):")
    for m in sorted(ds.models, key=lambda m: position["przyjeta"][m]):
        print(f"    {m:22} {position['przyjeta'][m]} -> {position['na100'][m]}")

    output = Path(find_run_dir(args.run_dir)) / "analysis" / "grounding_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "run": Path(find_run_dir(args.run_dir)).name,
        "n_uzytecznych": len(usable_items),
        "czy_miara_porzadkuje_modele": {
            m: {
                "nazw_na_tekst": reasoning_hits[m],
                "nazw_na_100_znakow": na100[m],
                "nazw_w_wyjasnieniu_dla_programisty": explanation_hits[m],
                "srednia_dlugosc_znakow": length[m],
            } for m in ds.models
        },
        "korelacja_rang_przyjeta_wobec_na100": spearman(
            [reasoning_hits[m] for m in ds.models], [na100[m] for m in ds.models]),
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"\nWritten: {output}")

    print("\n== MESSAGE ALONE VERSUS MESSAGE WITH DESCRIPTION ==")
    print("The H4 baseline is message+description. The split shows that the SpotBugs")
    print("advantage sits in the description field, not in the message alone.")
    print(f"{'tool':<14}{'message only':>14}{'message+descr.':>16}{'n':>6}")
    for tool in sorted({f.tool for f in usable_items}):
        pod = [f for f in usable_items if f.tool == tool]
        sam = sum(bool(home(f.message or "", f)) for f in pod) / len(pod)
        both = message_share(pod, home)
        print(f"{tool:<14}{sam:>13.3f}{both:>14.3f}{len(pod):>6}")

    print("\n== A SINGLE REPETITION VERSUS THREE JOINED ==")
    print("The measure is computed from one repetition, because the tool text is also one. The")
    print("'three joined' row shows how much of the edge comes from text length alone.")
    message_text = message_share(usable_items, home)
    for label, pick in (("first repetition", lambda d: d.reasonings[:1]),
                              ("three joined", lambda d: d.reasonings)):
        value_of = []
        for m in ds.models:
            t = c = 0
            for f in usable_items:
                d = f.decisions.get(m)
                if d is None or not d.reasonings:
                    continue
                c += 1
                t += bool(home(" ".join(pick(d)), f))
            if c:
                value_of.append(t / c)
        print(f"  {label:<20} models {min(value_of):.3f}-{max(value_of):.3f}"
              f"   komunikat {message_text:.3f}   przepasc srednia {sum(value_of)/len(value_of) - message_text:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
