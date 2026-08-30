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

def tekst_modelu(decyzja):
    return decyzja.reasonings[0] if decyzja and decyzja.reasonings else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=int, default=6)
    args = ap.parse_args()

    from analysis.dataset import load_dataset
    from analysis.grounding import (_build_code_context, _strip_line_prefixes, _strip_comments,
                                    _as_raw, build_grounding, score_text,
                                    IDENTIFIER, _looks_like_identifier)

    ds = load_dataset(find_run_dir(args.run_dir))
    gr = {f.key: build_grounding(f, ds.source_root) for f in ds.findings}
    uzyteczne = [f for f in ds.findings if gr[f.key].context_available]

    def udzial_modeli(dataset, punktuj):
        wyn = {}
        for m in ds.models:
            t = c = 0
            for f in dataset:
                d = f.decisions.get(m)
                if d is None or not d.reasonings:
                    continue
                c += 1
                t += bool(punktuj(tekst_modelu(d), f))
            wyn[m] = t / c if c else float("nan")
        return wyn

    def udzial_komunikatu(dataset, punktuj):
        return sum(bool(punktuj(f"{f.message} {f.description}", f)) for f in dataset) / len(dataset)

    dom = lambda txt, f: score_text(txt, gr[f.key])["grounded"]

    print("== ILE NAZW Z KODU NARZĘDZIE PODAJE SAMO ==")
    print("Tekst narzędzia trafia do polecenia, więc te nazwy model dostaje bez czytania kodu.")
    print(f"{'narzędzie':<14}{'nazw na ostrzeżenie':>22}{'n':>6}")
    for tool in sorted({f.tool for f in uzyteczne}):
        sub = [f for f in uzyteczne if f.tool == tool]
        srednia = sum(len(gr[f.key].message_identifiers) for f in sub) / len(sub)
        print(f"{tool:<14}{srednia:>22.2f}{len(sub):>6}")
    print()
    print("== H4 W ROZBICIU NA NARZĘDZIE ==")
    print("Agregat zaciera różnicę: przy SpotBugs punktu odniesienia nie przewyższa żaden model.")
    print(f"{'narzędzie':<14}{'komunikat':>10}{'modele min':>12}{'modele maks':>13}{'n':>6}")
    for tool in sorted({f.tool for f in uzyteczne}):
        sub = [f for f in uzyteczne if f.tool == tool]
        u = udzial_modeli(sub, dom)
        print(f"{tool:<14}{udzial_komunikatu(sub, dom):>10.3f}"
              f"{min(u.values()):>12.3f}{max(u.values()):>13.3f}{len(sub):>6}")

    def tokens(txt, minlen, filtruj):
        return {t for t in IDENTIFIER.findall(txt or "")
                if len(t) >= minlen and (not filtruj or _looks_like_identifier(t))}

    window = {}
    for f in uzyteczne:
        ctx = _build_code_context(f.raw or _as_raw(f), ds.source_root)
        surowy = "\n".join(k for _, k in _strip_line_prefixes(ctx))
        window[f.key] = (surowy, _strip_comments(surowy))

    def przepasc(usun_komentarze, filtruj, minlen=4):
        mod = kom = lm = lk = 0
        for f in uzyteczne:
            code = window[f.key][1 if usun_komentarze else 0]
            kt = tokens(code, minlen, filtruj)
            kom += bool(tokens(f"{f.message} {f.description}", minlen, filtruj) & kt)
            lk += 1
            for m in ds.models:
                d = f.decisions.get(m)
                if d and d.reasonings:
                    mod += bool(tokens(tekst_modelu(d), minlen, filtruj) & kt)
                    lm += 1
        return mod / lm, kom / lk

    print("\n== WARIANTY TOKENIZACJI (próg 4 znaki) ==")
    print(f"{'wariant':<40}{'modele':>9}{'komunikat':>11}{'przepaść':>10}")
    for name, (uk, fl) in {
        "przyjęty (bez komentarzy, z filtrem)": (True, True),
        "bez komentarzy, bez filtru": (True, False),
        "z komentarzami, z filtrem": (False, True),
        "każde słowo (bez obu)": (False, False),
    }.items():
        a, b = przepasc(uk, fl)
        print(f"{name:<40}{a:>9.3f}{b:>11.3f}{a - b:>10.3f}")

    print("\n== WRAŻLIWOŚĆ NA PRÓG DŁUGOŚCI ==")
    print(f"{'próg':>6}{'modele':>9}{'komunikat':>11}{'przepaść':>10}")
    for minlen in (2, 3, 4, 5):
        a, b = przepasc(True, True, minlen)
        print(f"{minlen:>6}{a:>9.3f}{b:>11.3f}{a - b:>10.3f}")

    # Blok komentarza otwarty przed początkiem wycinka zostaje niewykryty — w wycinku widać samo domknięcie.
    truncated = 0
    for f in uzyteczne:
        code = window[f.key][0]
        truncated += code.count("*/") > code.count("/*")
    print(f"\n== WYCINKI Z BLOKIEM KOMENTARZA OTWARTYM PRZED POCZATKIEM ==")
    print(f"{truncated} z {len(uzyteczne)} — czyszczenie ich nie wykrywa")

    def rangi(x):
        s = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for position, i in enumerate(s):
            r[i] = position + 1
        return r

    def spearman(a, b):
        ra, rb = rangi(a), rangi(b)
        n = len(a)
        sa, sb = sum(ra) / n, sum(rb) / n
        counts = sum((x - sa) * (y - sb) for x, y in zip(ra, rb))
        mia = (sum((x - sa) ** 2 for x in ra) * sum((y - sb) ** 2 for y in rb)) ** 0.5
        return counts / mia if mia else float("nan")

    code_tokens, comment_tokens = {}, {}
    for f in uzyteczne:
        kt = tokens(window[f.key][1], 4, True)
        km = tokens(f"{f.message} {f.description}", 4, True)
        code_tokens[f.key] = kt
        comment_tokens[f.key] = km

    def hits(pole):
        wyn = {}
        for m in ds.models:
            sumy = []
            for f in uzyteczne:
                d = f.decisions.get(m)
                texts = getattr(d, pole, None) if d else None
                if not texts:
                    continue
                kt = code_tokens[f.key]
                km = comment_tokens[f.key]
                scored = tokens(texts[0], 4, True)
                sumy.append(len((scored & kt) - km))
            wyn[m] = sum(sumy) / len(sumy) if sumy else float("nan")
        return wyn

    print("\n== CZY MIARA PORZADKUJE MODELE ==")
    reasoning_hits = hits("reasonings")
    explanation_hits = hits("explanations")
    length = {}
    for m in ds.models:
        t = [len(f.decisions[m].reasonings[0]) for f in uzyteczne
             if f.decisions.get(m) and f.decisions[m].reasonings]
        length[m] = sum(t) / len(t) if t else float("nan")
    na100 = {m: reasoning_hits[m] / length[m] * 100 for m in ds.models}
    position = {column: {m: i + 1 for i, m in enumerate(sorted(ds.models, key=lambda m: -d[m]))}
           for column, d in (("przyjeta", reasoning_hits), ("na100", na100))}
    print(f"{'model':22}{'przyjęta':>10}{'na 100 zn.':>12}{'wyjaśnienie':>13}{'dł. znaków':>12}")
    for m in sorted(ds.models, key=lambda m: -reasoning_hits[m]):
        print(f"{m:22}{reasoning_hits[m]:>10.3f}{na100[m]:>12.4f}{explanation_hits[m]:>13.3f}{length[m]:>12.0f}")
    print("\n  pozycja w uporządkowaniu (przyjęta -> na 100 znaków):")
    for m in sorted(ds.models, key=lambda m: position["przyjeta"][m]):
        print(f"    {m:22} {position['przyjeta'][m]} -> {position['na100'][m]}")

    wyjscie = Path(find_run_dir(args.run_dir)) / "analysis" / "grounding_diagnostics.json"
    wyjscie.parent.mkdir(parents=True, exist_ok=True)
    wyjscie.write_text(json.dumps({
        "run": Path(find_run_dir(args.run_dir)).name,
        "n_uzytecznych": len(uzyteczne),
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
    print(f"\nZapisano: {wyjscie}")

    print("\n== SAM KOMUNIKAT WOBEC KOMUNIKATU Z OPISEM ==")
    print("Punktem odniesienia H4 jest message+description. Rozbicie pokazuje, ze przewaga")
    print("SpotBugs siedzi w polu description, nie w samym message.")
    print(f"{'narzedzie':<14}{'sam message':>13}{'message+opis':>14}{'n':>6}")
    for tool in sorted({f.tool for f in uzyteczne}):
        pod = [f for f in uzyteczne if f.tool == tool]
        sam = sum(bool(dom(f.message or "", f)) for f in pod) / len(pod)
        oba = udzial_komunikatu(pod, dom)
        print(f"{tool:<14}{sam:>13.3f}{oba:>14.3f}{len(pod):>6}")

    print("\n== POJEDYNCZY PRZEBIEG WOBEC SKLEJKI TRZECH ==")
    print("Wskaznik liczy sie z jednego przebiegu, bo tekst narzedzia tez jest jeden. Wiersz")
    print("'sklejka trzech' pokazuje, ile przewagi dokłada sama długość tekstu.")
    komunikat = udzial_komunikatu(uzyteczne, dom)
    for label, wybierz in (("pierwszy przebieg", lambda d: d.reasonings[:1]),
                              ("sklejka trzech", lambda d: d.reasonings)):
        wart = []
        for m in ds.models:
            t = c = 0
            for f in uzyteczne:
                d = f.decisions.get(m)
                if d is None or not d.reasonings:
                    continue
                c += 1
                t += bool(dom(" ".join(wybierz(d)), f))
            if c:
                wart.append(t / c)
        print(f"  {label:<20} modele {min(wart):.3f}-{max(wart):.3f}"
              f"   komunikat {komunikat:.3f}   przepasc srednia {sum(wart)/len(wart) - komunikat:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
