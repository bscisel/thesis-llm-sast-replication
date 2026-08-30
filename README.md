# Materiały do pracy magisterskiej

Dane, skrypty i tabele stojące za pracą *Analiza porównawcza skuteczności wybranych modeli
językowych w postprocessingu raportów z narzędzi do statycznej analizy kodu*.

## Układ

| katalog | zawartość |
|---|---|
| `results/` | sześć przebiegów: dane wejściowe i wyliczone wyniki w `analysis/` |
| `tables/` | 27 plików CSV — po jednym na tabelę z pracy, plus `extra-*` na liczby z prozy |
| `scripts/` | skrypty liczące, dobór próby, konwertery raportów, uruchamianie modeli |

Wszystko poza `analysis/` jest wejściem, wszystko w `analysis/` jest wyliczone.

## Przebiegi

| katalog | co to | ostrzeżeń |
|---|---|---|
| `002_jetty-population` | populacja ostrzeżeń przed próbkowaniem | — |
| `003_jetty-stage-1` | Jetty, pierwszy etap doboru | 300 |
| `004_error-benchmark` | baza przykładowych błędów | 43 |
| `005_jetty-stage-2` | Jetty, drugi etap | 60 |
| `006_jetty-combined-sample` | **próba łączona, główne wyniki** | 360 |
| `007_jetty-no-code` | ten sam zbiór bez pokazanego kodu | 360 |

## Przeliczenie od zera

```bash
pip install -r requirements.txt
export SOURCE_ROOT=/katalog/z/projektami     # zawiera jetty.project i java-errors-benchmark
python3 scripts/reproduce.py
```

Szesnaście kroków, około dziewięciu minut — tyle kosztuje 10 000 losowań bootstrapu
i 20 000 permutacji dla czterech przebiegów. Wyniki są w repozytorium policzone,
więc uruchamianie tego nie jest potrzebne.

Bez `SOURCE_ROOT` pomijana jest miara osadzenia uzasadnień w kodzie (H4 i tabela 7.5),
a rodzina Holma zawęża się z 22 do 15 wartości — plik wyników odnotowuje to
w `holm_family.complete` i `holm_family.missing`.

## Czego tu nie ma

Kodu Jetty i bazy błędów (są publiczne), pełnych raportów przebiegu 002 (60 MB),
kluczy API. Uruchomienie modeli wymaga `.env` na wzór `.env.example`.
