# Materiały do pracy magisterskiej

Dane, skrypty i tabele stojące za pracą *Analiza porównawcza skuteczności wybranych modeli
językowych w postprocessingu raportów z narzędzi do statycznej analizy kodu*.

## Katalogi

| katalog | zawartość |
|---|---|
| `results/` | przebiegi badania |
| `tables/` | tabele z pracy w formacie CSV |
| `scripts/analysis/` | wczytywanie danych, metryki, testy statystyczne, bootstrap, miara odwołań do kodu |
| `scripts/static_analysis/` | pobranie i konwersja raportów SonarQube, SpotBugs i Error Prone do wspólnego formatu |
| `scripts/pipeline/` | dobór próby |
| `scripts/llm/` | prompty i uruchamianie modeli językowych |
| `scripts/thesis/` | obliczenie hipotez, metryk i tabel |
| `scripts/reproduce.py` | przeliczenie wszystkiego jednym poleceniem |

## Przebiegi

| katalog | co to | ostrzeżeń |
|---|---|---|
| `002_jetty-population` | populacja ostrzeżeń przed próbkowaniem | — |
| `003_jetty-stage-1` | Jetty, pierwszy etap doboru | 300 |
| `004_error-benchmark` | baza przykładowych błędów | 43 |
| `005_jetty-stage-2` | Jetty, drugi etap doboru | 60 |
| `006_jetty-combined-sample` | próba łączona, główne wyniki pracy | 360 |
| `007_jetty-no-code` | ten sam zbiór bez pokazanego kodu | 360 |

Pliki w katalogu przebiegu:

| plik | zawartość |
|---|---|
| `static_analysis/` | ostrzeżenia narzędzi: raporty surowe i sprowadzone do wspólnego formatu |
| `llm/` | odpowiedzi modeli, po katalogu na model |
| `ground_truth.json` | etykiety odniesienia ostrzeżeń |
| `category_ground_truth.json` | kategorie odniesienia reguł |
| `sample_info.json` | sposób doboru próby |
| `run_info.json` | opis przebiegu |
| `analysis/` | wyniki wyliczone przez `scripts/reproduce.py` |

## Przeliczenie

```bash
pip install -r requirements.txt
export SOURCE_ROOT=/katalog/z/projektami     # zawiera jetty.project i java-errors-benchmark
python3 scripts/reproduce.py
```
