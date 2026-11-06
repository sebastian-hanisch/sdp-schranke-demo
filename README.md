# SDP-Schranke fürs Zuordnungsproblem – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-sdp-schranke-demo.streamlit.app/)** -->

Stück 7 von 15 (gelb, optional) der **Konzepte-Reihe „Anordnen: relaxieren und runden“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Konzepte-Seite der Reihe gibt es noch nicht). Die Schranken der Leiter stehen in den Stücken [3, Gilmore–Lawler](https://github.com/sebastian-hanisch/gilmore-lawler-demo), [4, Eigenwert und Projektion](https://github.com/sebastian-hanisch/eigenwert-schranke-demo) und [5, Linearisierung und Dual Ascent](https://github.com/sebastian-hanisch/linearisierung-schranke-demo), der Baum, der sie einsetzt, in [8, Branch and Bound](https://github.com/sebastian-hanisch/schranke-im-baum-demo); Stück [15, Verfahrenswahl](https://github.com/sebastian-hanisch/anordnen-verfahrenswahl-demo) stellt die Verfahren der Reihe gegeneinander. Die PSD-Projektion ist dieselbe Rechnung wie in der Konvexen Linie ([Max-Cut als SDP](https://github.com/sebastian-hanisch/kegel-maxcut-demo)).

## Warum dieses Problem

Eine untere Schranke der Kosten eines quadratischen Zuordnungsproblems belegt, wie gut eine gefundene Anordnung höchstens noch werden kann. Die Leiter aus Stück 3 bis 5 wird stärker und teurer: Gilmore–Lawler (eine lineare Zuordnung), die projizierte Eigenwertschranke, Dual Ascent, die Linearisierung (ein LP mit `n⁴` Variablen). Die **semidefinite Relaxation** hebt die Anordnung zu einer Matrix `Y` der Größe `n² + 1` an (der Lift), lässt den Rang eins fallen und behält `Y ⪰ 0` samt den linearen Eigenschaften einer Anordnung. Die Frage dieses Stücks: **Wie weit kommt sie, und wie weit trägt ein ADMM sie?**

Die Antwort dieser Messung ist **zur Hälfte bestätigt und zur Hälfte korrigiert**: Mit **Nichtnegativität** (`Y ≥ 0`, Stufe R3) ist die SDP im Median die **stärkste** Schranke der Leiter (in 2 von 30 Instanzen bei `n` = 6 liegt die Linearisierung darüber): **0,00 / 0,00 / 0,28 %** vom Optimum bei `n` = 8 / 9 / 12 (Gilmore–Lawler 8,3 / 10,2 / 13,3 %, Dual Ascent 5,2 / 8,0 / 11,3 %, Linearisierung 1,5 / 1,8 %). Die übliche Stufe **ohne** sie (R2: Pfeil und Gangster) ist es **nicht**: im Median 3,1 / 5,4 / 8,1 %, in 10 von 30, 7 von 18 und 8 von 18 Instanzen sogar **unter der Gilmore–Lawler-Schranke**. Aus der Dualmatrix des ADMM entsteht eine **sichere** untere Schranke, die bei jeder Toleranz gilt; das **Runden** der Lösung dagegen ist schwächer als 200 Abstiege und wird mit `n` schlechter (3 von 18 Treffern bei `n` = 12).

## Modell

- **Aufgabe** (wie in Stück 1 bis 8): `n` Bauteile auf `n` Plätze eines Rasters (Manhattan-Abstand), Fluss `F`; `cost(π) = Σ_{i<k} F[i,k] · D[π_i, π_k] = ½ xᵀ (D ⊗ F) x` mit `x = vec(X)` der Permutationsmatrix (Spaltenstapelung: Index `1 + j·n + i` für Bauteil `i` auf Platz `j`). Sechs erfundene Familien der Flussmatrix (Netzliste, gleichverteilt, dünn besetzt, zwei Cluster, Stern, wenige starke Flüsse); Raster 2×2 bis 3×4 in der App.
- **Lift und Bedingungen** (`sdp_relax.py`, eigene Herleitung nach der Art von Zhao, Karisch, Rendl und Wolkowicz, **nicht eingesehen**): `Y = (1, x)(1, x)ᵀ` der Größe `N = n² + 1`, `cost = <L, Y>`. Relaxiert: `Y ⪰ 0`, `Y00 = 1`, **Pfeil** `Y_{0a} = Y_{aa}`, **Gangster** `Y_{ab} = 0` für Paare `a = (i,j)`, `b = (k,l)`, die sich in genau einer Koordinate gleichen (`n²(n−1)` Paare). **Facial Reduction** `Y = V̂ R V̂ᵀ` erzwingt `Xe = e`, `Xᵀe = e` und senkt die Matrixgröße auf `r = (n−1)² + 1` (`n` = 8: 50, 12: 122, 16: 226). **R2** = diese Bedingungen, **R3** = zusätzlich `Y ≥ 0` (doppelt nichtnegativ).
- **ADMM** (eigene Fassung im Raum von `Y`): `Y⁺ = P_A(Z − U − L/ρ)`, `Z⁺ = P_K(Y⁺ + U)`, `U⁺ = U + Y⁺ − Z⁺`. `P_A` (die Gleichungen, bei R3 mit Klemmen `Y ≥ 0`) ist **einträgeweise**, `P_K` eine Eigenzerlegung der `r × r`-Matrix; `ρ` wird alle 100 Schritte verdoppelt oder halbiert, wenn ein Residuum zehnmal größer ist als das andere. Abbruch bei Primal- und Dualresiduum unter der Toleranz (Standard 1e-5; Schrittgrenze 6 000 bei R2, 15 000 bei R3). Das ist **nicht** der ADMM der Konvexen Linie (`kad_admm.py` invertiert `AᵀA` über alle Variablen: bei `n = 12` rund 7 400 Variablen und 1 700 Gleichungen, dicht nicht tragbar); die PSD-Projektion ist dieselbe Rechnung.
- **Sichere untere Schranke** aus der Dualmatrix `M = −ρU ∈ K*`: Multiplikatoren aus den Pfeil- und `Y00`-Einträgen von `L − M` (kleinste Quadrate), `T = L − 𝒜ᵀ(a, y)` (bei R3 `− N`, `N = max(0, L − 𝒜ᵀy − M) ≥ 0`), Schranke `a + (n+1) · min(0, λ_min(ṼᵀTṼ))`. Sie gilt für **jede** Dualmatrix, weil jedes zulässige `Y` die Spur `n + 1` hat; der **Wert** des ADMM dagegen ist bei grober Toleranz keine Schranke.
- **Lücke** = (Vergleichswert − Schranke) / Vergleichswert; der Vergleichswert ist bis `n = 9` das aufgezählte Optimum, bei `n = 12` das vom Baum (Stück 8, Gilmore–Lawler-Knotenschranken, Anfangswert die beste von 200 Abstiegen) bewiesene; bei einer einzelnen Instanz in der App mit `n = 12` die beste von 200 Abstiegen (die Lücke ist dann eine **obere** Grenze der wahren Lücke).
- **Rundung:** Zeile 0 von `Y` (die relaxierte Zuordnung `X`) durch lineare Zuordnung (Ungarische Methode, größte Summe).
- **Instanzen:** feste Seeds ab 100000: `n` = 6 / 8 mit 30, `n` = 9 / 12 mit 18 Instanzen (je Familie 5 / 5 / 3 / 3), die Instanzen der Beweisseite von Stück 15.
- **Experimente:** (1) die Relaxation (Lift, Zuordnung, Handbeispiel); (2) die Leiter; (3) der ADMM (`ρ`, Toleranz, sichere Schranke); (4) das Runden; (5) die Größe (bis `n = 16`). Formeln im 📐-Expander der App.

## Methodik

Der Abschnitt „Ergebnis“ und Experiment 1 rechnen **eine** Instanz (Familie, `n` = 4, 6, 8, 9, 12, Seed, Stufe, Toleranz, `ρ` und seine Anpassung sind Regler; `n = 12` nur auf Knopfdruck). Die Experimente 2 bis 5 laufen auf Knopfdruck über feste Instanzen und brauchen einige Minuten (auf Streamlit Cloud länger). Standard: Zwei Cluster, `n = 8`, Seed 48, R3, Toleranz 1e-5, `ρ = 1` mit Anpassung. Zeiten sind **Sekunden dieser numpy-Fassung auf dem jeweiligen Rechner** und schwanken von Lauf zu Lauf um das Doppelte; **Zeiten stehen in Anzeigen, nie in Tests.**

## Befunde (gemessen, keine Behauptungen)

Alles Messung an den sechs erfundenen Familien; Zeiten sind die dieser Fassung auf dem Rechner des Autors (Windows).

1. **Mit `Y ≥ 0` im Median die stärkste Schranke der Leiter.** Lücke zum Optimum im Median (R3, sichere Schranke): **0,00 / 0,00 / 0,00 / 0,28 %** bei `n` = 6 / 8 / 9 / 12, im Mittel 0,63 / 0,06 / 0,89 / 0,70 %, größter Fall 3,7 / 0,4 / 4,2 / 3,5 %; scharf (unter 0,05 %) in 22 von 30, 25 von 30, 12 von 18 und 7 von 18 Instanzen. R3 liegt **nie** unter Gilmore–Lawler, Dual Ascent oder der Eigenwertschranke, bei `n = 6` in 2 Instanzen unter der Linearisierung, bei `n = 8` und 9 nie; über der besten der vier anderen in 21 / 20 / 16 / 18 Instanzen. Schwach bleibt die Familie „zwei Cluster“ bei `n = 9` (Median 3,9 %) und bei `n = 6` (2,4 %).
2. **Ohne `Y ≥ 0` (R2) nicht.** Median **3,2 / 3,1 / 5,4 / 8,1 %**; in **8 von 30, 10 von 30, 7 von 18 und 8 von 18** Instanzen unter der Gilmore–Lawler-Schranke, bei `n` = 8 und 9 in 22 von 30 und 15 von 18 unter der Linearisierung. Schlecht auf dünnen und schiefen Familien (Median bei `n = 8` 15,3 % und 20,1 %, bei `n = 12` 15,4 % und 23,8 %), gut auf „zwei Cluster“ bei `n` = 8 und 12 (0,2 und 0,9 %), nicht bei `n` = 6 und 9 (3,5 und 5,4 %). Die Schärfe kommt von `Y ≥ 0`, nicht vom Lift allein.
3. **Die sichere Schranke hält.** Sie lag in 192 Läufen (96 Instanzen, beide Stufen, Phase 0 und Demo) und in allen Läufen der Empfindlichkeitsreihe nie über dem Optimum. Bei Toleranz 1e-2 liegt die sichere Schranke von R2 höchstens 2,8 unter dem Wert eines Laufs mit 1e-8 (Kosten um 200), bei 1e-4 höchstens 0,06, bei 1e-5 höchstens 0,006; der **Wert** des ADMM liegt bei R3 und 1e-2 bis zu 2,4 **über** dem Optimum.
4. **Zwei Algorithmen, dieselbe Zahl.** Ein dichter Innere-Punkte-Löser (Barrierenmethode, Gleichungen durch Nullraum eliminiert; nur im Test, `tests/ipm.py`) und der ADMM treffen denselben Wert auf vier Stellen (`n` = 4 und 6, beide Stufen; `n = 8` in der Vormessung). Der Innere-Punkte-Löser braucht bei `n = 6` rund 2,6 s und bei `n = 8` rund 9 bis 10 s gegen 0,1 bis 0,3 s und 1,3 bis 6 s des ADMM. **Über `n = 8` hinaus ist der Vergleich nicht gemessen** (mein Prototyp lieferte bei `n = 10` einen falschen Wert, Ursache nicht untersucht); eine Schätzung nach der Größe der Newton-Systeme (rund 5 800 freie Variablen bei `n = 12`) gibt eine halbe Stunde je Instanz.
5. **Die Rundung der Lösung ist schwach.** Sie trifft das Optimum bei R3 in 15 von 30, 19 von 30, 6 von 18 und 3 von 18 Instanzen (Lücke im Median 0,6 / 0,0 / 1,8 / 15,7 %), bei R2 in 11, 6, 1 und 0; 200 Abstiege aus Zufall treffen in 30, 30, 18 und 17 Fällen. Die Schranke ist stark, das Runden nicht: wie in den Stücken 9 bis 11 liegt der Nutzen der Relaxation bei der Schranke.
6. **Kosten des ADMM.** R3, Median der Schritte 358 / 576 / 864 / 2 290 und der Zeit 0,1 / 1,3 / 2,7 / 19 s (größter Fall 1,2 / 32 / 52 / 129 s, unter vier parallelen Prozessen gerechnet) bei `n` = 6 / 8 / 9 / 12; **nicht konvergiert** (Schrittgrenze) in 0 / 1 / 4 / 1 Instanzen (R2: 0 / 2 / 0 / 2). **`ρ` ist die Stellschraube:** bei `n = 8` braucht R2 je nach Start-`ρ` von 0,1 bis 10 zwischen rund 430 und 20 000 Schritten (angepasst: 430 bis 5 250), R3 zwischen rund 420 und 5 600 (angepasst: 420 bis 1 000); die Anpassung hilft fast überall.
7. **Die Größe** (eine Instanz, „zwei Cluster“, Seed 100000, Zeit je Schritt in der Eigenzerlegung): Zeit je Schritt (R2) 0,15 / 1,4 / 3,6 / 4,6 / 9,0 / 17 ms bei `n` = 6 / 8 / 10 / 12 / 14 / 16 (R3 ähnlich: 0,15 / 1,9 / 3,2 / 4,8 / 8,4 / 20 ms), die PSD-Matrix wächst von 26×26 auf 226×226. **R2 konvergiert in allen sechs Größen** (1 134 bis 3 684 Schritte, 0,2 s bis 31 s). **R3 konvergiert bei `n` = 6, 8, 12** (252 bis 2 226 Schritte), **bei `n` = 10, 14, 16 nicht in 8 000 Schritten** (25 / 68 / 163 s); dort liegt die sichere Schranke bei 321,9 / 863,9 / 1 169,8 gegen die beste von 200 Abstiegen 324 / 866 / 1 188 (bei `n = 16` höchstens 1,5 % unter einer oberen Grenze des Optimums) und Gilmore–Lawler 281 / 758,5 / 1 018,5. Die Matrixgröße trägt also bis `n = 16`; die Schärfe von R3 bei `n ≥ 10` ist an einer Instanz mit Schrittgrenze nicht gesichert.

**Handbeispiel** (Stück 3, 5, 8): vier Plätze auf einer Zeile, Gilmore–Lawler-Wurzel 5, Optimum 6; beide Stufen liefern **6,000**, Dual Ascent 5,94, Linearisierung 6. Von Hand prüfbar: `N = 17`, `r = 10`, 48 Gangster-Paare, Spur `n + 1 = 5`, und der Permutationspunkt `Y* = (1, x*)(1, x*)ᵀ` des Optimums liegt in der Relaxation mit Kosten 6.

### Lücke zum Optimum in Prozent (Median; sichere Schranke der SDP)

| Schranke | n = 6 | 8 | 9 | 12 |
|---|---|---|---|---|
| Gilmore–Lawler (Stück 3) | 5,4 | 8,3 | 10,2 | 13,3 |
| projizierte Eigenwertschranke (Stück 4) | 15,3 | 12,6 | 19,7 | 27,5 |
| Dual Ascent, 5 Runden (Stück 5) | 3,9 | 5,2 | 8,0 | 11,3 |
| Linearisierung (Stück 5) | 2,5 | 1,5 | 1,8 | – |
| **SDP R2** | 3,2 | 3,1 | 5,4 | 8,1 |
| **SDP R3** | **0,00** | **0,00** | **0,00** | **0,28** |
| R2 unter Gilmore–Lawler | 8 von 30 | 10 von 30 | 7 von 18 | 8 von 18 |
| R3 über der besten der anderen | 21 von 30 | 20 von 30 | 16 von 18 | 18 von 18 |

Instanzen: 30 / 30 / 18 / 18. Die Werte der Lücken, Zählungen und Treffer stehen mit Toleranz in `tests/test_claims.py`; die Zeiten stehen in keinem Test.

## Ehrliche Grenzen

- **Eigene Herleitung:** Lift, Pfeil, Gangster, Facial Reduction, R2 und R3, ADMM und sichere Schranke sind eigene Fassungen; Zhao, Karisch, Rendl, Wolkowicz (Relaxation), Povh und Rendl (Nichtnegativität) und Wen, Goldfarb, Yin (ADMM für SDP) sind **nicht nachgelesen**. Die Namen R2 und R3 sind Bezeichnungen dieser Demo. Selbstangaben der Literatur zur Güte werden nicht übernommen.
- **Nur kleine Größen mit bekanntem Optimum:** `n` = 6 bis 12; bei `n = 16` nur eine Instanz ohne Optimum. Instanzen aus QAPLIB sind nicht untersucht; ob R3 bei `n ≥ 20` so scharf bleibt, ist offen.
- **Der ADMM konvergiert nicht immer:** R3 erreicht schon bei `n` = 8 und 9 in einzelnen Instanzen die Schrittgrenze von 15 000, in der Größenreihe bei `n` = 10, 14 und 16 die Grenze von 8 000 Schritten; an der Schrittgrenze ist der Wert keine Schranke, die sichere Schranke bleibt gültig (aber schwächer). Die App sagt es.
- **Keine weiteren Schnitte:** Dreiecksungleichungen und andere Schnitte sind nicht gebaut; die Schärfe von R3 gilt für die gemessenen Instanzen und Familien.
- **Rundung nur durch lineare Zuordnung;** bessere Rundungen (mehrere Anordnungen ziehen, dann Tausch) sind nicht gebaut.
- **Schranke im Baum nicht gemessen:** ob die SDP am Knoten die Knotenzahl senkt, ist offen; sie kostet Sekunden je Knoten.
- **Innere-Punkte-Vergleich nur bis `n = 8`** (siehe Befund 4).
- **Familien erfunden:** „zwei Cluster“ verhält sich bei `n = 9` anders als bei `n = 8` und 12; jede Aussage nennt die Familien.
- **Zeit ist implementierungsabhängig:** numpy, Eigenzerlegung; die Zeit je Instanz hängt am Rechner und an der Zahl paralleler Prozesse. In Tests stehen nie Zeiten.
- **Rechenabhängigkeit:** Eigenzerlegungen und die Schrittgrenze machen die Ergebnisse von Instanzen nahe der Grenze rechnerabhängig; Mediane und Zählungen haben deshalb Toleranzen in `tests/test_claims.py` (wie in Stück 11 und 15). Die sichere Schranke bleibt auf jedem Rechner unter dem Optimum.

## Verifikation

- **Tests** (`pytest tests/`): **285 Tests**, rund achtzehn Minuten bei einem früheren Windows-Lauf (Python 3.14), im Linux-CI dagegen 3 h 37 min ( die festen Instanzen der Zahlen-Tests rechnen parallel, allein `n = 12` etwa sieben Minuten). Handwerte (`N = 17`, `r = 10`, 48 Gangster-Paare, Spur 5, Permutationspunkt mit Kosten 6 am Handbeispiel; alle 24 Anordnungen zulässig), Eigenschaften (`Ṽ` orthonormal, `P_K` und `P_A` idempotent und nächstgelegen, R3 ≥ R2, die Zeilensummen von `X` gleich `Y00`), Orakel (Aufzählung aller Anordnungen für das Optimum; **schwache Dualität** für jede zufällige Dualmatrix und bei jeder Toleranz; ein **zweiter Algorithmus**, der dichte Innere-Punkte-Löser, trifft den ADMM-Wert; unabhängig geschriebene Kernkopien aus Stück 1 bis 8 mit ihren Tests), Verdikte (Art und Text folgen den Zahlen, jeder Zweig an festen Eingaben), AppTest (alle Regler an ihren Grenzen, alle Rechenknöpfe mit verkleinerten Reihen, PDF), und die **Zahlen der README** an den festen Instanzen (`tests/test_claims.py`, parallel gerechnet, mit Toleranzen für die Streuung zwischen Rechnern und Sammelmeldung statt Abbruch beim ersten Fehler).
- **Kopie:** die Kerne (Instanzen, Zuordnung, Aufzählung, Gilmore–Lawler, Dual Ascent, Linearisierung, Eigenwertschranke, Baum) sind Kopien aus Stück 1 bis 8; Tests vergleichen die Familien mit denen von Stück 1 bis 12, sofern die Schwester-Repos nebenan liegen.
- **Nicht getestet:** Zeiten; kein Mutationstest; Linux-CI: grün (Lauf 42175981352 auf main, Laufzeit 3 h 37 min wegen der rechenintensiven Zahlen-Tests; die Toleranzen sind von Anfang an auf die Streuung zwischen Rechnern ausgelegt); die Zahlen der Tabellen stammen aus dem Windows-Lauf.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, fünf Experimente, Grenzen, Formulierung |
| `sdp_relax.py` | Die Relaxation (Lift, Facial Reduction, Projektionen), der ADMM, die sichere Schranke, die Rundung |
| `sdp_evaluation.py` | Eine Instanz, Zeilen der Vergleiche, Lücke, Zählungen, Verdikte |
| `sdp_rng.py`, `sdp_cost.py`, `sdp_landscape.py` | SplitMix64, Kosten, Aufzählung (Kopien aus Stück 1 und 2) |
| `sdp_instance.py` | Sechs Familien der Flussmatrix, Raster, Handbeispiele |
| `sdp_assignment.py` | Ungarische Methode samt Dualwerten (Kopie aus Stück 8) |
| `sdp_bounds.py` | Gilmore–Lawler, Dual Ascent, Abstiege aus Zufall (Kopien aus Stück 3 und 5) |
| `sdp_spectral.py` | Projizierte Eigenwertschranke (Kopie aus Stück 4) |
| `sdp_lp.py` | Linearisierung als LP (Kopie aus Stück 5, scipy) |
| `sdp_tree.py` | Branch and Bound (Kopie aus Stück 8; belegt das Optimum bei `n = 12`) |
| `sdp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `sdp_pdf_export.py` | PDF-Export des Ergebnisses |
| `sdp_constants.py` | Größen, Familien, Stufen, ADMM-Einstellungen, feste Reihen, Farben, Presets |
| `sdp_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Button |
| `sdp_layout.py` | Fester Kopf (Stylesheet) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` für die Linearisierung) |
| `tests/` | Relaxation, Orakel, Auswertung, Verdikte, Kerne, Abbildungen, Presets, Layout, AppTest, Zahlen der README (`test_claims.py`, `rows.py`), Innere-Punkte-Orakel (`ipm.py`), Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Weitere Schnitte** (Dreiecksungleichungen, Teilmengen-Schnitte) und andere Relaxationen (Kopositiv, Burer–Monteiro).
- **Die SDP im Baum** (Knotenschranke) und **Instanzen aus QAPLIB**.
- **Ein Innere-Punkte-Löser in der App** (nur Orakel im Test, `n ≤ 6`).
- **Rundungsverfahren** über die lineare Zuordnung hinaus.
- **Zeiten in Tests.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
