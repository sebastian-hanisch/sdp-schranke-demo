"""SDP-Schranke fürs Zuordnungsproblem - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 7 (gelb) der Reihe „Anordnen: relaxieren und runden“: wie weit kommt eine semidefinite Relaxation des quadratischen Zuordnungsproblems, und wie weit trägt ein ADMM sie?
Eigene Fassung (Lift, Pfeil, Gangster, Facial Reduction; R3 mit Y ≥ 0), ADMM mit einträgeweiser Projektion und eine **sichere** untere Schranke aus der Dualmatrix. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import sdp_constants as C
from sdp_evaluation import (
    Settings,
    admm_study,
    below,
    beats_all_others,
    convergence_stats,
    count,
    de,
    family_medians,
    gap,
    hand_data,
    instance_data,
    ladder_stats,
    ladder_sweep,
    num,
    pct,
    pooled,
    round_stats,
    safe_violations,
    scale_row,
    secs,
    verdict_admm,
    verdict_core,
    verdict_family,
    verdict_hand,
    verdict_ladder,
    verdict_relax,
    verdict_round,
    verdict_scale,
)
from sdp_layout import dunkles_design, kopf_css
from sdp_pdf_export import build_pdf
from sdp_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from sdp_visualization import build_family_bars, build_instance_bounds, build_ladder_bars, build_lift, build_residuals, build_rho_bars, build_round_bars, build_scale, build_tol_chart

st.set_page_config(page_title="SDP-Schranke fürs Zuordnungsproblem – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne die Relaxation …")
def _instance(settings: Settings):
    return instance_data(settings)


@st.cache_data(show_spinner=False)
def _hand():
    return hand_data()


@st.cache_data(show_spinner=False)
def _ladder(family, n):
    return ladder_sweep(family, n)


@st.cache_data(show_spinner=False)
def _study(variant):
    return admm_study(variant)


@st.cache_data(show_spinner=False)
def _scale(n, variant):
    return scale_row(n, variant)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


def run_knob(flag, label, info, todo, work):
    """Rechenknopf: `todo` ist eine Liste von Beschriftungen und Aufruf-Argumenten; nach dem Lauf steht `flag` im Session State."""
    if st.session_state.get(flag):
        return True
    st.info(info)
    if st.button(label, key=flag + "_button"):
        bar = st.progress(0.0, text="Rechne …")
        for i, (text, args) in enumerate(todo):
            work(*args)
            bar.progress((i + 1) / len(todo), text=f"{text} fertig")
        st.session_state[flag] = True
        st.rerun()
    return False


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sdp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔷 SDP-Schranke fürs Zuordnungsproblem")
st.markdown(
    """
**Wie weit kommt eine semidefinite Relaxation, und wie weit trägt ein ADMM sie?** Dieselbe Aufgabe wie in Stück 1 bis 8: `n` Bauteile auf `n` Plätze eines Rasters, Kosten gleich der Summe aus **Fluss mal Abstand**; gesucht ist eine **untere Schranke** der Kosten.
Die Anordnung wird zu einer Matrix `Y` der Größe `n² + 1` „hochgehoben“ (der Lift); man lässt den Rang eins fallen und verlangt `Y ⪰ 0` samt den linearen Eigenschaften einer Anordnung (**Pfeil** und **Gangster**: **R2**), bei **R3** zusätzlich `Y ≥ 0`.
Gelöst wird mit einem **ADMM**, und aus seiner Dualmatrix entsteht eine **sichere** untere Schranke, die bei jeder Toleranz gilt. Die Fassung ist **eigen** (die Quellen sind nicht eingesehen); die sechs Familien der Flussmatrix sind **eigene Erfindungen**.
Stück 7 der Reihe „Anordnen: relaxieren und runden“ (gelb, optional).

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Der Lift:** `Y = (1, x)(1, x)ᵀ` mit `x = vec(X)` für die Permutationsmatrix `X`; die Kosten sind linear in `Y`: `<L, Y>` mit `L = ½ (D ⊗ F)` (und einer Null-Zeile und -Spalte). Man ersetzt die Menge der `Y` zu Anordnungen durch eine größere, bei der nur noch Linearität und Semidefinitheit bleiben.
2. **Die Bedingungen:** `Y00 = 1`; **Pfeil** `Y0a = Yaa`; **Gangster**: die Einträge zu zwei Plätzen *eines* Bauteils oder zwei Bauteilen *eines* Platzes sind null. Die **Facial Reduction** `Y = V̂ R V̂ᵀ` erzwingt zusätzlich, dass jede Zeile und Spalte von `X` die Summe 1 hat; die PSD-Matrix hat dann nur `(n−1)² + 1` Zeilen.
3. **R2 gegen R3:** R3 verlangt außerdem `Y ≥ 0` (doppelt nichtnegativ). Das kostet nichts an Größe, ändert aber die Schärfe stark (Experiment 2).
4. **Der ADMM:** abwechselnd Projektion auf die Gleichungen (einträgeweise, bei R3 mit Klemmen), Projektion auf den PSD-Kegel (eine Eigenzerlegung der `(n−1)² + 1`-Matrix) und ein Dualschritt; `ρ` wird angepasst, wenn ein Residuum zehnmal größer ist als das andere.
5. **Die sichere Schranke:** aus der Dualmatrix wird eine Schranke gebaut, die für **jede** Dualmatrix gilt (jedes zulässige `Y` hat die Spur `n + 1`). Der **Wert** des ADMM dagegen ist bei grober Toleranz keine Schranke.
6. **Das Maß:** Lücke = (Vergleichswert − Schranke) / Vergleichswert; der Vergleichswert ist bis `n = 9` das aufgezählte Optimum, bei `n = 12` das vom Baum (Stück 8) bewiesene (feste Reihen) oder die beste von 200 Abstiegen (einzelne Instanz).
7. **Feste Instanzen:** Vergleiche laufen über feste Seeds ab 100000 (`n` = 6, 8, 9, 12 mit 30 / 30 / 18 / 18 Instanzen), nie über einen einzelnen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Instanz**")
    family = st.selectbox("Familie der Flussmatrix", C.FAMILIES, key="family_select", format_func=lambda f: C.FAMILY_NAMES[f], help="Wie der Fluss zwischen den Bauteilen verteilt ist. Die Beschreibung steht unter dem Regler.")
    st.caption(C.FAMILY_HELP[family])
    n = st.select_slider("Bauteile n", options=C.SIZES, key="n_select", help=f"Raster von 2×2 bis 3×4. Bis n = 9 wird alles aufgezählt (Vergleichswert: Optimum); n = {C.HEAVY_N} läuft nur auf Knopfdruck.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz", width="stretch", on_click=randomize_seed)
    st.markdown("**Die Relaxation**")
    variant = st.radio("Stufe", C.VARIANTS, key="variant_select", format_func=lambda k: C.VARIANT_NAMES[k], help="R2: Pfeil und Gangster. R3: zusätzlich Y ≥ 0.")
    st.caption(C.VARIANT_HELP[variant])
    st.markdown("**Der ADMM**")
    tol = st.select_slider("Toleranz", options=C.TOL_CHOICES, key="tol_select", format_func=C.tol_text, help="Abbruch, wenn Primal- und Dualresiduum höchstens so groß sind (Kosten auf den größten Eintrag 1 normiert).")
    rho = st.select_slider("Start-ρ", options=C.RHO_CHOICES, key="rho_select", format_func=lambda x: de(x, 1), help="Strafgewicht der Aufspaltung; entscheidet stark über die Zahl der Schritte.")
    adaptive = st.toggle("ρ anpassen", key="adaptive_toggle", help=f"Alle {C.ADAPT_EVERY} Schritte ρ verdoppeln oder halbieren, wenn ein Residuum zehnmal größer ist als das andere.")

sync_query_params({"family_select": family, "n_select": int(n), "seed_input": int(seed), "variant_select": variant, "tol_select": float(tol), "rho_select": float(rho), "adaptive_toggle": bool(adaptive)})
settings = Settings(family, int(n), int(seed), variant, float(tol), float(rho), bool(adaptive))

v = None
if settings.n < C.HEAVY_N or st.session_state.get("heavy_go") == settings:
    v = _instance(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if v is None:
    st.info(f"Für n = {settings.n} (Matrix {settings.n ** 2 + 1}×{settings.n ** 2 + 1}) braucht der ADMM im Median etwa zehn Sekunden, im ungünstigen Fall über eine Minute (auf Streamlit Cloud länger). Mit dem Knopf starten:")
    if st.button("⏱️ Relaxation rechnen", key="heavy_button"):
        st.session_state["heavy_go"] = settings
        st.rerun()
else:
    res = v["res"]
    ref_label = "Optimum" if v["exact"] else "Beste von 200 Abstiegen"
    m1, m2, m3 = st.columns(3)
    m4, m5 = st.columns(2)
    m1.metric("Sichere untere Schranke", de(res.bound, 2), delta=f"Lücke {pct(gap(res.bound, v['ref']), 2)}", delta_color="off", help="Gilt bei jeder Toleranz; Lücke gegen den Vergleichswert.")
    m2.metric(ref_label, num(v["ref"]), help="Die kleinsten Kosten aller n! Anordnungen (aufgezählt)." if v["exact"] else "Die beste von 200 Abstiegen aus Zufallsstarts: eine obere Schranke des Optimums.")
    m3.metric("Schritte des ADMM", count(res.iterations), delta="konvergiert" if res.converged else "NICHT konvergiert (Schrittgrenze)", delta_color="off", help=f"Schrittgrenze {count(C.MAX_ITER[settings.variant])} für {C.VARIANT_SHORT[settings.variant]}.")
    m4.metric("Wert des ADMM", de(res.value, 3), help="<L, Z> am Ende; bei grober Toleranz keine Schranke (kann über dem Optimum liegen).")
    m5.metric("Rechenzeit", secs(res.seconds), help="Sekunden dieser numpy-Fassung auf diesem Rechner; schwankt von Lauf zu Lauf.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_instance_bounds(v, settings.variant), width="stretch", key="instance_bounds")
    with c2:
        st.plotly_chart(build_residuals(res), width="stretch", key="residuals")
    if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
        st.session_state["pdf_made"] = (settings, build_pdf(settings, v))
    made = st.session_state.get("pdf_made")
    if made and made[0] == settings:
        st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="sdp-schranke.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie weit kommt die SDP, und gilt die Zahl?")
if v is not None:
    verdict(verdict_core(settings, v))

st.markdown("---")

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Die Relaxation")
st.caption("Links der Lift Y (Zeile und Spalte 0 gehören zu Y00 und x; die Gangster-Einträge sind null), rechts die relaxierte Zuordnung X (Zeile Bauteil, Spalte Platz) mit der Anordnung des Vergleichswerts (Kreuze). Darunter das Handbeispiel.")
if v is not None:
    st.plotly_chart(build_lift(v["rel"], v["res"].Z, v["ref_perm"], f"Lift und Zuordnung, {C.VARIANT_SHORT[settings.variant]}"), width="stretch", key="lift")
    verdict(verdict_relax(settings, v))
else:
    st.caption("Die Relaxation ist noch nicht gerechnet (Knopf oben).")
h = _hand()
verdict(verdict_hand(h))
with st.expander("Handbeispiel zum Nachrechnen"):
    st.markdown(
        f"**Vier Plätze auf einer Zeile** (Stück 3, 5, 8; Abstände 1 bis 3), Fluss A–C, A–D, B–C, B–D je 1. Die Gilmore–Lawler-Schranke der Wurzel ist **{num(h['glb'])}**, das Optimum **{h['opt']}**: A C B D (Plätze 1 bis 4) kostet 1 + 3 + 1 + 1 = 6. "
        f"Der Lift hat `N = 4² + 1 = 17`, die PSD-Matrix nach der Facial Reduction `(4−1)² + 1 = 10` Zeilen; Gangster-Paare gibt es `n²(n−1) = 48`. Der **Permutationspunkt** des Optimums, `Y* = (1, x*)(1, x*)ᵀ`, erfüllt alle Bedingungen und hat die Kosten **{num(h['point_cost'])}** und die Spur **{num(h['point_trace'])}** (`= n + 1`): "
        f"also kann keine Schranke der Relaxation über 6 liegen. Beide Stufen liefern hier **{de(h['r2']['res'].value, 3)}**: das Optimum, belegt durch die sichere Schranke."
    )

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Leiter")
st.caption("Die sichere Schranke von R2 und R3 neben den Schranken der Stücke 3 bis 5, über feste Instanzen (n = 6, 8, 9, 12; sechs Familien): Lücke zum Optimum im Median. Bei n = 12 ist das Optimum vom Baum aus Stück 8 bewiesen.")
ladder_ok = run_knob("ladder_ok", "⏱️ Die Leiter rechnen (n = 6, 8, 9, 12)", "Die Rechnung läuft über 96 Instanzen mit beiden Stufen, den anderen Schranken, 200 Abstiegen und (bei n = 12) dem Baum und dauert einige Minuten (auf Streamlit Cloud länger; bei n = 12 im Median etwa 20 Sekunden je Instanz). Mit dem Knopf starten:",
                     [(f"n = {m}, {C.FAMILY_NAMES[f]}", (f, m)) for m in C.LADDER_SIZES for f in C.FAMILIES], lambda f, m: _ladder(f, m))
rows_by_n = {m: pooled({f: _ladder(f, m) for f in C.FAMILIES}) for m in C.LADDER_SIZES} if ladder_ok else {}
if ladder_ok:
    st.plotly_chart(build_ladder_bars(rows_by_n, ladder_stats, "Lücke der sicheren Schranken zum Optimum, Median"), width="stretch", key="ladder_bars")
    tab = []
    for m in C.LADDER_SIZES:
        rs = rows_by_n[m]
        row = {"n": m, "Instanzen": len(rs)}
        for k in C.LADDER_KEYS:
            row[C.LADDER_NAMES[k]] = de(ladder_stats(rs, k)["median"], 2) + " %" if ladder_stats(rs, k)["count"] else "–"
        row["R2 unter Gilmore–Lawler"] = f"{below(rs, 'r2', 'glb')} von {len(rs)}"
        row["R3 über der besten anderen"] = f"{beats_all_others(rs, 'r3')} von {len(rs)}"
        row["R3 scharf"] = f"{ladder_stats(rs, 'r3')['sharp']} von {len(rs)}"
        tab.append(row)
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    verdict(verdict_ladder(rows_by_n))
    n2 = st.radio("Größe für die Familien", C.LADDER_SIZES, key="n2_select", horizontal=True, format_func=lambda m: f"n = {m}")
    st.plotly_chart(build_family_bars(rows_by_n[n2], family_medians, f"Lücke je Familie, n = {n2}"), width="stretch", key=f"family_bars_{n2}")
    verdict(verdict_family(rows_by_n[n2], n2))

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Der ADMM")
st.caption(f"n = {C.RHO_N}, drei Familien, die gewählte Stufe: Schritte je Start-ρ (fest gegen angepasst) und der Wert des ADMM gegen die sichere Schranke je Toleranz. Bezugswert ist das aufgezählte Optimum.")
if run_knob("study_ok", "⏱️ ρ und Toleranz rechnen", "Die Rechnung läuft über 3 Instanzen mit je einem Bezugslauf, zehn ρ-Läufen und sechs Toleranzläufen und dauert einige Minuten (auf Streamlit Cloud länger). Mit dem Knopf starten (gerechnet wird für die gewählte Stufe):",
            [(f"{C.VARIANT_SHORT[variant]}", (variant,))], lambda vv: _study(vv)):
    study = _study(variant)
    st.plotly_chart(build_rho_bars(study, f"Schritte je Start-ρ, {C.VARIANT_SHORT[variant]}, n = {C.RHO_N}"), width="stretch", key="rho_bars")
    st.plotly_chart(build_tol_chart(study, f"Wert und sichere Schranke je Toleranz, {C.VARIANT_SHORT[variant]}"), width="stretch", key="tol_chart")
    verdict(verdict_admm(study, variant))

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Das Runden")
st.caption("Die Lösung ist gebrochen. Rundet man Zeile 0 von Y (die relaxierte Zuordnung) durch lineare Zuordnung, entsteht eine Anordnung: wie oft ist sie das Optimum, gegen 200 Abstiege aus Zufall (Stück 12)? Gerechnet mit den Reihen aus Experiment 2.")
if ladder_ok:
    st.plotly_chart(build_round_bars(rows_by_n, round_stats, "Trifft die gerundete Lösung das Optimum?"), width="stretch", key="round_bars")
    tab = []
    for m in C.LADDER_SIZES:
        r2, r3 = round_stats(rows_by_n[m], "r2"), round_stats(rows_by_n[m], "r3")
        if r3["count"]:
            tab.append({"n": m, "Instanzen mit Optimum": r3["count"], "Rundung R2 trifft": r2["hit"], "Rundung R3 trifft": r3["hit"], "Lücke der Rundung R3 (Median)": pct(r3["median"], 1), "200 Abstiege treffen": r3["descents_hit"]})
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    verdict(verdict_round(rows_by_n))
else:
    st.info("Rechnen Sie zuerst die Leiter in Experiment 2.")

st.markdown("---")

# --- Experiment 5 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Die Größe")
st.caption(f"Eine Instanz je Größe (Familie „zwei Cluster“, Seed {C.FIXED_SEED0}), beide Stufen, Schrittgrenze {count(C.SCALE_MAX_ITER)}: Zeit je Schritt (Eigenzerlegung der PSD-Matrix, wächst wie r³) und Schritte; hohle Punkte sind nicht konvergiert.")
if run_knob("scale_ok", "⏱️ Größen rechnen (n = 6 bis 16)", "Die Rechnung läuft über sechs Größen mit beiden Stufen und dauert einige Minuten (n = 16 mit R3 an der Schrittgrenze etwa zwei Minuten; auf Streamlit Cloud länger). Mit dem Knopf starten:",
            [(f"n = {m}, {C.VARIANT_SHORT[vv]}", (m, vv)) for m in C.SCALE_SIZES for vv in C.VARIANTS], lambda m, vv: _scale(m, vv)):
    srows = [_scale(m, vv) for m in C.SCALE_SIZES for vv in C.VARIANTS]
    st.plotly_chart(build_scale(srows), width="stretch", key="scale_chart")
    tab = [{"n": r["n"], "Stufe": C.VARIANT_SHORT[r["variant"]], "N": r["N"], "PSD-Matrix": f"{r['r']}×{r['r']}", "Schritte": count(r["iterations"]), "konvergiert": "ja" if r["converged"] else "nein", "Zeit je Schritt": f"{de(1000 * r['per_iteration'], 1)} ms",
            "sichere Schranke": de(r["bound"], 1), "Gilmore–Lawler": de(r["glb"], 1), "beste von 200 Abstiegen": num(r["ub"])} for r in srows]
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    verdict(verdict_scale(srows))
    st.caption("Der Vergleich mit einem dichten Innere-Punkte-Löser (Barrierenmethode, Gleichungen durch Nullraum eliminiert) ist **nur bis n = 8** gemessen (gleiche Werte auf vier Stellen; er braucht bei n = 8 rund neun Sekunden gegen etwa eine bis sechs des ADMM); darüber ist er hier nicht gemessen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eigene Herleitung** | Lift, Pfeil, Gangster, Facial Reduction, die Stufen R2 und R3, der ADMM und die sichere Schranke sind eigene Fassungen; Zhao, Karisch, Rendl, Wolkowicz (Relaxation), Povh und Rendl (Nichtnegativität) und Wen, Goldfarb, Yin (ADMM für SDP) sind **nicht eingesehen**. Die Namen R2 und R3 sind Bezeichnungen dieser Demo. | Konvexe Linie (SDP), Stück 3 bis 6 |
| **Nur kleine Größen mit bekanntem Optimum** | n = 6 bis 12; bei n = 16 nur eine Instanz ohne Optimum. Wie die Stufen bei n = 20 und mehr arbeiten, ist nicht untersucht. | – |
| **Der ADMM konvergiert nicht immer** | R3 braucht bei n ≥ 10 teils mehr als 8 000 Schritte; an der Schrittgrenze ist der Wert keine Schranke, die sichere Schranke bleibt gültig (aber schwächer). Die App sagt es. | – |
| **Eine Stufe, keine weiteren Schnitte** | Dreiecksungleichungen und andere Schnitte (stärkere Relaxationen) sind nicht gebaut; die Schärfe von R3 gilt für die gemessenen Instanzen und Familien. | – |
| **Rundung nur durch lineare Zuordnung** | Es gibt bessere Rundungen (aus der Lösung mehrere Anordnungen ziehen, dann Tausch); hier steht nur die einfachste. | Stück 12 (Tauschabstieg) |
| **Schranke im Baum nicht gemessen** | Ob die SDP am Knoten die Knotenzahl senkt, ist offen: sie kostet Sekunden je Knoten. | Stück 8 |
| **Innere-Punkte-Vergleich nur bis n = 8** | Der Vergleich der Zeit mit einem dichten Innere-Punkte-Löser ist bis n = 8 gemessen; darüber steht nur eine Größenordnungs-Schätzung in der README. | – |
| **Die sechs Familien sind erfundene** | Echte Platinen haben andere Strukturen; Aussagen über Familien gelten nur für diese (zwei Cluster bei n = 9 verhalten sich anders als bei n = 8 und 12). | Stück 2 |
| **Zeit ist implementierungsabhängig** | numpy, Eigenzerlegung; in anderer Umsetzung liegen die Zeiten woanders und schwanken von Lauf zu Lauf um das Doppelte. In Tests stehen nie Zeiten. | – |
| **Wenige Instanzen** | n = 6, 8: 30 Instanzen, n = 9, 12: 18; kleine Unterschiede sind nicht belastbar. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe.** Flüsse $F$, Abstände $D$ (Manhattan auf dem Raster), Anordnung $\pi$: $\mathrm{cost}(\pi) = \sum_{i<k} F_{ik} D_{\pi_i \pi_k}$. Mit der Permutationsmatrix $X$ ($X_{ij} = 1$ für Bauteil $i$ auf Platz $j$) und $x = \mathrm{vec}(X)$: $\mathrm{cost} = \tfrac12 x^{\top} (D \otimes F)\, x$.

**Der Lift.** $Y = \begin{pmatrix} 1 \\ x \end{pmatrix} \begin{pmatrix} 1 \\ x \end{pmatrix}^{\top}$ der Größe $N = n^2 + 1$; $\mathrm{cost} = \langle L, Y \rangle$ mit $L = \begin{pmatrix} 0 & 0 \\ 0 & \tfrac12 D \otimes F \end{pmatrix}$.

**Die Relaxation.** Man lässt $\mathrm{rank}\, Y = 1$ fallen: $\min \langle L, Y \rangle$ mit $Y \succeq 0$, $Y_{00} = 1$, **Pfeil** $Y_{0a} = Y_{aa}$, **Gangster** $Y_{ab} = 0$ für $a = (i,j)$, $b = (k,l)$ mit genau einer Übereinstimmung $i = k$ oder $j = l$; bei **R3** zusätzlich $Y \ge 0$.
Jede Anordnung ist zulässig (Test: Permutationspunkt), also ist das Minimum eine untere Schranke der Kosten.

**Facial Reduction.** $Y = \hat V R \hat V^{\top}$ mit $\hat V = \begin{pmatrix} 1 & 0 \\ \tfrac1n e & V \otimes V \end{pmatrix}$, $V \in \mathbb{R}^{n \times (n-1)}$ orthonormal, $V^{\top} e = 0$; das erzwingt $X e = e$ und $X^{\top} e = e$. Mit $\tilde V = \hat V\, \mathrm{diag}(1/\sqrt2, 1, \dots, 1)$ (orthonormale Spalten) ist $K = \{\tilde V \tilde R \tilde V^{\top} : \tilde R \succeq 0\}$, $r = (n-1)^2 + 1$, und $P_K(Y) = \tilde V\, P_{\mathrm{PSD}}(\tilde V^{\top} Y \tilde V)\, \tilde V^{\top}$.

**ADMM.** $Y^{+} = P_{\mathcal A}(Z - U - L/\rho)$, $Z^{+} = P_K(Y^{+} + U)$, $U^{+} = U + Y^{+} - Z^{+}$. $P_{\mathcal A}$ setzt die Gangster-Einträge auf 0, $Y_{00}$ auf 1, mittelt je Pfeil-Paar (mit den Gewichten 2 und 1 der Einträge $(0,a)$, $(a,a)$) und klemmt bei R3 auf $\ge 0$. Residuen $\lVert Y - Z \rVert_F$ und $\rho \lVert Z^{+} - Z \rVert_F$.

**Sichere untere Schranke.** Mit $M = -\rho U \in K^{*}$ und Multiplikatoren $a$ (für $Y_{00} = 1$) und $y_k$ (Pfeil) aus kleinsten Quadraten: $T = L - \mathcal A^{*}(a, y)$ (bei R3 $- N$, $N = \max(0, L - \mathcal A^{*}y - M) \ge 0$), $T_{\mathrm{gang}} = M_{\mathrm{gang}}$. Für jedes zulässige $Y$ gilt $\langle L, Y \rangle \ge a + \langle T, Y \rangle \ge a + (n+1)\min(0, \lambda_{\min}(\tilde V^{\top} T \tilde V))$, weil $\mathrm{tr}\, Y = n + 1$ ($Y_{00} = 1$, Pfeil und $\sum x = n$).

**Lücke.** $100\,(U - b)/U$ mit dem Vergleichswert $U$ (Optimum oder beste gefundene Anordnung) und der Schranke $b$.

**Literatur.** Die SDP-Relaxation des QAP mit Facial Reduction und Gangster-Bedingung (Zhao, Karisch, Rendl, Wolkowicz), die Nichtnegativität (Povh und Rendl) und den ADMM für SDP (Wen, Goldfarb, Yin) kenne ich nur als Namen aus der Fachliteratur: **nicht nachgelesen**. Die Fassung hier ist eigen und wird durch Tests geprüft
(Aufzählung aller Anordnungen, Permutationspunkt, zweiter Algorithmus: ein dichter Innere-Punkte-Löser, schwache Dualität). Die Zahlen sind eigene Rechnungen an eigenen, erfundenen Familien.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
