"""Pfade der App, die der Rauchtest nicht erreicht: die Experimente im Seiteninhalt, PDF, beide Stufen, jede Familie, n = 12, die Rechenknöpfe (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import pytest
import streamlit as st

import sdp_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
KIND = {"family_select": "selectbox", "n_select": "select_slider", "seed_input": "number_input", "variant_select": "radio", "tol_select": "select_slider", "rho_select": "select_slider", "adaptive_toggle": "toggle"}


@pytest.fixture(autouse=True)
def _clear_caches():
    """st.cache_data gilt im ganzen Prozess: Tests, die Konstanten ändern, dürfen keine Ergebnisse an andere weitergeben."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


LABELS = {"Sichere untere Schranke", "Optimum", "Schritte des ADMM", "Wert des ADMM", "Rechenzeit"}


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        getattr(at.sidebar, KIND[key])(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown) + list(at.caption))


def test_default_page_shows_the_metrics_and_the_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert set(values) == LABELS and values["Optimum"] == "183" and values["Sichere untere Schranke"] == "183,00" and values["Rechenzeit"].endswith(" s")
    assert next(m for m in at.metric if m.label == "Schritte des ADMM").delta == "konvergiert" and next(m for m in at.metric if m.label == "Sichere untere Schranke").delta == "Lücke 0,00 %"
    text = _texts(at)
    for part in ("Familie **Zwei Cluster**, 8 Bauteile, Raster 2×4, Seed 48, SDP R3", "Die sichere Schranke liegt auf dem Vergleichswert", "N = 65", "Optimum **6** (Anordnung ACBD)", "fünf Experimente"):
        assert part in text, part
    assert len(at.get("plotly_chart")) == 3


def test_the_experiments_the_hand_example_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬 Experiment")] == [f"🔬 Experiment {i}" for i in (1, 2, 3, 4, 5)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "Handbeispiel zum Nachrechnen", "📐 Mathematische Formulierung"]
    assert not at.error
    assert any("Gilmore–Lawler-Schranke der Wurzel ist **5**" in str(m.value) and "`N = 4² + 1 = 17`" in str(m.value) and "**6**" in str(m.value) and "`n²(n−1) = 48`" in str(m.value) for m in at.markdown)
    assert [b.label for b in at.button if "⏱️" in b.label] == ["⏱️ Die Leiter rechnen (n = 6, 8, 9, 12)", "⏱️ ρ und Toleranz rechnen", "⏱️ Größen rechnen (n = 6 bis 16)"]


def test_r2_shows_a_weaker_bound_on_a_sparse_instance():
    at = _app(family_select="sparse", seed_input=100000, variant_select="r2")
    text = _texts(at)
    assert "SDP R2" in text and "**unter** der besten der anderen Schranken" in text and next(m for m in at.metric if m.label == "Sichere untere Schranke").delta != "Lücke 0,00 %"


def test_four_places_and_nine_places_render():
    for n in (4, 6, 9):
        at = _app(n_select=n)
        assert f"{n} Bauteile" in _texts(at) and len(at.get("plotly_chart")) == 3 and {m.label for m in at.metric} == LABELS


def test_every_family_and_both_stages_render():
    for fam in C.FAMILIES:
        for v in C.VARIANTS:
            at = _app(family_select=fam, variant_select=v, n_select=6)
            assert f"Familie **{C.FAMILY_NAMES[fam]}**" in _texts(at) and f"SDP {v.upper()}" in _texts(at)


def test_coarse_tolerance_and_fixed_rho_change_the_run():
    base = next(m for m in _app().metric if m.label == "Schritte des ADMM").value
    coarse = next(m for m in _app(tol_select=1e-2).metric if m.label == "Schritte des ADMM").value
    assert int(coarse.replace(" ", "")) < int(base.replace(" ", ""))
    fixed = _app(rho_select=10.0, adaptive_toggle=False)
    assert not fixed.exception and next(m for m in fixed.metric if m.label == "Schritte des ADMM").value != base


def test_twelve_waits_for_its_button(monkeypatch):
    monkeypatch.setitem(C.MAX_ITER, "r3", 60)                         # nur für die Geschwindigkeit des Tests: an der Schrittgrenze endet der Lauf, der Text sagt es
    at = _app(n_select=12)
    assert "⏱️ Relaxation rechnen" in [b.label for b in at.button] and not at.metric and "Mit dem Knopf starten" in _texts(at)
    at.button(key="heavy_button").click().run()
    assert not at.exception and {m.label for m in at.metric} == (LABELS - {"Optimum"}) | {"Beste von 200 Abstiegen"} and "nicht konvergiert" in _texts(at) and "beste von 200 Abstiegen" in _texts(at)


def test_the_pdf_button_makes_a_download():
    at = _app(n_select=6)
    at.button(key="pdf_make").click().run()
    assert not at.exception and any(d.proto.label == "⬇️ PDF herunterladen" for d in at.get("download_button"))


def test_the_comparison_buttons_fill_their_sections(monkeypatch):
    monkeypatch.setattr(C, "LADDER_SIZES", (6, 8))                                # nur für die Geschwindigkeit des Tests: die echten Reihen stehen in test_claims.py
    monkeypatch.setattr(C, "SEEDS", {6: 1, 8: 1, 9: 1, 12: 1})
    monkeypatch.setattr(C, "RHO_ROWS", (("uniform", 100001),))
    monkeypatch.setattr(C, "RHO_CHOICES", (1.0, 3.0))
    monkeypatch.setattr(C, "TOL_ROWS", (1e-2, 1e-4))
    monkeypatch.setattr(C, "RHO_REF_MAX_ITER", 3000)
    monkeypatch.setattr(C, "SCALE_SIZES", (6, 8))
    monkeypatch.setattr(C, "SCALE_MAX_ITER", 300)
    at = _app()
    assert "Rechnen Sie zuerst die Leiter in Experiment 2." in _texts(at)
    at.button(key="ladder_ok_button").click().run()
    text = _texts(at)
    assert not at.exception and "n = 6 (6 Instanzen)" in text and "n = 8 (6 Instanzen)" in text and "Rechnen Sie zuerst die Leiter" not in text and "Rundung trifft das Optimum" in text and len(at.get("plotly_chart")) == 3 + 3
    at.radio(key="n2_select").set_value(8).run()
    assert not at.exception and "Bei n = 8, Lücke im Median je Familie" in _texts(at)
    at.button(key="study_ok_button").click().run()
    text = _texts(at)
    assert not at.exception and "SDP R3, n = 8, drei Familien" in text and len(at.get("plotly_chart")) == 3 + 3 + 2
    at.button(key="scale_ok_button").click().run()
    text = _texts(at)
    assert not at.exception and "Zeit je Schritt" in text and "Nicht konvergiert in 300 Schritten" in text and len(at.get("plotly_chart")) == 3 + 3 + 2 + 1
