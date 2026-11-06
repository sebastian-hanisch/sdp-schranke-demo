"""Abbildungen: sie entstehen für jede Eingabe, Achsen sind gesperrt, Daten stimmen mit den Werten überein; PDF-Export."""

import numpy as np
import pytest

import sdp_constants as C
import sdp_evaluation as E
import sdp_instance as I
import sdp_visualization as V
from oracle import make_ladder_row
from sdp_pdf_export import build_pdf
from test_verdicts import _scale_rows, _study_rows


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


@pytest.fixture(scope="module")
def v8():
    return E.instance_data(E.Settings())


def test_arrangement_for_every_family_and_size():
    for fam in C.FAMILIES:
        for n in (4, 6, 9, 12):
            inst = I.make_instance(fam, n, 3)
            fig = V.build_arrangement(inst, list(range(n)), fam)
            assert len(fig.data[-1].x) == n and _locked(fig)


def test_lift_chart_shows_y_and_x_and_the_reference_arrangement(v8):
    fig = V.build_lift(v8["rel"], v8["res"].Z, v8["ref_perm"])
    y, x, mark = fig.data
    assert np.asarray(y.z).shape == (65, 65) and np.asarray(x.z).shape == (8, 8) and list(mark.x) == [int(p) for p in v8["ref_perm"]] and list(mark.y) == list(range(8))
    assert np.allclose(np.asarray(x.z), v8["res"].Z[0, 1:].reshape(8, 8).T) and _locked(fig)


def test_residual_chart_has_one_point_per_step(v8):
    fig = V.build_residuals(v8["res"])
    assert [t.name for t in fig.data] == ["Primalresiduum", "Dualresiduum", "ρ"] and len(fig.data[0].x) == v8["res"].iterations == len(fig.data[1].y) and fig.layout.yaxis.type == "log" and _locked(fig)


def test_instance_bounds_chart(v8):
    fig = V.build_instance_bounds(v8, "r3")
    assert list(fig.data[0].x)[-1] == pytest.approx(v8["res"].bound) and list(fig.data[0].x)[:4] == [v8["bound"][k] for k in v8["bound"]] and len(fig.layout.shapes) == 1 and _locked(fig)
    assert list(fig.data[0].y)[-1] == "SDP R3, sichere Schranke"


def test_ladder_charts():
    rows = {8: [make_ladder_row(8, 100.0, r2=85.0, r3=100.0, family="sparse"), make_ladder_row(8, 200.0, r2=150.0, r3=199.0, family="cluster")], 12: [make_ladder_row(12, 300.0, r2=250.0, r3=299.0)]}
    fig = V.build_ladder_bars(rows, E.ladder_stats, "t")
    names = [t.name for t in fig.data]
    assert names == [C.LADDER_NAMES[k] for k in C.LADDER_KEYS] and len(fig.data[3].x) == 1 and len(fig.data[5].x) == 2 and fig.data[5].y[0] == pytest.approx(E.ladder_stats(rows[8], "r3")["median"])
    fam = V.build_family_bars(rows[8], E.family_medians, "t")
    assert [t.name for t in fam.data] == ["SDP R2", "SDP R3"] and list(fam.data[0].x) == ["Dünn besetzt", "Zwei Cluster"] and fam.data[0].y[0] == pytest.approx(15.0)
    rb = V.build_round_bars(rows, E.round_stats, "t")
    assert [t.name for t in rb.data] == ["Rundung R2", "Rundung R3", "200 Abstiege"] and list(rb.data[0].x) == ["n = 8", "n = 12"]


def test_admm_charts():
    rows = _study_rows()
    rho = V.build_rho_bars(rows, "t")
    assert [t.name for t in rho.data] == ["ρ fest", "ρ angepasst"] and list(rho.data[0].y) == [400.0, 2600.0] and list(rho.data[1].y) == [200.0, 1300.0] and list(rho.data[0].x) == ["1,0", "10,0"]
    tol = V.build_tol_chart(rows, "t")
    assert len(tol.data) == 4 and tol.layout.xaxis.type == "log" and list(tol.data[0].y) == pytest.approx([0.0005, 0.4]) and list(tol.data[1].y) == pytest.approx([-0.001, -1.0])
    assert tol.layout.xaxis.autorange == "reversed" and len(tol.layout.shapes) == 1


def test_scale_chart_marks_the_unconverged_runs():
    fig = V.build_scale(_scale_rows())
    r3 = [t for t in fig.data if t.name == "SDP R3"]
    assert len(r3) == 2 and list(r3[0].x) == list(C.SCALE_SIZES) and list(r3[0].marker.symbol).count("circle-open") == 2 and fig.layout.yaxis.type == "log"


def test_pdf_export(v8):
    pdf = build_pdf(E.Settings(), v8)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500
    s = E.Settings(variant="r2", tol=1e-2, adaptive=False)
    assert build_pdf(s, E.instance_data(s))[:4] == b"%PDF"
