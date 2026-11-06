"""Verdikte: datengetrieben (Art und Text folgen den Zahlen), jeder Zweig an festen Eingaben (DEMO-PLAYBOOK §1b)."""

import copy

import pytest

import sdp_constants as C
import sdp_evaluation as E
from oracle import make_ladder_row


@pytest.fixture(scope="module")
def v8():
    return E.instance_data(E.Settings())


S8 = E.Settings()


def test_core_verdict_success_when_the_bound_sits_on_the_optimum(v8):
    kind, text = E.verdict_core(S8, v8)
    assert kind == "success" and "sichere Schranke **183,00** gegen das aufgezählte Optimum 183" in text and "liegt auf dem Vergleichswert" in text and "konvergiert" in text and "Eine Instanz" in text
    assert "Andere Schranken: Gilmore–Lawler" in text and "SDP R3" in text


def test_core_verdict_info_with_the_gap_for_r2_below_the_best_other():
    v = E.instance_data(E.Settings(family="sparse", seed=100000, variant="r2"))
    kind, text = E.verdict_core(E.Settings(family="sparse", seed=100000, variant="r2"), v)
    assert kind == "info" and "Die Lücke der sicheren Schranke ist" in text and "**unter** der besten der anderen Schranken" in text and "SDP R2" in text


def test_core_verdict_names_a_missing_convergence_and_a_coarse_value(v8):
    v = copy.deepcopy(v8)
    object.__setattr__(v["res"], "converged", False)
    _, text = E.verdict_core(S8, v)
    assert "an der Schrittgrenze, nicht konvergiert" in text and "die sichere Schranke bleibt gültig, der Wert nicht" in text
    w = copy.deepcopy(v8)
    object.__setattr__(w["res"], "value", w["ref"] + 0.5)
    assert "liegt über dem Optimum: bei dieser Toleranz keine Schranke" in E.verdict_core(S8, w)[1]
    assert "keine Schranke" not in E.verdict_core(S8, v8)[1]


def test_core_verdict_for_an_instance_without_enumeration_says_upper_bound(v8):
    v = copy.deepcopy(v8)
    v["exact"] = False
    assert "die beste von 200 Abstiegen (obere Grenze des Optimums)" in E.verdict_core(S8, v)[1]


def test_core_verdict_compares_with_the_best_other_bound(v8):
    v = copy.deepcopy(v8)
    v["bound"] = dict(v["bound"], lp=v["res"].bound - 5.0)
    assert "über der besten der anderen Schranken" in E.verdict_core(S8, v)[1]
    v["bound"] = dict(v["bound"], lp=v["res"].bound + 5.0)
    assert "**unter** der besten der anderen Schranken (Linearisierung (LP)" in E.verdict_core(S8, v)[1]


def test_relax_verdict_reports_sizes_trace_and_fractionality(v8):
    kind, text = E.verdict_relax(S8, v8)
    assert kind == "info" and "N = 65" in text and "50×50" in text and "448 Gangster-Paare" in text and "Spur der Lösung ist 9,000" in text and "n + 1 = 9" in text
    w = copy.deepcopy(v8)
    n = w["rel"].n
    import numpy as np
    Z = np.array(w["res"].Z)
    Z[0, 1:] = w["rel"].permutation_point(list(range(n)))[0, 1:]
    object.__setattr__(w["res"], "Z", Z)
    assert "die Lösung ist (fast) eine Anordnung" in E.verdict_relax(S8, w)[1] and "die Lösung ist gebrochen" in text


def test_hand_verdict():
    kind, text = E.verdict_hand(E.hand_data())
    assert kind == "info" and "Optimum **6** (Anordnung ACBD)" in text and "Gilmore–Lawler 5" in text and "Dual Ascent 5,94" in text and "Linearisierung 6" in text and "N = 17, r = 10, 48 Gangster-Paare" in text and "Spur 5 = n + 1" in text


def test_ladder_verdict_has_a_clause_per_size_and_counts_the_violations():
    rows = {8: [make_ladder_row(8, 100.0, r2=85.0, r3=100.0), make_ladder_row(8, 200.0, r2=150.0, r3=199.0)], 12: [make_ladder_row(12, 300.0, r2=250.0, r3=299.0)]}
    kind, text = E.verdict_ladder(rows)
    assert kind == "info" and "n = 8 (2 Instanzen)" in text and "n = 12 (1 Instanzen)" in text and "R2 unter Gilmore–Lawler in 1" in text and "R3 über der besten der anderen in 2" in text and "Sichere Schranke über dem Vergleichswert in 0 Läufen" in text
    assert "Linearisierung" in text.split("n = 12")[0] and "Linearisierung" not in text.split("n = 12")[1].split(";")[0]
    rows[8].append(make_ladder_row(8, 100.0, r2=85.0, r3=101.0))
    assert "Sichere Schranke über dem Vergleichswert in 1 Läufen" in E.verdict_ladder(rows)[1]


def test_family_verdict():
    rows = [make_ladder_row(8, 100.0, r2=80.0, r3=99.0, family="sparse"), make_ladder_row(8, 100.0, r2=99.0, r3=100.0, family="cluster")]
    kind, text = E.verdict_family(rows, 8)
    assert kind == "info" and "Bei n = 8" in text and "Dünn besetzt 20,0 / 1,00 %" in text and "Zwei Cluster 1,0 / 0,00 %" in text


def _study_rows():
    rows = []
    for fm in ("uniform", "cluster"):
        base = {"family": fm, "ref": 100.0, "opt": 100.0}
        for rho, it in ((1.0, 400), (10.0, 2600)):
            rows.append({**base, "kind": "rho", "rho": rho, "adaptive": False, "iterations": it, "converged": True, "seconds": 1.0, "value": 100.0, "bound": 99.9})
            rows.append({**base, "kind": "rho", "rho": rho, "adaptive": True, "iterations": it // 2, "converged": it < 2000, "seconds": 1.0, "value": 100.0, "bound": 99.9})
        for tol, val, b in ((1e-2, 100.4, 99.0), (1e-5, 100.0005, 99.999)):
            rows.append({**base, "kind": "tol", "tol": tol, "iterations": 10, "converged": True, "seconds": 1.0, "value": val, "bound": b})
    return rows


def test_admm_verdict_reports_the_ranges_and_the_coarse_tolerance():
    kind, text = E.verdict_admm(_study_rows(), "r3")
    assert kind == "info" and "SDP R3, n = 8, drei Familien" in text and "von 400 bis 2 600 (festes ρ)" in text and "und 200 bis 1 300 (angepasst)" in text and "2 von 8 Läufen an der Schrittgrenze" in text
    assert "Bei Toleranz 1e-2 liegt der Wert des ADMM bis zu 0,40 **über** dem Optimum" in text and "die sichere Schranke höchstens 1,00 darunter" in text and "bei 1e-5 höchstens 0,001" in text and "in 0 von 12 Läufen über dem Optimum" in text
    rows = _study_rows()
    rows[-1]["bound"] = 100.5
    assert "in 1 von 12 Läufen über dem Optimum" in E.verdict_admm(rows, "r3")[1]


def test_round_verdict():
    rows = {9: [make_ladder_row(9, 100.0, r2=90.0, r3=100.0, round2=110.0, round3=100.0, ub=100.0), make_ladder_row(9, 100.0, r2=90.0, r3=99.5, round2=120.0, round3=103.0, ub=100.0)], 12: [make_ladder_row(12, 300.0, r2=250.0, r3=299.0, opt=None)]}
    kind, text = E.verdict_round(rows)
    assert kind == "info" and "n = 9: Rundung trifft das Optimum bei R3 in 1 von 2 Instanzen (Lücke im Median 1,5 %), bei R2 in 0; 200 Abstiege in 2" in text and "n = 12" not in text and "Die Schranke ist stark, das Runden der Lösung nicht" in text


def _scale_rows(nc3=(14, 16), nc2=()):
    rows = []
    for n in C.SCALE_SIZES:
        for v, nc in (("r2", nc2), ("r3", nc3)):
            rows.append({"n": n, "variant": v, "N": n * n + 1, "r": (n - 1) ** 2 + 1, "iterations": 1000, "converged": n not in nc, "seconds": 5.0, "per_iteration": 0.001 * n, "t_proj": 0.001, "value": 100.0, "bound": 99.0 * n, "glb": 90.0 * n, "ub": 100 * n,
                         "primal": 1e-5, "dual": 1e-5, "round_cost": 101 * n})
    return rows


def test_scale_verdict_lists_the_unconverged_sizes():
    kind, text = E.verdict_scale(_scale_rows())
    assert kind == "info" and "Bis n = 16 (N = 257, PSD-Matrix 226×226)" in text and "6,0 ms, 8,0 ms, 10,0 ms, 12,0 ms, 14,0 ms, 16,0 ms bei n = 6, 8, 10, 12, 14, 16" in text
    assert "R3 bei n = 14, 16; R2 bei n = –" in text and "R3 1 584,0" not in text and "Sichere Schranke bei n = 16: R3 1 584,0" not in text
    assert "R3 bei n = –" in E.verdict_scale(_scale_rows(nc3=()))[1] and "R2 bei n = 6" in E.verdict_scale(_scale_rows(nc2=(6,)))[1]
