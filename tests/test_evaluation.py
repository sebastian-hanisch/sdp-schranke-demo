"""Kennzahlen, feste Instanzen, Zeilen der Vergleiche und Formatierung: an künstlichen Zeilen mit bekannten Antworten und an kleinen echten Instanzen (ORACLE-PLAYBOOK §5)."""

import math

import numpy as np
import pytest

import sdp_constants as C
import sdp_evaluation as E
import sdp_landscape as L
from oracle import make_ladder_row


def test_formatting():
    assert E.de(-1.5, 1) == "−1,5" and E.pct(12.34, 1) == "12,3 %" and E.num(27) == "27" and E.num(27.5) == "27,5" and E.count(1200) == "1 200"
    assert E.secs(0.0164) == "0,016 s" and E.secs(0.42) == "0,42 s" and E.secs(5.13) == "5,1 s" and E.secs(31.4) == "31 s"
    assert C.tol_text(1e-5) == "1e-5" and C.tol_text(1e-2) == "1e-2"


def test_gap_is_in_percent_and_zero_for_equal_values():
    assert E.gap(90.0, 100.0) == pytest.approx(10.0) and E.gap(100.0, 100.0) == 0.0 and E.gap(100.0 + 1e-12, 100.0) == 0.0 and E.gap(101.0, 100.0) == pytest.approx(-1.0)
    assert math.isnan(E.gap(1.0, 0.0))


def test_fixed_instances():
    assert {n: len(E.ladder_jobs(n)) for n in C.LADDER_SIZES} == {6: 30, 8: 30, 9: 18, 12: 18}
    jobs = E.ladder_jobs(8)
    assert [j[2] for j in jobs if j[0] == "star"] == list(range(C.FIXED_SEED0, C.FIXED_SEED0 + 5)) and {j[0] for j in jobs} == set(C.FAMILIES) and len(set(jobs)) == len(jobs)


def test_reference_exact_tree_and_upper_bound(monkeypatch):
    from sdp_instance import make_instance
    inst6 = make_instance("cluster", 6, 100000)
    r6 = E.reference(inst6, 100000)
    assert r6["exact"] and r6["opt"] == r6["ref"] == int(L.costs_of(L.all_perms(6), inst6.F, inst6.D).min()) and r6["ub"] >= r6["opt"] and sorted(r6["ref_perm"]) == list(range(6))
    inst12 = make_instance("sparse", 12, 100000)
    plain = E.reference(inst12, 100000)
    assert not plain["exact"] and plain["opt"] is None and plain["ref"] == plain["ub"]
    proved = E.reference(inst12, 100000, tree=True)
    assert proved["exact"] and proved["opt"] <= plain["ub"] and proved["tree_nodes"] > 0 and proved["ref"] == proved["opt"]
    monkeypatch.setattr(C, "TREE_CAP", 5)
    capped = E.reference(inst12, 100000, tree=True)
    assert not capped["exact"] and capped["opt"] is None and capped["ref"] == capped["ub"]


def test_other_bounds_follow_the_size():
    from sdp_instance import make_instance
    b6, s6 = E.other_bounds(make_instance("uniform", 6, 100000))
    b12, s12 = E.other_bounds(make_instance("uniform", 12, 100000))
    assert set(b6) == {"glb", "pb", "da5", "lp"} and set(b12) == {"glb", "pb", "da5"} and set(s6) == set(b6)
    assert b6["glb"] <= b6["da5"] + 1e-9 and all(v > 0 for v in b6.values())


def test_instance_data_for_the_default_settings():
    s = E.Settings()
    v = E.instance_data(s)
    assert v["exact"] and v["ref"] == v["opt"] == int(L.costs_of(L.all_perms(8), v["inst"].F, v["inst"].D).min())
    assert v["res"].bound <= v["ref"] + 1e-9 and v["res"].variant == "r3" and sorted(v["round_perm"]) == list(range(8)) and v["round_cost"] >= v["ref"]
    assert set(v["bound"]) == {"glb", "pb", "da5", "lp"} and all(b <= v["ref"] + 1e-6 for b in v["bound"].values()) and sorted(v["ref_perm"]) == list(range(8))
    r2 = E.instance_data(E.Settings(variant="r2"))
    assert r2["res"].variant == "r2" and r2["res"].bound <= v["res"].bound + 0.05


def test_hand_data():
    h = E.hand_data()
    assert h["opt"] == 6 and h["glb"] == 5.0 and h["da5"] == pytest.approx(5.9375) and h["lp"] == pytest.approx(6.0) and h["point_cost"] == pytest.approx(6.0) and h["point_trace"] == pytest.approx(5.0)
    for v in C.VARIANTS:
        assert h[v]["res"].value == pytest.approx(6.0, abs=1e-3) and h[v]["res"].converged
    assert "".join("ABCD"[i] for i in np.argsort(h["opt_perm"])) == "ACBD"


def test_ladder_row_on_a_small_instance():
    r = E.ladder_row("cluster", 6, 100000)
    assert r["opt"] == r["ref"] == 85.0 and r["exact"] and set(r["sdp"]) == {"r2", "r3"} and set(r["bound"]) == {"glb", "pb", "da5", "lp"}
    assert all(r["sdp"][v]["bound"] <= r["ref"] + 1e-9 for v in C.VARIANTS) and r["sdp"]["r3"]["bound"] >= r["sdp"]["r2"]["bound"] - 0.02
    assert all(r["sdp"][v]["converged"] and r["sdp"][v]["round_cost"] >= r["ref"] for v in C.VARIANTS)
    assert E.ladder_sweep("cluster", 6)[0]["seed"] == C.FIXED_SEED0 and len(E.ladder_sweep("cluster", 6)) == C.SEEDS[6]


def test_ladder_stats_and_comparison_counts_on_known_rows():
    rows = [make_ladder_row(8, 100.0, r2=85.0, r3=100.0), make_ladder_row(8, 200.0, r2=150.0, r3=199.0), make_ladder_row(8, 50.0, r2=48.0, r3=50.0)]
    st = E.ladder_stats(rows, "r3")
    assert st["count"] == 3 and st["median"] == pytest.approx(0.0) and st["mean"] == pytest.approx((0 + 0.5 + 0) / 3) and st["max"] == pytest.approx(0.5) and st["sharp"] == 2
    assert E.ladder_stats(rows, "r2")["median"] == pytest.approx(E.gap(85.0, 100.0)) or E.ladder_stats(rows, "r2")["median"] == pytest.approx(4.0)
    assert E.ladder_stats(rows, "glb")["median"] == pytest.approx(10.0)
    assert E.below(rows, "r2", "glb") == 2 and E.below(rows, "r3", "glb") == 0                      # 85 < 90, 150 < 180; 48 ≥ 45
    assert E.beats_all_others(rows, "r3") == 3 and E.beats_all_others(rows, "r2") == 0
    no_lp = [make_ladder_row(12, 300.0, r2=250.0, r3=299.0)]
    assert E.ladder_stats(no_lp, "lp")["count"] == 0 and math.isnan(E.ladder_stats(no_lp, "lp")["median"]) and E.below(no_lp, "lp", "glb") == 0


def test_family_medians():
    rows = [make_ladder_row(8, 100.0, r2=90.0, r3=100.0, family="sparse"), make_ladder_row(8, 100.0, r2=70.0, r3=99.0, family="sparse"), make_ladder_row(8, 100.0, r2=99.0, r3=100.0, family="cluster")]
    f2, f3 = E.family_medians(rows, "r2"), E.family_medians(rows, "r3")
    assert f2 == {"sparse": pytest.approx(20.0), "cluster": pytest.approx(1.0)} and f3["sparse"] == pytest.approx(0.5) and list(f2) == ["sparse", "cluster"]


def test_round_convergence_and_violation_stats():
    rows = [make_ladder_row(9, 100.0, r2=90.0, r3=100.0, round2=110.0, round3=100.0, ub=100.0, iterations=300, seconds=1.0),
            make_ladder_row(9, 100.0, r2=90.0, r3=99.5, round2=120.0, round3=102.0, ub=101.0, iterations=700, converged=False, seconds=3.0),
            make_ladder_row(12, 100.0, r2=90.0, r3=99.5, opt=None, round2=130.0, round3=130.0, ub=100.0)]
    rs = E.round_stats(rows, "r3")
    assert rs["count"] == 2 and rs["hit"] == 1 and rs["median"] == pytest.approx(1.0) and rs["descents_hit"] == 1
    assert E.round_stats(rows, "r2")["hit"] == 0 and E.round_stats([rows[2]], "r3")["count"] == 0 and math.isnan(E.round_stats([rows[2]], "r3")["median"])
    cs = E.convergence_stats(rows[:2], "r3")
    assert cs["not_converged"] == 1 and cs["iterations_median"] == pytest.approx(500.0) and cs["iterations_max"] == 700 and cs["sec_max"] == 3.0
    assert E.safe_violations(rows) == 0
    bad = make_ladder_row(8, 100.0, r2=90.0, r3=100.5)
    assert E.safe_violations([bad]) == 1 and E.safe_violations([make_ladder_row(12, 100.0, r2=90.0, r3=105.0, opt=None)]) == 0


def test_scale_row_and_study_on_small_inputs(monkeypatch):
    monkeypatch.setattr(C, "SCALE_MAX_ITER", 200)
    r = E.scale_row(6, "r3")
    assert (r["n"], r["N"], r["r"], r["iterations"]) == (6, 37, 26, r["iterations"]) and r["iterations"] <= 200 and r["per_iteration"] > 0 and r["bound"] <= r["ub"] + 1e-9 and r["glb"] <= r["ub"]
    monkeypatch.setattr(C, "RHO_ROWS", (("uniform", 100001),))
    monkeypatch.setattr(C, "RHO_CHOICES", (1.0, 3.0))
    monkeypatch.setattr(C, "TOL_ROWS", (1e-2, 1e-4))
    monkeypatch.setattr(C, "RHO_REF_MAX_ITER", 3000)
    rows = E.admm_study("r3")
    assert [r_["kind"] for r_ in rows] == ["rho"] * 4 + ["tol"] * 2 and {r_["family"] for r_ in rows} == {"uniform"}
    assert all(r_["bound"] <= r_["opt"] + 1e-9 for r_ in rows) and [(r_["rho"], r_["adaptive"]) for r_ in rows[:4]] == [(1.0, False), (1.0, True), (3.0, False), (3.0, True)]
    tol = [r_ for r_ in rows if r_["kind"] == "tol"]
    assert tol[0]["iterations"] < tol[1]["iterations"] and abs(tol[1]["value"] - tol[1]["ref"]) < abs(tol[0]["value"] - tol[0]["ref"]) + 0.5
