"""Zweiter Algorithmus als Orakel (ORACLE-PLAYBOOK §5): ein dichter Innere-Punkte-Löser (Barrierenmethode, Gleichungen durch Nullraum eliminiert, `tests/ipm.py`) löst dieselbe SDP; der ADMM trifft seinen Wert."""

import pytest

import ipm
import sdp_landscape as L
import sdp_relax as R
from sdp_instance import make_instance


@pytest.mark.parametrize("family", ("uniform", "star", "skewed"))
@pytest.mark.parametrize("nonneg", (False, True))
def test_the_barrier_method_and_the_admm_agree_at_four_places(family, nonneg):
    inst = make_instance(family, 4, 100020)
    rel = R.Relaxation(inst.F, inst.D, nonneg)
    a = ipm.solve_all_perms(rel)
    b = R.admm(rel, tol=1e-7, max_iter=30000)
    assert b.value == pytest.approx(a["value"], abs=2e-3) and b.bound <= a["value"] + 1e-6
    opt = int(L.costs_of(L.all_perms(4), inst.F, inst.D).min())
    assert a["value"] <= opt + 1e-5 and a["start_res"] < 1e-9


@pytest.mark.parametrize("nonneg", (False, True))
def test_the_barrier_method_and_the_admm_agree_at_six_places(nonneg):
    inst = make_instance("uniform", 6, 100000)
    rel = R.Relaxation(inst.F, inst.D, nonneg)
    a = ipm.solve_all_perms(rel)
    b = R.admm(rel, tol=1e-6, max_iter=8000)
    assert b.value == pytest.approx(a["value"], abs=5e-3) and b.bound <= a["value"] + 1e-6 and b.bound >= a["value"] - 0.02
    assert a["value"] == pytest.approx(94.8649 if not nonneg else 95.0, abs=1e-3)                       # Messwerte der Phase 0 (Aufzählung: Optimum 95)


def test_the_barrier_solution_is_a_lower_bound_of_the_enumerated_optimum_for_both_stages():
    inst = make_instance("cluster", 6, 100000)
    opt = int(L.costs_of(L.all_perms(6), inst.F, inst.D).min())
    v2 = ipm.solve_all_perms(R.Relaxation(inst.F, inst.D, False))["value"]
    v3 = ipm.solve_all_perms(R.Relaxation(inst.F, inst.D, True))["value"]
    assert v2 <= v3 + 1e-5 and v3 <= opt + 1e-5 and v2 == pytest.approx(82.3862, abs=1e-3)
