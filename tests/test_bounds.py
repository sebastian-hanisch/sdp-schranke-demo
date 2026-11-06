"""Schranken der reinen Zuordnung (Kopien aus Stück 3 und 5) und ihr Orakel: die Aufzählung aller Anordnungen (ORACLE-PLAYBOOK §5)."""

import numpy as np
import pytest

import sdp_bounds as B
import sdp_constants as C
import sdp_landscape as L
import sdp_lp as LP
from sdp_instance import make_instance


def enumerated(inst):
    P = L.all_perms(inst.n)
    c = L.costs_of(P, inst.F, inst.D)
    return int(c.min()), P, c


def test_hand_example_the_bound_lies_below_the_optimum(hand_gap):
    """Vier Plätze auf einer Zeile: Gilmore–Lawler 5, Optimum 6 (Handrechnung in der README)."""
    g = B.gilmore_lawler(hand_gap.F, hand_gap.D)
    assert g.glb == 5.0 and g.lap == 10 and enumerated(hand_gap)[0] == 6 and g.rounded_cost >= 6


def test_hand_example_dual_ascent_rises_towards_the_optimum_and_the_lp_reaches_it(hand_gap):
    da = B.dual_ascent(hand_gap.F, hand_gap.D, 5)
    assert da.bounds == (5.0, 5.5, 5.75, 5.875, 5.9375) and da.final < 6
    assert LP.solve_lp(hand_gap.F, hand_gap.D).value == pytest.approx(6.0)


@pytest.mark.parametrize("family", C.FAMILIES)
def test_bounds_are_ordered_and_never_above_the_optimum(family):
    for seed in (100000, 100001):
        inst = make_instance(family, 6, seed)
        opt, _, _ = enumerated(inst)
        g = B.gilmore_lawler(inst.F, inst.D)
        da = B.dual_ascent(inst.F, inst.D, 5)
        lp = LP.solve_lp(inst.F, inst.D).value
        assert g.glb <= opt + 1e-9 and da.final <= opt + 1e-6 and lp <= opt + 1e-6
        assert da.bounds[0] == pytest.approx(g.glb) and all(b2 >= b1 - 1e-9 for b1, b2 in zip(da.bounds, da.bounds[1:])) and da.final >= g.glb - 1e-9


@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_best_found_is_an_upper_bound_and_a_valid_arrangement(family):
    inst = make_instance(family, 8, 100003)
    opt, _, _ = enumerated(inst)
    ub, perm = B.best_found(inst.F, inst.D, 100003, 50)
    assert ub >= opt and sorted(int(x) for x in perm) == list(range(8))
    assert ub == int(L.costs_of(np.array([perm]), inst.F, inst.D)[0])


def test_descents_end_in_local_optima():
    inst = make_instance("uniform", 8, 100004)
    ends, costs = B.descents(inst.F, inst.D, B.random_starts(8, 12, 100004))
    for q, c in zip(ends, costs):
        assert int(L.costs_of(np.array([q]), inst.F, inst.D)[0]) == c
        for a in range(8):
            for b in range(a + 1, 8):
                r = list(q)
                r[a], r[b] = r[b], r[a]
                assert int(L.costs_of(np.array([r]), inst.F, inst.D)[0]) >= c


def test_the_lp_solution_of_a_permutation_point_costs_the_arrangement(hand_gap):
    lp = LP.build_lp(hand_gap.F, hand_gap.D)
    z = LP.permutation_point(lp, [1, 0, 3, 2])
    assert float(lp.c @ z) == pytest.approx(float(L.costs_of(np.array([[1, 0, 3, 2]]), hand_gap.F, hand_gap.D)[0]))
    assert np.allclose(lp.A @ z, lp.rhs)
