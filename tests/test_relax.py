"""Die Relaxation, ihre Projektionen, der ADMM und die sichere Schranke: Handwerte, Eigenschaften und Orakel (Aufzählung aller Anordnungen; ein zweiter Algorithmus, siehe test_oracle_ipm.py)."""

from itertools import permutations

import numpy as np
import pytest

import sdp_constants as C
import sdp_landscape as L
import sdp_relax as R
from sdp_cost import cost as qap_cost
from sdp_instance import make_instance


def enum_opt(inst):
    return int(L.costs_of(L.all_perms(inst.n), inst.F, inst.D).min())


@pytest.mark.parametrize("nonneg", (False, True))
def test_hand_example_sizes_and_counts(hand_gap, nonneg):
    """Vier Plätze auf einer Zeile: N = 17, r = 10, 48 Gangster-Paare (n²(n−1)); n = 6: 37, 26, 180."""
    rel = R.Relaxation(hand_gap.F, hand_gap.D, nonneg)
    assert (rel.N, rel.r, rel.gangster_pairs, rel.n) == (17, 10, 48, 4)
    inst = make_instance("uniform", 6, 3)
    r6 = R.Relaxation(inst.F, inst.D, nonneg)
    assert (r6.N, r6.r, r6.gangster_pairs) == (37, 26, 180) and r6.gangster_pairs == 6 ** 2 * 5


def test_the_reduced_basis_is_orthonormal_and_spans_the_row_and_column_sums():
    inst = make_instance("star", 6, 5)
    rel = R.Relaxation(inst.F, inst.D)
    assert np.allclose(rel.Vt.T @ rel.Vt, np.eye(rel.r))
    n = 6
    ones_cols = np.zeros((rel.N, n))
    # Zeilensummen von X: Σ_j x_{(i,j)} − 1·y0 liegt im Kern der Ṽ-Bildes? Prüfung: jedes Y in K hat Σ_j Y_{0,(i,j)} = Y00 und Σ_i Y_{0,(i,j)} = Y00
    rng = np.random.default_rng(1)
    A = rng.normal(size=(rel.r, rel.r))
    Y = rel.Vt @ (A @ A.T) @ rel.Vt.T
    row0 = Y[0, 1:].reshape(n, n).T                      # X[i, j]
    assert np.allclose(row0.sum(axis=1), Y[0, 0]) and np.allclose(row0.sum(axis=0), Y[0, 0])
    del ones_cols


@pytest.mark.parametrize("nonneg", (False, True))
def test_every_permutation_point_is_feasible_with_the_cost_of_its_arrangement(hand_gap, nonneg):
    """Alle 24 Anordnungen: Y* liegt in der affinen Menge und im Kegel, <L, Y*> sind die Kosten, die Spur ist n + 1."""
    rel = R.Relaxation(hand_gap.F, hand_gap.D, nonneg)
    for p in permutations(range(4)):
        Y = rel.permutation_point(p)
        assert np.abs(rel.proj_A(Y.copy()) - Y).max() < 1e-12 and np.abs(rel.proj_K(Y) - Y).max() < 1e-12
        assert rel.cost_of(Y) == pytest.approx(qap_cost(list(p), hand_gap.F, hand_gap.D)) and np.trace(Y) == pytest.approx(5.0)
        assert Y[~rel.gang & (np.arange(17)[:, None] != -1)].min() >= 0


@pytest.mark.parametrize("nonneg", (False, True))
def test_the_projections_are_idempotent_and_closest(nonneg):
    inst = make_instance("cluster", 6, 7)
    rel = R.Relaxation(inst.F, inst.D, nonneg)
    rng = np.random.default_rng(2)
    W = rng.normal(size=(rel.N, rel.N))
    W = 0.5 * (W + W.T)
    pk, pa = rel.proj_K(W), rel.proj_A(W.copy())
    assert np.abs(rel.proj_K(pk) - pk).max() < 1e-10 and np.abs(rel.proj_A(pa.copy()) - pa).max() < 1e-12
    w = np.linalg.eigvalsh(rel.Vt.T @ pk @ rel.Vt)
    assert w[0] > -1e-10 and abs(np.sum((W - pk) * pk)) < 1e-8                            # in K, und W − P_K(W) ⊥ P_K(W)
    for p in list(permutations(range(6)))[:40:7]:
        Y = rel.permutation_point(p)
        assert np.linalg.norm(W - pk) <= np.linalg.norm(W - Y) + 1e-9 and np.linalg.norm(W - pa) <= np.linalg.norm(W - Y) + 1e-9


def test_the_affine_projection_sets_the_documented_entries(hand_gap):
    rel = R.Relaxation(hand_gap.F, hand_gap.D)
    W = np.full((17, 17), 0.7)
    P = rel.proj_A(W)
    assert P[0, 0] == 1.0 and (P[rel.gang] == 0).all() and np.allclose(P, P.T)
    for a in range(1, 17):
        assert P[0, a] == pytest.approx(P[a, a]) and P[0, a] == pytest.approx((2 * 0.7 + 0.7) / 3)
    R3 = R.Relaxation(hand_gap.F, hand_gap.D, True)
    assert R3.proj_A(np.full((17, 17), -0.4)).min() >= 0


@pytest.mark.parametrize("family", C.FAMILIES)
@pytest.mark.parametrize("nonneg", (False, True))
def test_the_safe_bound_is_below_the_optimum_for_any_dual_matrix(family, nonneg):
    """Jede Dualmatrix M ∈ K* (hier zufällig, in verschiedener Größe) liefert eine Schranke ≤ Optimum (schwache Dualität, bewiesen am Spurargument)."""
    inst = make_instance(family, 6, 100010)
    rel = R.Relaxation(inst.F, inst.D, nonneg)
    opt = enum_opt(inst)
    rng = np.random.default_rng(3)
    scale = float(np.abs(rel.L).max())
    for s in (0.0, 0.1, 1.0, 10.0):
        A = rng.normal(size=(rel.r, rel.r)) * s * scale / rel.r
        M = rel.Vt @ (A @ A.T) @ rel.Vt.T
        assert rel.safe_bound(M) <= opt + 1e-9


@pytest.mark.parametrize("family", C.FAMILIES)
def test_admm_bound_below_optimum_at_every_tolerance_and_r3_not_below_r2(family):
    inst = make_instance(family, 6, 100011)
    opt = enum_opt(inst)
    r2 = R.admm(R.Relaxation(inst.F, inst.D, False), tol=1e-6, max_iter=8000)
    r3 = R.admm(R.Relaxation(inst.F, inst.D, True), tol=1e-6, max_iter=8000)
    assert r2.bound <= opt + 1e-9 and r3.bound <= opt + 1e-9 and r3.value >= r2.value - 0.02 and r3.bound >= r2.bound - 0.02
    for tol in C.TOL_CHOICES:
        res = R.admm(R.Relaxation(inst.F, inst.D, True), tol=tol, max_iter=3000)
        assert res.bound <= opt + 1e-9, (family, tol)


def test_hand_example_both_stages_give_the_optimum(hand_gap):
    for nn in (False, True):
        rel = R.Relaxation(hand_gap.F, hand_gap.D, nn)
        res = R.admm(rel, tol=1e-7)
        assert res.converged and res.value == pytest.approx(6.0, abs=1e-3) and 5.99 < res.bound <= 6.0 + 1e-9
        assert np.trace(res.Z) == pytest.approx(5.0, abs=1e-3)
        perm = R.rounded_arrangement(rel, res.Z)
        assert sorted(perm) == [0, 1, 2, 3] and qap_cost(perm, hand_gap.F, hand_gap.D) >= 6


def test_admm_validates_and_reports_the_step_limit(hand_gap):
    rel = R.Relaxation(hand_gap.F, hand_gap.D)
    with pytest.raises(ValueError):
        R.admm(rel, rho=0.0)
    res = R.admm(rel, tol=1e-9, max_iter=3)
    assert not res.converged and res.iterations == 3 and len(res.history["primal"]) == 3 and res.variant == "r2"
    assert R.admm(R.Relaxation(hand_gap.F, hand_gap.D, True), max_iter=2).variant == "r3"


def test_adaptive_rho_changes_rho_and_fixed_rho_does_not():
    inst = make_instance("uniform", 8, 100001)
    rel = R.Relaxation(inst.F, inst.D)
    fixed = R.admm(rel, rho=0.1, tol=1e-4, max_iter=1500, adaptive=False)
    adapt = R.admm(rel, rho=0.1, tol=1e-4, max_iter=1500, adaptive=True)
    assert fixed.rho == 0.1 and fixed.rho_changes == 0 and adapt.rho_changes >= 1 and adapt.iterations <= fixed.iterations


def test_the_run_is_deterministic(hand_gap):
    rel = R.Relaxation(hand_gap.F, hand_gap.D, True)
    a, b = R.admm(rel, tol=1e-6), R.admm(rel, tol=1e-6)
    assert a.iterations == b.iterations and a.value == b.value and a.bound == b.bound and np.array_equal(a.Z, b.Z)


def test_the_rounding_returns_a_permutation_for_a_fractional_point():
    inst = make_instance("uniform", 6, 100012)
    rel = R.Relaxation(inst.F, inst.D)
    res = R.admm(rel, tol=1e-4, max_iter=400)
    perm = R.rounded_arrangement(rel, res.Z)
    assert sorted(perm) == list(range(6))
    x = rel.permutation_point(perm)
    assert np.trace(x) == pytest.approx(7.0)
