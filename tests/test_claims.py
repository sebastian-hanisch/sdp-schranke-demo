"""Die Zahlen der README und der Verdikte, an den festen Instanzen der Leiter nachgerechnet (Messung, nicht Satz).

Lehre aus Stück 11 und 15: Mittelwerte und Mediane der Verfahren, die von Eigenzerlegungen und der Schrittgrenze des ADMM abhängen, streuen mit der Hardware des Rechners. Die Toleranzen sind von Anfang an auf die Streuung
zwischen Rechnern ausgelegt (Windows-Lauf des Autors gegen Linux-CI), Abweichungen werden gesammelt gemeldet statt beim ersten abzubrechen. Zeiten stehen hier nie; die Schranken der Stücke 3 bis 5, die Kosten und die Knoten des Baums
sind ganzzahlig oder genau und werden enger geprüft. Die Aussage „der ADMM braucht k Sekunden“ gehört in die App und die README (Messung dieser Fassung), nicht in einen Test."""

import pytest

import sdp_constants as C
import sdp_evaluation as E
import rows as RW

SIZES = C.LADDER_SIZES                                                          # 6, 8, 9, 12
COUNT = {6: 30, 8: 30, 9: 18, 12: 18}
# Lücke zum Optimum im Median in Prozent (Windows-Lauf des Autors); Toleranz in Prozentpunkten
MEDIAN = {
    "r2": ({6: 3.21, 8: 3.14, 9: 5.44, 12: 8.05}, {6: 0.8, 8: 0.8, 9: 1.0, 12: 1.5}),
    "r3": ({6: 0.0, 8: 0.0, 9: 0.0, 12: 0.28}, {6: 0.3, 8: 0.3, 9: 0.3, 12: 0.7}),
    "glb": ({6: 5.44, 8: 8.33, 9: 10.22, 12: 13.30}, {6: 0.05, 8: 0.05, 9: 0.05, 12: 0.05}),
    "da5": ({6: 3.85, 8: 5.20, 9: 7.97, 12: 11.31}, {6: 0.05, 8: 0.05, 9: 0.05, 12: 0.05}),
    "pb": ({6: 15.27, 8: 12.64, 9: 19.69, 12: 27.51}, {6: 0.1, 8: 0.1, 9: 0.1, 12: 0.1}),
    "lp": ({6: 2.47, 8: 1.47, 9: 1.77}, {6: 0.1, 8: 0.1, 9: 0.1}),
}
R2_BELOW_GLB = {6: 8, 8: 10, 9: 7, 12: 8}                                       # Instanzen, in denen die sichere Schranke von R2 unter Gilmore–Lawler liegt; ± 2
R2_BELOW_LP = {8: 22, 9: 15}                                                    # ± 3
R3_BEATS_ALL = {6: 21, 8: 20, 9: 16, 12: 18}                                    # R3 über der besten der anderen vier Schranken; ± 3
R3_SHARP = {6: 22, 8: 25, 9: 12, 12: 7}                                         # R3 scharf (Lücke unter 0,05 %); ± 4
ROUND_HITS = {"r3": {6: 15, 8: 19, 9: 6, 12: 3}, "r2": {6: 11, 8: 6, 9: 1, 12: 0}}   # Rundung trifft das Optimum; ± 4 (R2: ± 3)
DESCENT_HITS = {6: 30, 8: 30, 9: 18, 12: 17}                                    # 200 Abstiege treffen das Optimum: ganzzahlig, genau
NOT_CONVERGED = {"r2": {6: 0, 8: 2, 9: 0, 12: 2}, "r3": {6: 0, 8: 1, 9: 4, 12: 1}}   # an der Schrittgrenze; erlaubt: Messwert + 2


def _check(off):
    assert not off, "Abweichungen:\n" + "\n".join(off)


@pytest.mark.parametrize("n", SIZES)
def test_the_rows_are_complete_and_the_optimum_is_known(n):
    rows = RW.ladder_rows(n)
    assert len(rows) == COUNT[n] and all(r["opt"] is not None and r["ref"] == r["opt"] for r in rows)
    assert all(set(r["bound"]) == ({"glb", "pb", "da5", "lp"} if n <= C.LP_MAX_N else {"glb", "pb", "da5"}) and set(r["sdp"]) == {"r2", "r3"} for r in rows)
    if n == 12:
        import numpy as np
        assert float(np.median([r["tree_nodes"] for r in rows])) == 7990.0                  # Stück 15: 7 990 Knoten im Median (ganzzahlig, auf jedem Rechner gleich)


@pytest.mark.parametrize("n", SIZES)
def test_the_safe_bound_never_exceeds_the_optimum(n):
    """Beweis, nicht Messung: die sichere Schranke liegt in jeder Instanz und bei beiden Stufen unter dem Optimum (Toleranz 0,01 Kosteneinheiten)."""
    rows = RW.ladder_rows(n)
    assert E.safe_violations(rows) == 0
    assert all(r["sdp"][v]["bound"] <= r["opt"] + 0.01 for r in rows for v in C.VARIANTS)


@pytest.mark.parametrize("n", SIZES)
def test_median_gap_of_every_bound(n):
    rows = RW.ladder_rows(n)
    off = []
    for k, (exp, tol) in MEDIAN.items():
        if n not in exp:
            assert E.ladder_stats(rows, k)["count"] == 0
            continue
        got = E.ladder_stats(rows, k)["median"]
        if abs(got - exp[n]) > tol[n]:
            off.append(f"n = {n}, {k}: Lücke {got:.2f} % statt {exp[n]} % ± {tol[n]}")
    _check(off)


@pytest.mark.parametrize("n", SIZES)
def test_r3_is_at_least_r2_and_never_below_gilmore_lawler_dual_ascent_or_eigenvalue(n):
    rows = RW.ladder_rows(n)
    assert all(r["sdp"]["r3"]["bound"] >= r["sdp"]["r2"]["bound"] - 0.02 for r in rows)
    off = [f"n = {n}: R3 unter {k} in {E.below(rows, 'r3', k)} Instanzen" for k in ("glb", "da5", "pb") if E.below(rows, "r3", k) > 1]
    _check(off)


@pytest.mark.parametrize("n", SIZES)
def test_counts_of_the_comparison(n):
    rows = RW.ladder_rows(n)
    off = []
    got = E.below(rows, "r2", "glb")
    if abs(got - R2_BELOW_GLB[n]) > 2:
        off.append(f"n = {n}: R2 unter Gilmore–Lawler in {got} statt {R2_BELOW_GLB[n]} ± 2")
    if n in R2_BELOW_LP and abs(E.below(rows, "r2", "lp") - R2_BELOW_LP[n]) > 3:
        off.append(f"n = {n}: R2 unter der Linearisierung in {E.below(rows, 'r2', 'lp')} statt {R2_BELOW_LP[n]} ± 3")
    if abs(E.beats_all_others(rows, "r3") - R3_BEATS_ALL[n]) > 3:
        off.append(f"n = {n}: R3 über der besten anderen in {E.beats_all_others(rows, 'r3')} statt {R3_BEATS_ALL[n]} ± 3")
    if abs(E.ladder_stats(rows, "r3")["sharp"] - R3_SHARP[n]) > 4:
        off.append(f"n = {n}: R3 scharf in {E.ladder_stats(rows, 'r3')['sharp']} statt {R3_SHARP[n]} ± 4")
    _check(off)


def test_r2_is_not_the_strongest_bound_but_r3_is():
    """Der Befund gegen die Erwartung: ohne Y ≥ 0 liegt die SDP in einem Drittel der Instanzen unter Gilmore–Lawler; mit Y ≥ 0 nie."""
    total_below = sum(E.below(RW.ladder_rows(n), "r2", "glb") for n in SIZES)
    total = sum(COUNT.values())
    assert total_below >= total // 5 and sum(E.below(RW.ladder_rows(n), "r3", "glb") for n in SIZES) == 0
    for n in SIZES:
        assert E.ladder_stats(RW.ladder_rows(n), "r3")["median"] < 0.7 < E.ladder_stats(RW.ladder_rows(n), "glb")["median"]


@pytest.mark.parametrize("n", (8, 12))
def test_families_where_r2_is_weak(n):
    rows = RW.ladder_rows(n)
    r2, r3 = E.family_medians(rows, "r2"), E.family_medians(rows, "r3")
    off = []
    if not (r2["sparse"] > 8 and r2["skewed"] > 8 and r2["cluster"] < 3):
        off.append(f"n = {n}: R2 je Familie {r2}")
    if not all(v < 2.5 for v in r3.values()):
        off.append(f"n = {n}: R3 je Familie {r3}")
    _check(off)


@pytest.mark.parametrize("n", SIZES)
def test_rounding_of_the_solution(n):
    rows = RW.ladder_rows(n)
    off = []
    for v, tol in (("r3", 4), ("r2", 3)):
        got = E.round_stats(rows, v)["hit"]
        if abs(got - ROUND_HITS[v][n]) > tol:
            off.append(f"n = {n}, {v}: Rundung trifft in {got} statt {ROUND_HITS[v][n]} ± {tol}")
    st = E.round_stats(rows, "r3")
    assert st["descents_hit"] == DESCENT_HITS[n] and st["count"] == COUNT[n]
    if n >= 9:
        assert st["hit"] < st["descents_hit"]                                    # das Runden der SDP-Lösung ist schlechter als 200 Abstiege
    _check(off)


@pytest.mark.parametrize("n", SIZES)
def test_convergence_status(n):
    rows = RW.ladder_rows(n)
    off = []
    for v in C.VARIANTS:
        cs = E.convergence_stats(rows, v)
        if cs["not_converged"] > NOT_CONVERGED[v][n] + 2:
            off.append(f"n = {n}, {v}: {cs['not_converged']} nicht konvergiert statt {NOT_CONVERGED[v][n]} + 2")
        assert cs["iterations_max"] <= C.MAX_ITER[v]
        assert all(r["sdp"][v]["primal"] <= C.DEFAULT_TOL * 1.0001 and r["sdp"][v]["dual"] <= C.DEFAULT_TOL * 1.0001 for r in rows if r["sdp"][v]["converged"])
    _check(off)


@pytest.mark.parametrize("n", (6, 8))
def test_scale_rows_at_the_small_sizes(n):
    for v in C.VARIANTS:
        r = E.scale_row(n, v)
        assert (r["N"], r["r"]) == (n * n + 1, (n - 1) ** 2 + 1) and r["converged"] and r["bound"] <= r["ub"] + 1e-9 and r["glb"] < r["bound"]
    assert E.scale_row(n, "r3")["bound"] >= E.scale_row(n, "r3")["ub"] - 2.0
