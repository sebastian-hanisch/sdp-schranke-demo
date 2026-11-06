"""Branch and Bound (Kopie aus Stück 8, nur die Arten ohne / linear / Gilmore–Lawler) gegen die Aufzählung (ORACLE-PLAYBOOK §5) und das Handbeispiel."""

import math

import numpy as np
import pytest

import sdp_constants as C
import sdp_landscape as L
import sdp_tree as T
from sdp_instance import make_instance


def optimum(inst):
    P = L.all_perms(inst.n)
    return int(L.costs_of(P, inst.F, inst.D).min())


def test_hand_example_node_counts_with_the_optimum_as_the_start_value(hand_gap):
    """Wurzel 5, Optimum 6: mit dem Anfangswert 6 bewertet der Baum 4 (Gilmore–Lawler), 16 (linear) und 64 Knoten (ohne Schranke)."""
    r = {art: T.solve(hand_gap.F, hand_gap.D, art, incumbent=6) for art in ("glb", "lin", "none")}
    assert {a: t.nodes for a, t in r.items()} == {"glb": 4, "lin": 16, "none": 64} and r["glb"].root == 5.0 and all(t.best == 6.0 and not t.capped for t in r.values())
    assert T.tree_size(4) == 64 == r["none"].nodes


def test_hand_example_without_a_start_value(hand_gap):
    assert {a: T.solve(hand_gap.F, hand_gap.D, a).nodes for a in ("glb", "lin", "none")} == {"glb": 10, "lin": 19, "none": 64}
    assert all(T.solve(hand_gap.F, hand_gap.D, a).best == 6.0 for a in ("glb", "lin", "none"))


@pytest.mark.parametrize("family", C.FAMILIES)
@pytest.mark.parametrize("art", T.ARTS)
def test_the_tree_proves_the_enumerated_optimum(family, art):
    inst = make_instance(family, 6, 100005)
    opt = optimum(inst)
    tr = T.solve(inst.F, inst.D, art)
    assert tr.best == opt and not tr.capped and tr.optimal
    if tr.perm is not None:
        assert int(L.costs_of(np.array([tr.perm]), inst.F, inst.D)[0]) == opt
    tr2 = T.solve(inst.F, inst.D, art, incumbent=opt + 1)
    assert tr2.best == opt and tr2.nodes <= tr.nodes + T.tree_size(6)


@pytest.mark.parametrize("art", ("glb", "lin"))
def test_a_start_value_of_the_optimum_never_costs_more_nodes(art):
    inst = make_instance("cluster", 8, 100006)
    opt = optimum(inst)
    assert T.solve(inst.F, inst.D, art, incumbent=opt).nodes <= T.solve(inst.F, inst.D, art).nodes


def test_the_node_cap_stops_the_search_and_says_so():
    inst = make_instance("uniform", 8, 100007)
    tr = T.solve(inst.F, inst.D, "glb", cap=5)
    assert tr.capped and tr.nodes <= 5 and not tr.optimal


def test_tree_size_and_pruning_rule():
    assert T.tree_size(1) == 1 and T.tree_size(4) == 64 and T.tree_size(5) == math.floor(math.e * 120) - 1
    assert T.prunes(5.2, 6) and not T.prunes(4.9, 6) and T.prunes(6.0, 6) and not T.prunes(5.0, 6)


def test_unknown_bound_kind_is_refused():
    with pytest.raises(ValueError):
        T.node_bound("pb", np.zeros((3, 3)), np.zeros((3, 3)), [], [], [0, 1, 2], [0, 1, 2], 0)
