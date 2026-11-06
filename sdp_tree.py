"""Branch and Bound für das quadratische Zuordnungsproblem (reine Zuordnung, ohne Pads): Knotenschranken für Teilzuordnungen und die Tiefensuche.

Kopie des Kerns `sib_tree.py` (Stück 8), gekürzt auf die Arten `none`, `lin` (lineare Zuordnung) und `glb` (Gilmore–Lawler); die projizierte Schranke, das Maximum und Dual Ascent der Knoten gehören zu Stück 8 und werden hier nicht gebraucht.
Kosten: cost(p) = Σ_{i<k} F[i,k] · D[p_i, p_k] (jedes Paar einmal; F und D symmetrisch, Diagonale 0).
Knoten = Teilzuordnung: die Bauteile `fac` stehen auf den Plätzen `loc`, die freien Bauteile und Plätze (je r Stück) sind offen. Jede Fortsetzung kostet mindestens die festen Kosten plus die kleinste lineare Zuordnung über die Wechselwirkung mit den gesetzten Bauteilen (`lin`),
`glb` addiert die Gilmore–Lawler-Schranke der Restaufgabe. Beschnitten wird, wenn die aufgerundete Schranke nicht unter dem besten Wert liegt (ganzzahlige Kosten). Ganzzahlige Arten rechnen exakt in doppelten Kosten: die Knotenzahlen sind auf jedem Rechner gleich."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

import sdp_assignment as A

EPS = 1e-7
ARTS = ("none", "lin", "glb")
ART_NAMES = {"none": "ohne Schranke (feste Kosten)", "lin": "lineare Zuordnung", "glb": "Gilmore–Lawler"}
ART_SHORT = {"none": "ohne", "lin": "linear", "glb": "Gilmore–Lawler"}
ORDERS = ("flow", "index", "light")
ORDER_NAMES = {"flow": "starke Flüsse zuerst", "index": "nach Nummer", "light": "schwache Flüsse zuerst"}


def tree_size(n: int) -> int:
    """Zahl der Knoten des vollen Baums ohne Beschneiden: Σ_{k=1..n} n!/(n−k)! (= ⌊e·n!⌋ − 1)."""
    return sum(math.perm(n, k) for k in range(1, n + 1))


def branch_order(F, kind: str) -> list:
    """Reihenfolge, in der die Bauteile gesetzt werden: nach Gesamtfluss absteigend, nach Nummer oder nach Gesamtfluss aufsteigend (stabil sortiert)."""
    F = np.asarray(F)
    n = len(F)
    tot = F.sum(axis=1)
    if kind == "flow":
        return [int(i) for i in np.argsort(-tot, kind="stable")]
    if kind == "index":
        return list(range(n))
    if kind == "light":
        return [int(i) for i in np.argsort(tot, kind="stable")]
    raise ValueError(f"unbekannte Reihenfolge {kind!r}")


def _sorted_rows_asc(M) -> np.ndarray:
    """Je Zeile die Einträge ohne die Diagonale, aufsteigend (r × (r−1))."""
    r = len(M)
    X = np.asarray(M, dtype=np.int64).copy()
    np.fill_diagonal(X, np.iinfo(np.int64).max // 4)
    return np.sort(X, axis=1)[:, : r - 1]


def _sorted_rows_desc(M) -> np.ndarray:
    r = len(M)
    X = np.asarray(M, dtype=np.int64).copy()
    np.fill_diagonal(X, -(np.iinfo(np.int64).max // 4))
    return -np.sort(-X, axis=1)[:, : r - 1]


def linear_costs(F, D, fac, loc, free_f, free_l) -> np.ndarray:
    """c[i,j] für freie Bauteile (Zeilen) und freie Plätze (Spalten): Wechselwirkung mit den gesetzten Bauteilen."""
    r = len(free_f)
    if not len(fac):
        return np.zeros((r, r), dtype=np.int64)
    return np.asarray(F)[np.ix_(free_f, fac)].astype(np.int64) @ np.asarray(D)[np.ix_(loc, free_l)].astype(np.int64)


def _lap(M) -> float:
    return A.hungarian(M)[1]


def node_bound(art: str, F, D, fac, loc, free_f, free_l, fixed) -> float:
    """Untere Schranke der Kosten aller Fortsetzungen der Teilzuordnung (Bauteil fac[a] auf Platz loc[a]); `fixed` = Kosten der gesetzten Paare."""
    if art not in ARTS:
        raise ValueError(f"unbekannte Art {art!r}")
    F, D = np.asarray(F), np.asarray(D)
    r = len(free_f)
    if r == 0 or art == "none":
        return float(fixed)
    c = linear_costs(F, D, fac, loc, free_f, free_l)
    if art == "lin" or r == 1:
        return fixed + _lap(c)
    Ff, Dd = F[np.ix_(free_f, free_f)], D[np.ix_(free_l, free_l)]
    out = -float("inf")
    if art == "glb":
        m2 = _sorted_rows_asc(Ff) @ _sorted_rows_desc(Dd).T            # doppelte Skalarprodukte (jedes Paar zweimal gezählt)
        out = max(out, fixed + _lap(2 * c + m2) / 2)
    return float(out)


def prunes(bound: float, best: float) -> bool:
    """Beschneiden, wenn keine Fortsetzung billiger als `best` sein kann (ganzzahlige Kosten)."""
    return math.ceil(bound - EPS) >= best


@dataclass(frozen=True, eq=False)
class Tree:
    art: str
    order: str
    n: int
    nodes: int                  # bewertete Kinder (ohne die Wurzel)
    best: float                 # beste gefundene Kosten (Anfangswert, falls nichts Besseres gefunden)
    perm: list | None           # Anordnung zu `best`, falls im Baum gefunden (None: der Anfangswert blieb)
    capped: bool                # Obergrenze der Knoten erreicht: nicht zu Ende gerechnet
    seconds: float
    depth: tuple                # bewertete Kinder je Tiefe (Tiefe 0 = Kinder der Wurzel)
    pruned: tuple               # davon beschnitten, je Tiefe
    root: float                 # Schranke der Wurzel (Teilzuordnung leer)
    incumbent: float | None     # Anfangswert (None: keiner)

    @property
    def optimal(self) -> bool:
        """Zu Ende gerechnet: `best` ist das Optimum (bewiesen)."""
        return not self.capped

    @property
    def share(self) -> float:
        """Anteil der Knoten am vollen Baum."""
        return self.nodes / tree_size(self.n)


def solve(F, D, art: str = "glb", incumbent=None, order: str = "flow", cap: int | None = None) -> Tree:
    """Tiefensuche über Teilzuordnungen. `incumbent`: Kosten einer bekannten Anordnung (None: unendlich). `cap`: höchstens so viele Knoten (sonst zu Ende gerechnet)."""
    F, D = np.asarray(F), np.asarray(D)
    n = len(F)
    seq = branch_order(F, order)
    best = float("inf") if incumbent is None else float(incumbent)
    found = {"perm": None}
    state = {"nodes": 0, "capped": False}
    depth = [0] * n
    pruned = [0] * n
    assign = [-1] * n
    start = time.perf_counter()
    root = node_bound(art, F, D, [], [], list(seq), list(range(n)), 0)

    def rec(k, fixed, fac, loc, free_l):
        nonlocal best
        if k == n:
            if fixed < best:
                best = float(fixed)
                found["perm"] = assign[:]
            return
        i = seq[k]
        free_f = seq[k + 1:]
        kids = []
        for j in free_l:
            if cap is not None and state["nodes"] >= cap:
                state["capped"] = True
                return
            state["nodes"] += 1
            depth[k] += 1
            fx = fixed + sum(int(F[i, a]) * int(D[j, l]) for a, l in zip(fac, loc))
            rest = [x for x in free_l if x != j]
            b = node_bound(art, F, D, fac + [i], loc + [j], free_f, rest, fx)
            if prunes(b, best):
                pruned[k] += 1
            else:
                kids.append((b, j, fx, rest))
        kids.sort(key=lambda t: (t[0], t[1]))
        for b, j, fx, rest in kids:
            if prunes(b, best):
                pruned[k] += 1                       # erst später beschnitten: ein besserer Wert wurde inzwischen gefunden
                continue
            assign[i] = j
            rec(k + 1, fx, fac + [i], loc + [j], rest)
            assign[i] = -1
            if state["capped"]:
                return

    rec(0, 0, [], [], list(range(n)))
    return Tree(art, order, n, state["nodes"], best, found["perm"], state["capped"], time.perf_counter() - start, tuple(depth), tuple(pruned), float(root), incumbent)
