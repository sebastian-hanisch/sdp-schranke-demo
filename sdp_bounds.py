"""Schranken der reinen Zuordnung (ohne Pads): Gilmore–Lawler (Stück 3), Dual Ascent (Stück 5), Abstiege aus Zufallsstarts (obere Schranke, Stück 3).

Kopie der Kerne `glb_bound.py` (Stück 3) und `rlt_dual.py` (Stück 5)."""

from dataclasses import dataclass

import numpy as np

import sdp_cost as K
from sdp_assignment import hungarian
from sdp_rng import SplitMix64


def flow_rows(F) -> np.ndarray:
    """Je Bauteil die Flüsse zu den anderen, aufsteigend sortiert (n × (n−1))."""
    F = np.asarray(F)
    n = len(F)
    return np.array([np.sort(np.delete(F[i], i)) for i in range(n)], dtype=np.int64).reshape(n, max(n - 1, 0))


def distance_rows(D) -> np.ndarray:
    """Je Platz die Abstände zu den anderen, absteigend sortiert (n × (n−1))."""
    D = np.asarray(D)
    n = len(D)
    return np.array([np.sort(np.delete(D[k], k))[::-1] for k in range(n)], dtype=np.int64).reshape(n, max(n - 1, 0))


def lower_matrix(F, D) -> np.ndarray:
    """L[i,k] (ganzzahlig): kleinstes Skalarprodukt der Flusszeile i mit der Abstandszeile k."""
    return flow_rows(F) @ distance_rows(D).T


@dataclass(frozen=True, eq=False)
class Bound:
    L: np.ndarray            # (n, n)
    assign: np.ndarray       # Platz je Bauteil (Zuordnung der linearen Aufgabe, eine kleinste)
    lap: int                 # Wert der linearen Zuordnung über L (ganzzahlig)
    rowmin2: int             # Σ_i min_k L[i,k] (ganzzahlig)
    rounded_cost: int        # echte Kosten der Anordnung `assign`

    @property
    def glb(self) -> float:
        """Die Gilmore–Lawler-Schranke (halbzahlig)."""
        return self.lap / 2

    @property
    def rowmin(self) -> float:
        """Die Schranke ohne Zuordnung."""
        return self.rowmin2 / 2


def gilmore_lawler(F, D) -> Bound:
    F, D = np.asarray(F), np.asarray(D)
    L = lower_matrix(F, D)
    assign, lap = hungarian(L)
    return Bound(L, assign, int(lap), int(L.min(axis=1).sum()), K.cost(assign.tolist(), F, D))


def costs_of_many(Q, F, D) -> np.ndarray:
    """Kosten aller Anordnungen in Q (Zeilen)."""
    F32, D32 = np.asarray(F).astype(np.int32), np.asarray(D).astype(np.int32)
    return ((F32[None] * D32[Q[:, :, None], Q[:, None, :]]).sum(axis=(1, 2)) // 2).astype(np.int64)


def random_starts(n: int, count: int, seed: int) -> np.ndarray:
    rng = SplitMix64(seed * 7919 + n * 13 + 5)
    out = np.empty((count, n), dtype=np.intp)
    for r in range(count):
        p = list(range(n))
        rng.shuffle(p)
        out[r] = p
    return out


def descents(F, D, starts) -> tuple:
    """Steilster Abstieg (kleinstes Δ, bei Gleichstand das erste Paar, solange Δ < 0) von jeder Startanordnung, alle Starts zugleich. Rückgabe: (Endanordnungen, Endkosten)."""
    F, D = np.asarray(F), np.asarray(D)
    cur = np.array(starts, dtype=np.intp).copy()
    n = cur.shape[1]
    cost = costs_of_many(cur, F, D)
    pairs = K.pair_list(n)
    pa = np.array([a for a, _ in pairs], dtype=np.intp)
    pb = np.array([b for _, b in pairs], dtype=np.intp)
    active = np.arange(len(cur))
    while len(active) and len(pairs):
        dl = K.all_deltas(cur[active], F, D)
        j = dl.argmin(axis=1)
        dm = dl[np.arange(len(active)), j]
        go = dm < 0
        idx = active[go]
        if not len(idx):
            break
        r = np.arange(len(idx))
        ta = cur[idx, pa[j[go]]].copy()
        cur[idx, pa[j[go]]] = cur[idx, pb[j[go]]]
        cur[idx, pb[j[go]]] = ta
        cost[idx] += dm[go]
        active = idx
    return cur, cost


def best_found(F, D, seed: int, count: int = 200) -> tuple:
    """(beste Kosten, Anordnung) der besten von `count` steilsten Abstiegen aus Zufallsstarts: eine **obere** Schranke des Optimums."""
    ends, cost = descents(F, D, random_starts(len(F), count, seed))
    i = int(cost.argmin())
    return int(cost[i]), ends[i]


# ----------------------------------------------------------------------------- Dual Ascent (Stück 5)


from sdp_assignment import hungarian_dual


@dataclass(frozen=True, eq=False)
class DualAscent:
    bounds: tuple            # Schranke nach jeder Runde (Runde 1 = Gilmore–Lawler)
    b: np.ndarray            # lineare Kosten nach der letzten Runde
    c: np.ndarray            # Restkosten c[i,j,k,l] ≥ 0 nach der letzten Runde

    @property
    def final(self) -> float:
        return self.bounds[-1]


def initial_costs(F, D) -> np.ndarray:
    """c[i,j,k,l] = ½ F[i,k] D[j,l] (Null für k = i oder l = j, weil die Diagonalen von F und D null sind)."""
    return 0.5 * np.einsum("ik,jl->ijkl", np.asarray(F, dtype=float), np.asarray(D, dtype=float))


def dual_ascent(F, D, rounds: int) -> DualAscent:
    """`rounds` Runden (mindestens eine); Rückgabe die Schranken je Runde und den Endzustand."""
    if rounds < 1:
        raise ValueError("mindestens eine Runde")
    n = len(F)
    c = initial_costs(F, D)
    b = np.zeros((n, n))
    bounds = []
    for _ in range(rounds):
        for i in range(n):
            ks = [k for k in range(n) if k != i]
            for j in range(n):
                ls = [l for l in range(n) if l != j]
                sub = c[i, j][np.ix_(ks, ls)]
                _, u, v, val = hungarian_dual(sub)
                c[i, j][np.ix_(ks, ls)] = np.maximum(sub - u[:, None] - v[None, :], 0.0)
                b[i, j] += val
        bounds.append(0.0 if n == 0 else float(hungarian(b)[1]))
        c = 0.5 * (c + c.transpose(2, 3, 0, 1))
    return DualAscent(tuple(bounds), b, c)


def invariant_gap(F, D, state_b, state_c, p) -> float:
    """Differenz zwischen den Kosten der Anordnung p und Σ_i b[i,p_i] + Σ_{i≠k} c[i,p_i,k,p_k]: bei gültigem Zustand null (nur für Tests)."""
    n = len(p)
    cost = 0.5 * sum(float(F[i, k]) * float(D[p[i], p[k]]) for i in range(n) for k in range(n) if i != k)
    rest = sum(float(state_b[i, p[i]]) for i in range(n)) + sum(float(state_c[i, p[i], k, p[k]]) for i in range(n) for k in range(n) if i != k)
    return cost - rest
