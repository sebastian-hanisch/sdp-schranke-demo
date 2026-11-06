"""Linearisierung (RLT1, Adams–Johnson) des quadratischen Zuordnungsproblems und ihr LP.

Kosten wie in Stück 1 bis 4: cost(p) = ½ Σ_{i≠k} F[i,k] · D[p_i, p_k] (alle geordneten Paare).
Variablen: x[i,j] (Bauteil i auf Platz j) und y[i,j,k,l] für i ≠ k, j ≠ l (gemeint: y = x[i,j] · x[k,l]).
Ziel: Σ ½ F[i,k] D[j,l] · y[i,j,k,l]. Nebenbedingungen (alle als Gleichung):
  Σ_j x[i,j] = 1 und Σ_i x[i,j] = 1                                              (Zuordnung)
  Σ_{k≠i} y[i,j,k,l] = x[i,j]   für alle i, j und l ≠ j                         (Spaltenbedingung mit x[i,j] multipliziert; der Summand k = i fällt weg, weil x[i,j] · x[i,l] = 0)
  Σ_{l≠j} y[i,j,k,l] = x[i,j]   für alle i, j und k ≠ i                         (Zeilenbedingung mit x[i,j] multipliziert)
  y[i,j,k,l] = y[k,l,i,j]                                                        (Symmetrie)
  x, y ≥ 0.
Jede Anordnung p ergibt mit x = Permutationsmatrix, y = x·x einen zulässigen Punkt mit Zielwert cost(p): der LP-Wert ist eine untere Schranke. Zahl der Variablen n² + n²(n−1)², der Gleichungen 2n + 2n²(n−1) + n²(n−1)²/2."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def sizes(n: int) -> tuple:
    """(Zahl der Variablen, Zahl der Gleichungen) des LP für n Bauteile."""
    return n * n + n * n * (n - 1) ** 2, 2 * n + 2 * n * n * (n - 1) + n * n * (n - 1) ** 2 // 2


@dataclass(frozen=True, eq=False)
class LP:
    n: int
    c: np.ndarray           # Zielvektor
    A: object               # dünn besetzte Matrix der Gleichungen
    rhs: np.ndarray
    y_index: np.ndarray     # y_index[i,j,k,l] = Spaltennummer der Variablen y[i,j,k,l] (−1 für k = i oder l = j)

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_eq(self) -> int:
        return self.A.shape[0]


def build_lp(F, D) -> LP:
    F, D = np.asarray(F, dtype=float), np.asarray(D, dtype=float)
    n = len(F)
    if n < 2:
        raise ValueError("mindestens zwei Bauteile")
    y_index = -np.ones((n, n, n, n), dtype=np.intp)
    nxt = n * n
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k == i:
                    continue
                for l in range(n):
                    if l != j:
                        y_index[i, j, k, l] = nxt
                        nxt += 1
    c = np.zeros(nxt)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k != i:
                    for l in range(n):
                        if l != j:
                            c[y_index[i, j, k, l]] = 0.5 * F[i, k] * D[j, l]
    rows, cols, vals, rhs = [], [], [], []

    def add(entries, b):
        r = len(rhs)
        for col, v in entries:
            rows.append(r)
            cols.append(col)
            vals.append(v)
        rhs.append(b)
    for i in range(n):
        add([(i * n + j, 1.0) for j in range(n)], 1.0)
    for j in range(n):
        add([(i * n + j, 1.0) for i in range(n)], 1.0)
    for i in range(n):
        for j in range(n):
            for l in range(n):
                if l != j:
                    add([(int(y_index[i, j, k, l]), 1.0) for k in range(n) if k != i] + [(i * n + j, -1.0)], 0.0)
            for k in range(n):
                if k != i:
                    add([(int(y_index[i, j, k, l]), 1.0) for l in range(n) if l != j] + [(i * n + j, -1.0)], 0.0)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k == i:
                    continue
                for l in range(n):
                    if l != j and (i, j) < (k, l):
                        add([(int(y_index[i, j, k, l]), 1.0), (int(y_index[k, l, i, j]), -1.0)], 0.0)
    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), nxt)).tocsr()
    return LP(n, c, A, np.array(rhs), y_index)


def permutation_point(lp: LP, p) -> np.ndarray:
    """Der zulässige Punkt zu einer Anordnung: x = Permutationsmatrix, y = x[i,j] · x[k,l]."""
    n = lp.n
    if sorted(int(q) for q in p) != list(range(n)):
        raise ValueError("p muss eine Permutation von 0..n-1 sein")
    z = np.zeros(lp.n_vars)
    for i in range(n):
        z[i * n + int(p[i])] = 1.0
    for i in range(n):
        for k in range(n):
            if k != i:
                z[lp.y_index[i, int(p[i]), k, int(p[k])]] = 1.0
    return z


@dataclass(frozen=True, eq=False)
class LPResult:
    value: float
    x: np.ndarray           # Teil x[i,j] der Lösung (n × n), oft gebrochen
    n_vars: int
    n_eq: int

    @property
    def integral(self) -> bool:
        """Ist x eine Permutationsmatrix? Dann ist die LP-Lösung eine Anordnung und der LP-Wert das Optimum."""
        return bool(np.all(np.minimum(np.abs(self.x), np.abs(self.x - 1.0)) < 1e-6))


def solve_lp(F, D) -> LPResult:
    """Löst das LP mit HiGHS (scipy). Wirft einen Fehler, wenn der Löser nicht optimal endet."""
    lp = build_lp(F, D)
    res = linprog(lp.c, A_eq=lp.A, b_eq=lp.rhs, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP nicht gelöst: {res.message}")
    n = lp.n
    return LPResult(float(res.fun), np.asarray(res.x[: n * n]).reshape(n, n), lp.n_vars, lp.n_eq)
