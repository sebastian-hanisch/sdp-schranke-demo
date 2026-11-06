"""Die semidefinite Relaxation des quadratischen Zuordnungsproblems (eigene Herleitung nach der Art von Zhao, Karisch, Rendl und Wolkowicz; die Quellen sind **nicht eingesehen**) und ein ADMM dafür.

Kosten wie in Stück 1 bis 8: cost(p) = Σ_{i<k} F[i,k] · D[p_i, p_k] = ½ xᵀ (D ⊗ F) x = <L, Y>, mit x = vec(X) (X Permutationsmatrix; Spaltenstapelung: Index 1 + j·n + i für Bauteil i auf Platz j)
und dem Lift Y = (1, x)(1, x)ᵀ der Größe N = n² + 1. Man lässt Rang eins fallen und verlangt Y ⪰ 0 und die linearen Eigenschaften der Permutationspunkte:
    Y00 = 1;   Pfeil: Y_{0a} = Y_{aa};   Gangster: Y_{ab} = 0, wenn die Paare a = (i, j) und b = (k, l) sich in genau einer Koordinate gleichen.
**R2** hat genau diese Bedingungen, **R3** verlangt zusätzlich Y ≥ 0 (alle Einträge außerhalb des Gangster-Musters).
**Facial Reduction:** Y = V̂ R V̂ᵀ mit V̂ = [[1, 0], [e/n, V⊗V]] (V: n×(n−1), orthonormale Basis von e^⊥) erzwingt X e = e und Xᵀ e = e; mit Ṽ = V̂ · diag(1/√2, 1, …, 1) (orthonormale Spalten, r = (n−1)² + 1)
ist K = {Ṽ R̃ Ṽᵀ : R̃ ⪰ 0} der Kegel und P_K(Y) = Ṽ P_PSD(ṼᵀYṼ) Ṽᵀ.
**ADMM** (eigene Fassung, im Raum von Y):  Y⁺ = P_A(Z − U − L/ρ),  Z⁺ = P_K(Y⁺ + U),  U⁺ = U + Y⁺ − Z⁺.  P_A (die Gleichungen, bei R3 mit Klemmen Y ≥ 0) ist einträgeweise; ρ wird alle `every` Schritte verdoppelt oder
halbiert (U mit umgerechnet), wenn ein Residuum zehnmal größer ist als das andere. Abbruch, wenn Primal- und Dualresiduum höchstens `tol` sind (L auf den größten Eintrag 1 normiert).
**Sichere untere Schranke** aus der Dualmatrix M = −ρU ∈ K*: y aus den Pfeil- und Y00-Einträgen von L − M (kleinste Quadrate), T = L − 𝒜ᵀy (bei R3 mit − N, N = max(0, L − 𝒜ᵀy − M) ≥ 0),
Schranke a + (n+1) · min(0, λ_min(Ṽᵀ T Ṽ)). Sie gilt für **jede** Dualmatrix (also bei jeder Toleranz), weil jedes zulässige Y die Spur n + 1 hat."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import sdp_assignment as A


def orthonormal_complement(n: int) -> np.ndarray:
    """n × (n−1) mit orthonormalen Spalten, senkrecht zu e."""
    Q, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), np.eye(n)[:, : n - 1]]))
    return Q[:, 1:n]


class Relaxation:
    """Die Daten der Relaxation für Flüsse `F` und Abstände `D`; `nonneg=True` ist die Stufe R3."""

    def __init__(self, F, D, nonneg: bool = False):
        F, D = np.asarray(F, dtype=float), np.asarray(D, dtype=float)
        self.nonneg = bool(nonneg)
        self.n = n = len(F)
        self.N = N = n * n + 1
        self.L = np.zeros((N, N))
        self.L[1:, 1:] = 0.5 * np.kron(D, F)
        V = orthonormal_complement(n)
        Vh = np.zeros((N, (n - 1) ** 2 + 1))
        Vh[0, 0] = 1.0
        Vh[1:, 0] = 1.0 / n
        Vh[1:, 1:] = np.kron(V, V)
        self.Vhat = Vh
        self.Vt = Vh * np.r_[1.0 / np.sqrt(2.0), np.ones((n - 1) ** 2)]
        j, i = np.divmod(np.arange(n * n), n)                      # Index a = j·n + i: Bauteil i = a mod n, Platz j = a div n
        g = np.zeros((N, N), dtype=bool)
        g[1:, 1:] = (i[:, None] == i[None, :]) ^ (j[:, None] == j[None, :])
        self.gang = g

    @property
    def r(self) -> int:
        return self.Vt.shape[1]

    @property
    def gangster_pairs(self) -> int:
        """Zahl der Gangster-Paare (Einträge oberhalb der Diagonale): n²(n−1)."""
        return int(np.triu(self.gang, 1).sum())

    def permutation_point(self, perm) -> np.ndarray:
        """Der zulässige Punkt Y* = (1, x)(1, x)ᵀ zur Anordnung `perm` (Platz je Bauteil)."""
        n = self.n
        X = np.zeros((n, n))
        for i in range(n):
            X[i, int(perm[i])] = 1.0
        y = np.r_[1.0, X.T.reshape(-1)]
        return np.outer(y, y)

    def cost_of(self, Y) -> float:
        return float(np.sum(self.L * Y))

    def proj_A(self, W: np.ndarray) -> np.ndarray:
        """Projektion auf die affine Menge (Y00 = 1, Pfeil, Gangster) und bei R3 auf Y ≥ 0; einträgeweise."""
        W = 0.5 * (W + W.T)
        W[self.gang] = 0.0
        W[0, 0] = 1.0
        idx = np.arange(1, self.N)
        t = (2.0 * W[0, idx] + W[idx, idx]) / 3.0                  # Projektion von (a, d) mit den Gewichten (2, 1) auf a = d
        W[0, idx] = t
        W[idx, 0] = t
        W[idx, idx] = t
        if self.nonneg:
            np.maximum(W, 0.0, out=W)
        return W

    def proj_K(self, Y: np.ndarray) -> np.ndarray:
        """Projektion auf K = {Ṽ R̃ Ṽᵀ : R̃ ⪰ 0}."""
        Rt = self.Vt.T @ Y @ self.Vt
        w, U = np.linalg.eigh(0.5 * (Rt + Rt.T))
        w = np.maximum(w, 0.0)
        return self.Vt @ ((U * w) @ U.T) @ self.Vt.T

    def safe_bound(self, M: np.ndarray) -> float:
        """Untere Schranke der Kosten aller Anordnungen aus der Dualmatrix `M` (symmetrisch; siehe Kopf)."""
        n, N = self.n, self.N
        idx = np.arange(1, N)
        T0 = self.L - M
        a = T0[0, 0]
        yk = 0.5 * T0[idx, idx] - T0[0, idx]
        A_y = np.zeros((N, N))
        A_y[0, 0] = a
        A_y[0, idx] = -0.5 * yk
        A_y[idx, 0] = -0.5 * yk
        A_y[idx, idx] = yk
        T = self.L - A_y
        if self.nonneg:
            T = M + np.minimum(T - M, 0.0)
        T[self.gang] = M[self.gang]
        Rt = self.Vt.T @ T @ self.Vt
        lam = np.linalg.eigvalsh(0.5 * (Rt + Rt.T))[0]
        return float(a + (n + 1) * min(0.0, lam))


@dataclass(frozen=True, eq=False)
class Result:
    variant: str
    iterations: int
    converged: bool
    value: float                 # <L, Z> am Ende (bei grober Toleranz keine Schranke)
    bound: float                 # sichere untere Schranke
    primal: float
    dual: float
    rho: float                   # ρ am Ende
    rho_changes: int
    seconds: float
    Z: np.ndarray
    history: dict                # je Schritt: "primal", "dual", "value", "rho" (Listen)


def admm(rel: Relaxation, rho: float = 1.0, tol: float = 1e-5, max_iter: int = 6000, adaptive: bool = True, every: int = 100) -> Result:
    """ADMM für die Relaxation `rel`; Start Z = e0 e0ᵀ, U = 0."""
    if rho <= 0:
        raise ValueError("rho muss positiv sein")
    scale = float(np.abs(rel.L).max()) or 1.0
    L = rel.L / scale
    N = rel.N
    Z = np.zeros((N, N))
    Z[0, 0] = 1.0
    U = np.zeros((N, N))
    hist = {"primal": [], "dual": [], "value": [], "rho": []}
    changes = 0
    t0 = time.perf_counter()
    r = d = float("inf")
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Y = rel.proj_A(Z - U - L / rho)
        Zn = rel.proj_K(Y + U)
        U = U + Y - Zn
        r = float(np.linalg.norm(Y - Zn))
        d = float(rho * np.linalg.norm(Zn - Z))
        Z = Zn
        hist["primal"].append(r)
        hist["dual"].append(d)
        hist["value"].append(float(np.sum(L * Y)) * scale)
        hist["rho"].append(rho)
        if r <= tol and d <= tol:
            converged = True
            break
        if adaptive and it % every == 0:
            if r > 10 * d:
                rho *= 2.0
                U /= 2.0
                changes += 1
            elif d > 10 * r:
                rho /= 2.0
                U *= 2.0
                changes += 1
    M = -rho * U * scale                                           # U liegt im Normalenkegel von K: M ∈ K*
    return Result("r3" if rel.nonneg else "r2", it, converged, float(np.sum(L * Z)) * scale, rel.safe_bound(M), r, d, rho, changes, time.perf_counter() - t0, Z, hist)


def rounded_arrangement(rel: Relaxation, Z: np.ndarray) -> list:
    """Die Anordnung, die die Zeile 0 von Z (x ∈ [0,1]) durch lineare Zuordnung rundet (größte Summe der x_ij)."""
    n = rel.n
    X = Z[0, 1:].reshape(n, n).T                                   # X[i, j] = x[j·n + i]
    assign, _ = A.hungarian(-X)
    return [int(j) for j in assign]
