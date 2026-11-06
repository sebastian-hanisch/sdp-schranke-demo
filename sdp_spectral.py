"""Eigenwertschranke (EB) und projizierte Eigenwertschranke (PB, vereinfachte Fassung) des quadratischen Zuordnungsproblems in der Koopmans–Beckmann-Form.

Kosten wie in Stück 1 bis 3: cost(p) = Σ_{i<k} F[i,k] · D[p_i, p_k] = ½ · tr(F X D Xᵀ) mit der Permutationsmatrix X[i, p_i] = 1 (F und D symmetrisch, Diagonale 0).

EB: Permutationsmatrizen sind orthogonal; für orthogonales X gilt tr(F X D Xᵀ) ≥ Σ_i λ_i(F) · λ_{n+1−i}(D) (Eigenwerte von F aufsteigend, von D absteigend gepaart). EB = ½ dieser Summe.

PB: Jede Permutationsmatrix ist X = e eᵀ/n + V Y Vᵀ mit V (n × (n−1), orthonormale Spalten, orthogonal zum Einsvektor e) und orthogonalem Y. Mit u = F e, w = D e, s_F = eᵀ F e, s_D = eᵀ D e, F̂ = Vᵀ F V, D̂ = Vᵀ D V gilt **exakt**
  tr(F X D Xᵀ) = − s_F s_D / n² + (2/n) · Σ_i u_i · w_{p_i} + tr(F̂ Y D̂ Yᵀ).
Der mittlere Term ist linear in X (kleinster Wert: u aufsteigend gegen w absteigend sortiert), der letzte wird wie bei der EB durch die gepaarten Eigenwerte von F̂ und D̂ abgeschätzt. PB = ½ der Summe der drei Mindestwerte.
Das ist eine eigene, vereinfachte Herleitung der Idee der projizierten Eigenwertschranke (Hadley, Rendl, Wolkowicz), nicht ihre Originalform."""

from dataclasses import dataclass

import numpy as np


def sorted_dot(a, b) -> float:
    """Kleinstes Skalarprodukt zweier Vektoren: a aufsteigend, b absteigend sortiert (Umordnungsungleichung)."""
    return float(np.dot(np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))[::-1]))


def eigenvalues(M) -> np.ndarray:
    """Eigenwerte einer symmetrischen Matrix, aufsteigend."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("quadratische Matrix erwartet")
    if not np.allclose(M, M.T):
        raise ValueError("symmetrische Matrix erwartet")
    return np.linalg.eigvalsh(M)


def complement_basis(n: int) -> np.ndarray:
    """n × (n−1)-Matrix mit orthonormalen Spalten, die alle senkrecht zum Einsvektor stehen (aus einer QR-Zerlegung von [e | e_1 … e_{n−1}])."""
    if n < 2:
        raise ValueError("mindestens zwei Bauteile")
    q, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), np.eye(n)[:, : n - 1]]))
    return q[:, 1:]


@dataclass(frozen=True, eq=False)
class Spectral:
    sdp_F: np.ndarray        # Eigenwerte von F, aufsteigend
    sdp_D: np.ndarray        # Eigenwerte von D, aufsteigend
    sdp_Fp: np.ndarray       # Eigenwerte von F̂ = Vᵀ F V (n−1 Stück), aufsteigend
    sdp_Dp: np.ndarray       # Eigenwerte von D̂
    eb: float                # Eigenwertschranke
    const: float             # ½ · (−s_F s_D / n²)
    linear: float            # ½ · (2/n) · kleinstes Skalarprodukt von u und w
    quad: float              # ½ · gepaarte Eigenwerte von F̂ und D̂

    @property
    def pb(self) -> float:
        """Projizierte Eigenwertschranke (vereinfachte Fassung): die Summe der drei Terme."""
        return self.const + self.linear + self.quad

    @property
    def products(self) -> np.ndarray:
        """Die Summanden der EB: ½ · λ_i(F) · λ_{n+1−i}(D) (F aufsteigend, D absteigend gepaart)."""
        return 0.5 * self.sdp_F * self.sdp_D[::-1]


def spectral_bounds(F, D) -> Spectral:
    F, D = np.asarray(F, dtype=float), np.asarray(D, dtype=float)
    n = len(F)
    lf, ld = eigenvalues(F), eigenvalues(D)
    V = complement_basis(n)
    e = np.ones(n)
    u, w = F @ e, D @ e
    s_f, s_d = float(e @ u), float(e @ w)
    lfp, ldp = eigenvalues(V.T @ F @ V), eigenvalues(V.T @ D @ V)
    return Spectral(lf, ld, lfp, ldp, 0.5 * sorted_dot(lf, ld), 0.5 * (-s_f * s_d / n**2), 0.5 * (2.0 / n) * sorted_dot(u, w), 0.5 * sorted_dot(lfp, ldp))


def permutation_matrix(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.intp)
    n = len(p)
    if sorted(p.tolist()) != list(range(n)):
        raise ValueError("p muss eine Permutation von 0..n-1 sein")
    X = np.zeros((n, n))
    X[np.arange(n), p] = 1.0
    return X


def decomposition_terms(F, D, p) -> tuple:
    """(links, rechts): tr(F X D Xᵀ) und die Summe der drei Terme der Zerlegung für die Permutation p; beide müssen übereinstimmen."""
    F, D = np.asarray(F, dtype=float), np.asarray(D, dtype=float)
    n = len(F)
    X = permutation_matrix(p)
    V = complement_basis(n)
    e = np.ones(n)
    u, w = F @ e, D @ e
    Y = V.T @ X @ V
    left = float(np.trace(F @ X @ D @ X.T))
    right = float(-(e @ u) * (e @ w) / n**2 + (2.0 / n) * (u @ X @ w) + np.trace((V.T @ F @ V) @ Y @ (V.T @ D @ V) @ Y.T))
    return left, right
