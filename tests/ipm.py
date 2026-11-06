"""Test-Orakel: ein dichter Innere-Punkte-Löser (primale Barrierenmethode) für dieselbe SDP wie der ADMM, als **zweiter, unabhängiger Algorithmus** (nur Tests, n ≤ 6; nicht Teil der App).

Die Gleichungen aus Y00, Pfeil und Gangster werden durch eine Nullraum-Parametrisierung eliminiert: R̃(w) = R̃0 + smat(N w) mit dem Mittel aller Permutationspunkte als streng zulässigem Start;
bei R3 kommt die Barriere −Σ log Y_e(w) für Y ≥ 0 hinzu. Newton mit Rückverfolgung auf dem t-Pfad, Abbruch bei Lückenschranke ν/t unter `tol`."""

import itertools
import time

import numpy as np
from scipy.linalg import null_space



def build(q, perms):
    r = q.Vt.shape[1]
    N, n = q.N, q.n
    iu = np.triu_indices(r)
    m = len(iu[0])
    wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))          # svec mit Faktor √2 auf den Außerdiagonalen: <A,B> = svec(A)·svec(B)
    def svec(A):
        return A[iu] * wgt
    def smat(v):
        A = np.zeros((r, r))
        A[iu] = v / wgt
        A = A + A.T - np.diag(np.diag(A))
        return A
    # Zeilen der Gleichungen: <ṼᵀEṼ, R̃> = b
    rows, rhs = [], []
    def add(E, b):
        E = 0.5 * (E + E.T)
        rows.append(svec(q.Vt.T @ E @ q.Vt))
        rhs.append(b)
    E = np.zeros((N, N)); E[0, 0] = 1.0; add(E, 1.0)
    for k in range(1, N):
        E = np.zeros((N, N)); E[0, k] = 1.0; E[k, k] = -1.0; add(E, 0.0)                # Y_0k − Y_kk = 0 (symmetrisiert, gleiche Nullstelle)
    gi, gj = np.nonzero(np.triu(q.gang, 1))
    for a, b_ in zip(gi, gj):
        E = np.zeros((N, N)); E[a, b_] = 1.0; add(E, 0.0)
    B = np.array(rows)
    rhs = np.array(rhs)
    # Startpunkt: Mittel der Permutationspunkte
    Y0 = np.zeros((N, N))
    for p in perms:
        X = np.zeros((n, n))
        for i in range(n):
            X[i, p[i]] = 1.0
        y = np.r_[1.0, X.T.reshape(-1)]
        Y0 += np.outer(y, y)
    Y0 /= len(perms)
    R0 = q.Vt.T @ Y0 @ q.Vt
    v0 = svec(R0)
    Nmat = null_space(B)
    return {"r": r, "m": m, "svec": svec, "smat": smat, "B": B, "rhs": rhs, "v0": v0, "Nmat": Nmat, "R0": R0, "res0": float(np.abs(B @ v0 - rhs).max())}


def solve(q, perms, nonneg: bool = False, tol: float = 1e-6, mu: float = 6.0, max_newton: int = 400):
    t0 = time.perf_counter()
    d = build(q, perms)
    r, Nmat, v0, smat, svec = d["r"], d["Nmat"], d["v0"], d["smat"], d["svec"]
    k = Nmat.shape[1]
    Lmat = svec(q.Vt.T @ q.L @ q.Vt)                                        # <L, Y> = <ṼᵀLṼ, R̃> (Y = Ṽ R̃ Ṽᵀ, Ṽ orthonormal)
    g = Nmat.T @ Lmat
    # Basis-Matrizen Q_i = smat(N_i)
    Q = np.array([smat(Nmat[:, i]) for i in range(k)])
    w = np.zeros(k)
    # Nicht-Negativität: Y_e(w) = <E_e, Ṽ R̃ Ṽᵀ> ≥ 0 für alle Nicht-Gangster-Einträge e = (a ≤ b)
    if nonneg:
        nong = np.triu(~q.gang, 0)
        ea, eb = np.nonzero(nong)
        Vt = q.Vt
        coef = np.einsum("ei,ej->eij", Vt[ea], Vt[eb])                       # Y_ab = Σ_ij Vt[a,i] R̃_ij Vt[b,j]
        coef = 0.5 * (coef + coef.transpose(0, 2, 1))
        Ce = np.array([svec(c) for c in coef])                               # (E, m)
        Ae = Ce @ Nmat                                                       # (E, k): Y_e(w) = Ce·v0 + Ae w
        y0 = Ce @ v0
    t = 1.0
    nu = r + (len(y0) if nonneg else 0)
    newton_total = 0

    Rw = lambda w: smat(v0 + Nmat @ w)
    def phi(w, t):
        R = Rw(w)
        try:
            Lc = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            return np.inf, None
        val = t * float(g @ w) - 2.0 * np.sum(np.log(np.diag(Lc)))
        if nonneg:
            ye = y0 + Ae @ w
            if ye.min() <= 0:
                return np.inf, None
            val -= np.sum(np.log(ye))
        return val, Lc
    while True:
        for _ in range(60):
            R = Rw(w)
            Ri = np.linalg.inv(R)
            P = np.einsum("ab,ibc->iac", Ri, Q)
            grad = t * g - np.einsum("iaa->i", P)
            H = P.reshape(k, -1) @ P.transpose(0, 2, 1).reshape(k, -1).T            # tr(P_i P_j)
            if nonneg:
                ye = y0 + Ae @ w
                grad -= Ae.T @ (1.0 / ye)
                H += (Ae / ye[:, None] ** 2).T @ Ae
            dw = -np.linalg.solve(H, grad)
            lam2 = float(-grad @ dw)
            newton_total += 1
            if lam2 / 2.0 < 1e-8:
                break
            f0, _ = phi(w, t)
            s = 1.0
            while True:
                f1, _ = phi(w + s * dw, t)
                if f1 <= f0 - 0.25 * s * lam2:
                    break
                s *= 0.5
                if s < 1e-12:
                    break
            w = w + s * dw
        gap = nu / t
        if gap < tol or newton_total > max_newton:
            break
        t *= mu
    obj = float(Lmat @ v0 + g @ w)
    return {"value": obj, "gap_bound": nu / t, "newton": newton_total, "sec": time.perf_counter() - t0, "k": k, "r": r, "m": d["m"], "rows": d["B"].shape[0], "start_res": d["res0"]}


def solve_all_perms(rel, nonneg: bool = None, tol: float = 1e-6) -> dict:
    """Löst mit dem Mittel aller n! Permutationspunkte als Start (n ≤ 7)."""
    nonneg = rel.nonneg if nonneg is None else nonneg
    return solve(rel, list(itertools.permutations(range(rel.n))), nonneg=nonneg, tol=tol)
