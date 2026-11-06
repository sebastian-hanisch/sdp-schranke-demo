"""Die Relaxation auf einer Instanz, Vergleiche über feste Instanzen und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Alle Aussagen gelten für die gerechneten Instanzen, die sechs erfundenen Familien und die **eigene Fassung** der Relaxation (Facial Reduction, Pfeil, Gangster; R3 mit Y ≥ 0) und des ADMM (ρ-Anpassung, Schrittgrenze, Toleranz).
**Lücke** = (Vergleichswert − Schranke) / Vergleichswert; der Vergleichswert ist bis n = 9 das aufgezählte Optimum, bei n = 12 das vom Baum bewiesene (feste Reihen) oder die beste von 200 Abstiegen (einzelne Instanz: dann ist die Lücke eine
**untere** Grenze der wahren Lücke). Der **Wert des ADMM** ist bei grober Toleranz keine Schranke; Aussagen über Schranken beziehen sich auf die **sichere Schranke** aus der Dualmatrix. Zeiten sind Sekunden dieser numpy-Fassung auf diesem Rechner
(schwanken von Lauf zu Lauf um das Doppelte; nur Größenordnungen sind belastbar) und stehen nie in Tests."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import sdp_bounds as B
import sdp_constants as C
import sdp_cost as K
import sdp_landscape as L
import sdp_lp as LP
import sdp_relax as R
import sdp_spectral as SP
import sdp_tree as T
from sdp_instance import hand_instance, make_instance


@dataclass(frozen=True)
class Settings:
    family: str = C.DEFAULT_FAMILY
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    variant: str = C.DEFAULT_VARIANT
    tol: float = C.DEFAULT_TOL
    rho: float = C.DEFAULT_RHO
    adaptive: bool = C.DEFAULT_ADAPTIVE


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    return de(x, digits) + " %"


def fam(name: str) -> str:
    return C.FAMILY_NAMES.get(name, name)


def num(x: float) -> str:
    """Ganze Zahl ohne Nachkommastelle, sonst eine Stelle: 27 / 27,5 / −12,1."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else de(x, 1)


def count(x: float) -> str:
    return C.n_text(int(round(x)))


def secs(x: float) -> str:
    """Sekunden mit sinnvoller Stellenzahl: 0,016 s / 0,42 s / 5,1 s / 31 s."""
    if x < 0.1:
        return de(x, 3) + " s"
    if x < 10:
        return de(x, 2 if x < 1 else 1) + " s"
    return de(x, 0) + " s"


def gap(value: float, ref: float) -> float:
    """Lücke in Prozent: (ref − value) / ref; Rundungsreste unter 1e-9 Prozent werden zu 0."""
    g = 100.0 * (ref - value) / ref if ref else float("nan")
    return 0.0 if abs(g) < 1e-9 else g


def timed(f):
    t = time.perf_counter()
    r = f()
    return r, time.perf_counter() - t


# ----------------------------------------------------------------------------- eine Instanz

def reference(inst, seed: int, tree: bool = False) -> dict:
    """Vergleichswert: das aufgezählte Optimum (bis n = 9), mit `tree=True` das vom Baum bewiesene (Gilmore–Lawler-Knotenschranken, Anfangswert die beste von 200 Abstiegen), sonst die beste von 200 Abstiegen."""
    F, D = inst.F, inst.D
    ub, ub_perm = B.best_found(F, D, seed, C.STARTS)
    ub = int(ub)
    if inst.n in C.EXACT_SIZES:
        P = L.all_perms(inst.n)
        costs = L.costs_of(P, F, D)
        k = int(costs.argmin())
        return {"ref": float(costs[k]), "opt": float(costs[k]), "exact": True, "ub": ub, "ub_perm": [int(x) for x in ub_perm], "ref_perm": [int(x) for x in P[k]], "tree_nodes": None}
    if tree:
        tr = T.solve(F, D, "glb", incumbent=ub, cap=C.TREE_CAP)
        if not tr.capped:
            return {"ref": float(tr.best), "opt": float(tr.best), "exact": True, "ub": ub, "ub_perm": [int(x) for x in ub_perm], "ref_perm": [int(x) for x in (tr.perm if tr.perm is not None else ub_perm)], "tree_nodes": tr.nodes}
    return {"ref": float(ub), "opt": None, "exact": False, "ub": ub, "ub_perm": [int(x) for x in ub_perm], "ref_perm": [int(x) for x in ub_perm], "tree_nodes": None}


def other_bounds(inst) -> tuple:
    """(Schranken, Sekunden): Gilmore–Lawler, projizierte Eigenwertschranke, Dual Ascent (5 Runden), Linearisierung (nur bis n = 9)."""
    F, D = inst.F, inst.D
    bound, sec = {}, {}
    g, sec["glb"] = timed(lambda: B.gilmore_lawler(F, D))
    bound["glb"] = float(g.glb)
    s, sec["pb"] = timed(lambda: SP.spectral_bounds(F, D))
    bound["pb"] = float(s.pb)
    da, sec["da5"] = timed(lambda: B.dual_ascent(F, D, 5))
    bound["da5"] = float(da.final)
    if inst.n <= C.LP_MAX_N:
        lp, sec["lp"] = timed(lambda: LP.solve_lp(F, D))
        bound["lp"] = float(lp.value)
    return bound, sec


def relax_run(inst, variant: str, tol: float = C.DEFAULT_TOL, rho: float = C.DEFAULT_RHO, adaptive: bool = True, max_iter: int | None = None):
    rel = R.Relaxation(inst.F, inst.D, nonneg=(variant == "r3"))
    res = R.admm(rel, rho=rho, tol=tol, max_iter=C.MAX_ITER[variant] if max_iter is None else max_iter, adaptive=adaptive, every=C.ADAPT_EVERY)
    perm = R.rounded_arrangement(rel, res.Z)
    return rel, res, perm, int(K.cost(perm, inst.F, inst.D))


def instance_data(s: Settings) -> dict:
    """Instanz, Vergleichswert, die anderen Schranken und der Lauf der gewählten Stufe mit der gerundeten Anordnung."""
    inst = make_instance(s.family, s.n, s.seed)
    ref = reference(inst, s.seed)
    bound, bsec = other_bounds(inst)
    rel, res, perm, rcost = relax_run(inst, s.variant, s.tol, s.rho, s.adaptive)
    return {"inst": inst, **ref, "bound": bound, "bsec": bsec, "rel": rel, "res": res, "round_perm": perm, "round_cost": rcost}


def hand_data() -> dict:
    """Vier Plätze auf einer Zeile (Stück 3, 5, 8): beide Stufen, die anderen Schranken und der Permutationspunkt des Optimums."""
    inst = hand_instance(C.HAND_GAP_FLOWS, *C.HAND_GAP_GRID)
    F, D = inst.F, inst.D
    P = L.all_perms(4)
    costs = L.costs_of(P, F, D)
    k = int(costs.argmin())
    out = {"inst": inst, "opt": int(costs.min()), "opt_perm": [int(x) for x in P[k]], "glb": float(B.gilmore_lawler(F, D).glb), "da5": float(B.dual_ascent(F, D, 5).final), "lp": float(LP.solve_lp(F, D).value)}
    for v in C.VARIANTS:
        rel = R.Relaxation(F, D, nonneg=(v == "r3"))
        res = R.admm(rel, tol=1e-7, max_iter=C.MAX_ITER[v])
        out[v] = {"rel": rel, "res": res}
    rel = out["r2"]["rel"]
    Y = rel.permutation_point(out["opt_perm"])
    out["point_cost"] = rel.cost_of(Y)
    out["point_trace"] = float(np.trace(Y))
    return out


# ----------------------------------------------------------------------------- Zeilen der Vergleiche

def ladder_row(family: str, n: int, seed: int) -> dict:
    """Eine Instanz: Vergleichswert (n = 12: vom Baum bewiesen), die anderen Schranken und beide Stufen der SDP mit der gerundeten Anordnung."""
    inst = make_instance(family, n, seed)
    ref = reference(inst, seed, tree=True)
    bound, bsec = other_bounds(inst)
    row = {"family": family, "n": n, "seed": seed, **{k: ref[k] for k in ("ref", "opt", "exact", "ub", "tree_nodes")}, "bound": bound, "bsec": bsec, "sdp": {}}
    for v in C.VARIANTS:
        rel, res, perm, rcost = relax_run(inst, v)
        row["sdp"][v] = {"value": res.value, "bound": res.bound, "iterations": res.iterations, "converged": res.converged, "seconds": res.seconds, "primal": res.primal, "dual": res.dual, "round_cost": rcost}
    return row


def ladder_jobs(n: int) -> list:
    return [(f, n, s) for f in C.FAMILIES for s in range(C.FIXED_SEED0, C.FIXED_SEED0 + C.SEEDS[n])]


def ladder_sweep(family: str, n: int) -> list:
    return [ladder_row(*job) for job in ladder_jobs(n) if job[0] == family]


def pooled(by_family: dict) -> list:
    return [r for f in C.FAMILIES if f in by_family for r in by_family[f]]


def admm_study(variant: str) -> list:
    """Reihe „Der ADMM“ (n = 8, drei Familien): Schritte, Zeit, Wert und sichere Schranke je ρ (fest und angepasst) und je Toleranz; Bezugswert ein Lauf mit Toleranz 1e-8."""
    rows = []
    for fm, seed in C.RHO_ROWS:
        inst = make_instance(fm, C.RHO_N, seed)
        ref = R.admm(R.Relaxation(inst.F, inst.D, nonneg=(variant == "r3")), rho=1.0, tol=C.RHO_REF_TOL, max_iter=C.RHO_REF_MAX_ITER)
        opt = float(L.costs_of(L.all_perms(C.RHO_N), inst.F, inst.D).min())
        base = {"family": fm, "seed": seed, "ref": ref.value, "opt": opt}
        for rho in C.RHO_CHOICES:
            for adaptive in (False, True):
                _, res, _, _ = relax_run(inst, variant, 1e-5, rho, adaptive, max_iter=20000)
                rows.append({**base, "kind": "rho", "rho": rho, "adaptive": adaptive, "iterations": res.iterations, "converged": res.converged, "seconds": res.seconds, "value": res.value, "bound": res.bound})
        for tol in C.TOL_ROWS:
            _, res, _, _ = relax_run(inst, variant, tol, 1.0, True, max_iter=C.RHO_REF_MAX_ITER)
            rows.append({**base, "kind": "tol", "tol": tol, "iterations": res.iterations, "converged": res.converged, "seconds": res.seconds, "value": res.value, "bound": res.bound})
    return rows


def scale_row(n: int, variant: str) -> dict:
    """Eine Größe der Reihe „Die Größe“: Familie „zwei Cluster“, Seed 100000, Schrittgrenze `C.SCALE_MAX_ITER`."""
    inst = make_instance("cluster", n, C.FIXED_SEED0)
    rel, res, perm, rcost = relax_run(inst, variant, C.DEFAULT_TOL, C.DEFAULT_RHO, True, max_iter=C.SCALE_MAX_ITER)
    ub, _ = B.best_found(inst.F, inst.D, C.FIXED_SEED0, C.STARTS)
    t0 = time.perf_counter()
    rel.proj_K(np.eye(rel.N))
    t_proj = time.perf_counter() - t0
    return {"n": n, "variant": variant, "N": rel.N, "r": rel.r, "iterations": res.iterations, "converged": res.converged, "seconds": res.seconds, "per_iteration": res.seconds / res.iterations, "t_proj": t_proj,
            "value": res.value, "bound": res.bound, "glb": float(B.gilmore_lawler(inst.F, inst.D).glb), "ub": int(ub), "primal": res.primal, "dual": res.dual, "round_cost": rcost}


# ----------------------------------------------------------------------------- Auswertungen

def ladder_value(row: dict, key: str) -> float:
    """Die Schranke `key` einer Zeile: für die SDP die sichere Schranke."""
    if key in ("r2", "r3"):
        return row["sdp"][key]["bound"]
    return row["bound"][key]


def ladder_stats(rows: list, key: str) -> dict:
    """Lücke (Median, Mittel, größte) in Prozent, Zahl der scharfen Schranken (unter 0,05 %) und Zahl der Instanzen, in denen die Schranke gerechnet wurde."""
    rs = [r for r in rows if key in ("r2", "r3") or key in r["bound"]]
    g = [gap(ladder_value(r, key), r["ref"]) for r in rs]
    if not g:
        return {"count": 0, "median": float("nan"), "mean": float("nan"), "max": float("nan"), "sharp": 0}
    return {"count": len(rs), "median": float(np.median(g)), "mean": float(np.mean(g)), "max": float(max(g)), "sharp": sum(abs(x) < 0.05 for x in g)}


def below(rows: list, a: str, b: str, eps: float = 1e-3) -> int:
    """Zahl der Instanzen, in denen die Schranke `a` unter der Schranke `b` liegt (nur Instanzen, in denen beide gerechnet wurden)."""
    return sum(ladder_value(r, a) < ladder_value(r, b) - eps for r in rows if (a in ("r2", "r3") or a in r["bound"]) and (b in ("r2", "r3") or b in r["bound"]))


def beats_all_others(rows: list, key: str, eps: float = 1e-3) -> int:
    """Zahl der Instanzen, in denen `key` über der besten der Schranken Gilmore–Lawler, Eigenwert, Dual Ascent und Linearisierung liegt."""
    return sum(ladder_value(r, key) > max(r["bound"].values()) + eps for r in rows)


def family_medians(rows: list, key: str) -> dict:
    out = {}
    for f in C.FAMILIES:
        g = [gap(ladder_value(r, key), r["ref"]) for r in rows if r["family"] == f]
        if g:
            out[f] = float(np.median(g))
    return out


def round_stats(rows: list, variant: str) -> dict:
    """Wie oft rundet die Lösung auf das Optimum (nur Zeilen mit bekanntem Optimum), Lücke der Rundung im Median in Prozent, und die 200 Abstiege zum Vergleich."""
    rs = [r for r in rows if r["opt"] is not None]
    gaps = [100.0 * (r["sdp"][variant]["round_cost"] - r["opt"]) / r["opt"] for r in rs]
    return {"count": len(rs), "hit": sum(r["sdp"][variant]["round_cost"] == r["opt"] for r in rs), "median": float(np.median(gaps)) if gaps else float("nan"), "descents_hit": sum(r["ub"] == r["opt"] for r in rs)}


def convergence_stats(rows: list, variant: str) -> dict:
    its = [r["sdp"][variant]["iterations"] for r in rows]
    secs_ = [r["sdp"][variant]["seconds"] for r in rows]
    return {"count": len(rows), "not_converged": sum(not r["sdp"][variant]["converged"] for r in rows), "iterations_median": float(np.median(its)), "iterations_max": int(max(its)), "sec_median": float(np.median(secs_)), "sec_max": float(max(secs_))}


def safe_violations(rows: list) -> int:
    """Zahl der Läufe, in denen die sichere Schranke über dem Vergleichswert liegt (bei bewiesenem Optimum: Fehler; Toleranz 0,01)."""
    return sum(r["sdp"][v]["bound"] > r["ref"] + 0.01 for r in rows for v in C.VARIANTS if r["opt"] is not None)


# ----------------------------------------------------------------------------- Verdikte

def _head(s: Settings, v: dict) -> str:
    inst = v["inst"]
    return f"Familie **{fam(s.family)}**, {s.n} Bauteile, Raster {inst.rows}×{inst.cols}, Seed {s.seed}, {C.VARIANT_SHORT[s.variant]}"


def _refname(v: dict) -> str:
    return "das aufgezählte Optimum" if v["exact"] else "die beste von 200 Abstiegen (obere Grenze des Optimums)"


def verdict_core(s: Settings, v: dict):
    """Die Kernfrage: wie weit kommt die SDP, und gilt die Zahl?"""
    res = v["res"]
    ref = v["ref"]
    safe_gap = gap(res.bound, ref)
    others = v["bound"]
    best_key = max(others, key=others.get)
    best_other = others[best_key]
    ladder = ", ".join(f"{C.LADDER_NAMES[k]} {de(others[k], 1)}" for k in others)
    if res.converged:
        status = f"Der ADMM ist nach {count(res.iterations)} Schritten konvergiert (Residuen {de(res.primal, 6)} und {de(res.dual, 6)}), {secs(res.seconds)}."
    else:
        status = (f"**Der ADMM ist nach {count(res.iterations)} Schritten an der Schrittgrenze, nicht konvergiert** (Residuen {de(res.primal, 5)} und {de(res.dual, 5)}, {secs(res.seconds)}); "
                  "die sichere Schranke bleibt gültig, der Wert nicht.")
    if safe_gap < 0.05:
        what, kind = "Die sichere Schranke liegt auf dem Vergleichswert", "success"
    else:
        what, kind = f"Die Lücke der sicheren Schranke ist {pct(safe_gap, 2)}", "info"
    if res.bound > best_other + 0.05:
        cmp_ = f"über der besten der anderen Schranken ({C.LADDER_NAMES[best_key]} {de(best_other, 1)})"
    elif res.bound < best_other - 0.05:
        cmp_ = f"**unter** der besten der anderen Schranken ({C.LADDER_NAMES[best_key]} {de(best_other, 1)})"
    else:
        cmp_ = f"auf der Höhe der besten der anderen Schranken ({C.LADDER_NAMES[best_key]} {de(best_other, 1)})"
    over = f" Der Wert des ADMM ({de(res.value, 2)}) liegt über dem Optimum: bei dieser Toleranz keine Schranke." if v["exact"] and res.value > ref + 0.005 else ""
    rgap = 100.0 * (v["round_cost"] - ref) / ref
    rounding = f"Die Rundung der Lösung (Zeile 0 von Y, lineare Zuordnung) hat die Kosten {num(v['round_cost'])} ({'das ist ' + _refshort(v) if rgap < 1e-9 else 'Lücke ' + pct(rgap, 1)})."
    return kind, (f"{_head(s, v)}: sichere Schranke **{de(res.bound, 2)}** gegen {_refname(v)} {num(ref)}. {what}; sie liegt {cmp_}. Andere Schranken: {ladder}. {status}{over} {rounding} Eine Instanz, kein Befund über die Familie.")


def _refshort(v: dict) -> str:
    return "das Optimum" if v["exact"] else "so gut wie der Vergleichswert"


def verdict_relax(s: Settings, v: dict):
    rel = v["rel"]
    res = v["res"]
    n = rel.n
    tr = float(np.trace(res.Z))
    x = res.Z[0, 1:]
    frac = float(np.sum((x > 1e-3) & (x < 1 - 1e-3))) / len(x)
    kind = "info"
    return kind, (f"{_head(s, v)}: der Lift Y hat die Größe N = {rel.N}, nach der Facial Reduction ist die PSD-Matrix {rel.r}×{rel.r}; {count(rel.gangster_pairs)} Gangster-Paare sind null, Pfeil und Y00 = 1 gelten bis auf die Residuen. "
                  f"Die Spur der Lösung ist {de(tr, 3)} (jedes zulässige Y hat die Spur n + 1 = {n + 1}). {de(100 * frac, 0)} % der Einträge von x liegen echt zwischen 0 und 1: "
                  + ("die Lösung ist (fast) eine Anordnung." if frac < 0.01 else "die Lösung ist gebrochen, die Rundung muss entscheiden."))


def verdict_hand(h: dict):
    r2, r3 = h["r2"]["res"], h["r3"]["res"]
    return "info", (f"Vier Plätze auf einer Zeile (Abstände 1 bis 3), Fluss A–C, A–D, B–C, B–D je 1: Optimum **{h['opt']}** (Anordnung {''.join('ABCD'[i] for i in np.argsort(h['opt_perm']))}), Gilmore–Lawler {num(h['glb'])}, Dual Ascent {de(h['da5'], 2)}, Linearisierung {num(h['lp'])}, "
                    f"SDP R2 {de(r2.value, 3)} (sichere Schranke {de(r2.bound, 3)}), SDP R3 {de(r3.value, 3)} (sichere Schranke {de(r3.bound, 3)}). "
                    f"Der Permutationspunkt des Optimums liegt in der Relaxation (Kosten {num(h['point_cost'])}, Spur {num(h['point_trace'])} = n + 1), N = {h['r2']['rel'].N}, r = {h['r2']['rel'].r}, {h['r2']['rel'].gangster_pairs} Gangster-Paare.")


def verdict_ladder(rows_by_n: dict):
    parts = []
    for n in sorted(rows_by_n):
        rs = rows_by_n[n]
        st = {k: ladder_stats(rs, k) for k in C.LADDER_KEYS}
        lp = f", Linearisierung {de(st['lp']['median'], 1)} %" if st["lp"]["count"] else ""
        parts.append(f"n = {n} ({len(rs)} Instanzen): Lücke im Median Gilmore–Lawler {de(st['glb']['median'], 1)} %, Dual Ascent {de(st['da5']['median'], 1)} %{lp}, **SDP R2 {de(st['r2']['median'], 1)} %**, **SDP R3 {de(st['r3']['median'], 2)} %**; "
                     f"R2 unter Gilmore–Lawler in {below(rs, 'r2', 'glb')}, R3 über der besten der anderen in {beats_all_others(rs, 'r3')}, scharf (unter 0,05 %) in {st['r3']['sharp']}")
    return "info", ("; ".join(parts) + f". Sichere Schranke über dem Vergleichswert in {sum(safe_violations(r) for r in rows_by_n.values())} Läufen. Das Y ≥ 0 macht die SDP stark; ohne es ist sie auf dünnen und schiefen Familien oft schwächer als Gilmore–Lawler.")


def verdict_family(rows: list, n: int):
    r2, r3 = family_medians(rows, "r2"), family_medians(rows, "r3")
    return "info", (f"Bei n = {n}, Lücke im Median je Familie (R2 / R3): " + "; ".join(f"{fam(f)} {de(r2[f], 1)} / {de(r3[f], 2)} %" for f in r2) + ".")


def verdict_admm(rows: list, variant: str):
    rho_rows = [r for r in rows if r["kind"] == "rho"]
    tol_rows = [r for r in rows if r["kind"] == "tol"]
    fixed = [r["iterations"] for r in rho_rows if not r["adaptive"]]
    adapt = [r["iterations"] for r in rho_rows if r["adaptive"]]
    nc = sum(not r["converged"] for r in rho_rows)
    tols = sorted({r["tol"] for r in tol_rows})
    coarse = [r for r in tol_rows if r["tol"] == tols[-1]]
    over = max(r["value"] - r["ref"] for r in coarse)
    under = max(r["opt"] - r["bound"] for r in coarse)
    fine = [r for r in tol_rows if r["tol"] == tols[0]]
    return "info", (f"{C.VARIANT_SHORT[variant]}, n = {C.RHO_N}, drei Familien: Schritte je ρ von {count(min(fixed))} bis {count(max(fixed))} (festes ρ) und {count(min(adapt))} bis {count(max(adapt))} (angepasst); {nc} von {len(rho_rows)} Läufen an der Schrittgrenze. "
                    f"Bei Toleranz {C.tol_text(tols[-1])} liegt der Wert des ADMM bis zu {de(max(over, 0.0), 2)} **über** dem Optimum (keine Schranke), die sichere Schranke höchstens {de(max(under, 0.0), 2)} darunter; "
                    f"bei {C.tol_text(tols[0])} höchstens {de(max(r['opt'] - r['bound'] for r in fine), 3)}. Die sichere Schranke lag in {sum(r['bound'] > r['opt'] + 0.005 for r in rows)} von {len(rows)} Läufen über dem Optimum.")


def verdict_round(rows_by_n: dict):
    parts = []
    for n in sorted(rows_by_n):
        r3, r2 = round_stats(rows_by_n[n], "r3"), round_stats(rows_by_n[n], "r2")
        if not r3["count"]:
            continue
        parts.append(f"n = {n}: Rundung trifft das Optimum bei R3 in {r3['hit']} von {r3['count']} Instanzen (Lücke im Median {pct(r3['median'], 1)}), bei R2 in {r2['hit']}; 200 Abstiege in {r3['descents_hit']}")
    return "info", ("; ".join(parts) + ". Die Schranke ist stark, das Runden der Lösung nicht: wie in den Stücken 9 bis 11 liegt der Nutzen der Relaxation bei der Schranke.")


def verdict_scale(rows: list):
    r3 = sorted((r for r in rows if r["variant"] == "r3"), key=lambda r: r["n"])
    r2 = sorted((r for r in rows if r["variant"] == "r2"), key=lambda r: r["n"])
    nc3 = [r["n"] for r in r3 if not r["converged"]]
    nc2 = [r["n"] for r in r2 if not r["converged"]]
    last = r3[-1]
    return "info", (f"Bis n = {last['n']} (N = {last['N']}, PSD-Matrix {last['r']}×{last['r']}): Zeit je Schritt {', '.join(f'{de(1000 * r['per_iteration'], 1)} ms' for r in r3)} bei n = {', '.join(str(r['n']) for r in r3)} (Eigenzerlegung, wächst wie r³). "
                    f"Nicht konvergiert in {C.SCALE_MAX_ITER} Schritten: R3 bei n = {', '.join(map(str, nc3)) or '–'}; R2 bei n = {', '.join(map(str, nc2)) or '–'}. "
                    f"Sichere Schranke bei n = {last['n']}: R3 {de(last['bound'], 1)} gegen Gilmore–Lawler {de(last['glb'], 1)} und die beste von 200 Abstiegen {num(last['ub'])}.")
