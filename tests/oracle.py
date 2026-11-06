"""Hilfen für die Tests: künstliche Zeilen der Leiter mit bekannten Antworten (ORACLE-PLAYBOOK §5)."""

import sdp_constants as C


def sdp_entry(bound, value=None, iterations=500, converged=True, seconds=1.0, round_cost=None):
    return {"value": bound if value is None else value, "bound": bound, "iterations": iterations, "converged": converged, "seconds": seconds, "primal": 1e-6, "dual": 1e-6, "round_cost": round_cost}


def make_ladder_row(n, ref, r2, r3, bound=None, family="cluster", seed=100000, opt="same", ub=None, round2=None, round3=None, **kw):
    """Künstliche Zeile von `ladder_row`: Vergleichswert `ref`, sichere Schranken `r2`, `r3`, die anderen Schranken in `bound` (Standard: glb 0,9 ref, pb 0,8 ref, da5 0,92 ref, lp 0,97 ref bis n = 9)."""
    if bound is None:
        bound = {"glb": 0.9 * ref, "pb": 0.8 * ref, "da5": 0.92 * ref}
        if n <= C.LP_MAX_N:
            bound["lp"] = 0.97 * ref
    return {"family": family, "n": n, "seed": seed, "ref": float(ref), "opt": float(ref) if opt == "same" else opt, "exact": (opt == "same") or opt is not None, "ub": ref if ub is None else ub, "tree_nodes": None,
            "bound": bound, "bsec": {k: 0.01 for k in bound}, "sdp": {"r2": sdp_entry(r2, round_cost=round2 if round2 is not None else ref + 5, **kw), "r3": sdp_entry(r3, round_cost=round3 if round3 is not None else ref + 1, **kw)}}
