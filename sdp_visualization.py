"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sdp_constants as C



def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_arrangement(inst, perm, title: str) -> go.Figure:
    """Die gerundete Anordnung: Bauteile auf den Plätzen, Verbindungen mit Fluss als Linien."""
    fig = go.Figure()
    xs, ys = inst.xy[:, 0].astype(float), -inst.xy[:, 1].astype(float)
    pos = [(xs[int(perm[i])], ys[int(perm[i])]) for i in range(inst.n)]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(symbol="square", size=36, color="rgba(150,150,150,0.18)", line=dict(width=1, color="rgba(150,150,150,0.5)")), hoverinfo="skip", showlegend=False))
    lx, ly = [], []
    for a in range(inst.n):
        for b in range(a + 1, inst.n):
            if inst.F[a, b]:
                lx += [pos[a][0], pos[b][0], None]
                ly += [pos[a][1], pos[b][1], None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(width=1.5, color="rgba(76,120,168,0.35)"), hoverinfo="skip", name="Fluss"))
    fig.add_trace(go.Scatter(x=[q[0] for q in pos], y=[q[1] for q in pos], mode="markers+text", text=list(inst.names), textposition="middle center", showlegend=False, marker=dict(size=26, color=C.COLORS["best"]), textfont=dict(color="white", size=10)))
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[xs.min() - 0.7, xs.max() + 0.7], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[ys.min() - 0.7, ys.max() + 0.7])
    return _base(fig, height=300)


def build_lift(rel, Z, ref_perm, title: str = "Der Lift Y und die relaxierte Zuordnung") -> go.Figure:
    """Links der Lift Y (Zeile und Spalte 0 gehören zu Y00 und x), rechts die relaxierte Zuordnung X (Zeile Bauteil, Spalte Platz) mit der Anordnung des Vergleichswerts (Kreuze)."""
    n = rel.n
    X = Z[0, 1:].reshape(n, n).T
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], subplot_titles=(f"Y ({rel.N}×{rel.N})", f"X: Bauteil gegen Platz (n = {n})"), horizontal_spacing=0.12)
    fig.add_trace(go.Heatmap(z=Z, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, showscale=False, hovertemplate="Zeile %{y}, Spalte %{x}: %{z:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=X, colorscale="Greens", zmin=0, zmax=1, colorbar=dict(thickness=10, len=0.8), hovertemplate="Bauteil %{y}, Platz %{x}: %{z:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[int(ref_perm[i]) for i in range(n)], y=list(range(n)), mode="markers", marker=dict(symbol="x", size=12, color="#c0504d", line=dict(width=2)), name="Anordnung des Vergleichswerts", hoverinfo="skip"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2, scaleanchor="x2", scaleratio=1)
    fig.update_layout(title=title)
    return _base(fig, height=400)


def build_residuals(res, title: str = "Residuen des ADMM") -> go.Figure:
    """Primal- und Dualresiduum je Schritt (logarithmisch), dazu ρ."""
    it = list(range(1, len(res.history["primal"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=it, y=res.history["primal"], mode="lines", name="Primalresiduum", line=dict(color=C.COLORS["line"])))
    fig.add_trace(go.Scatter(x=it, y=res.history["dual"], mode="lines", name="Dualresiduum", line=dict(color=C.COLORS["warn"])))
    fig.add_trace(go.Scatter(x=it, y=res.history["rho"], mode="lines", name="ρ", line=dict(color="#9aa0a6", dash="dot")))
    fig.update_layout(title=title)
    fig.update_xaxes(title="Schritt")
    fig.update_yaxes(type="log", title="Betrag")
    return _base(fig, height=320)


def build_ladder_bars(rows_by_n: dict, ladder_stats, title: str) -> go.Figure:
    """Lücke zum Optimum im Median je Schranke und Größe (Gilmore–Lawler, Eigenwert, Dual Ascent, Linearisierung, SDP R2, SDP R3)."""
    fig = go.Figure()
    for k in C.LADDER_KEYS:
        ns = [n for n in sorted(rows_by_n) if ladder_stats(rows_by_n[n], k)["count"]]
        vals = [ladder_stats(rows_by_n[n], k)["median"] for n in ns]
        fig.add_trace(go.Bar(x=[f"n = {n}" for n in ns], y=vals, name=C.LADDER_NAMES[k], marker_color=C.COLORS[k], text=[f"{v:.1f}" for v in vals], textposition="outside"))
    fig.update_layout(title=title, barmode="group")
    fig.update_yaxes(title="Lücke zum Vergleichswert in % (Median)", rangemode="tozero")
    return _base(fig, height=420)


def build_family_bars(rows: list, family_medians, title: str) -> go.Figure:
    """Lücke im Median je Familie für R2 und R3."""
    r2, r3 = family_medians(rows, "r2"), family_medians(rows, "r3")
    fs = [f for f in C.FAMILIES if f in r2]
    fig = go.Figure()
    for k, d in (("r2", r2), ("r3", r3)):
        fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fs], y=[d[f] for f in fs], name=C.VARIANT_SHORT[k], marker_color=C.COLORS[k], text=[f"{d[f]:.1f}" for f in fs], textposition="outside"))
    fig.update_layout(title=title, barmode="group")
    fig.update_yaxes(title="Lücke in % (Median)", rangemode="tozero")
    return _base(fig, height=340)


def build_rho_bars(rows: list, title: str) -> go.Figure:
    """Schritte je ρ (fest gegen angepasst), Median über die drei Familien; Läufe an der Schrittgrenze tragen die Grenze."""
    rr = [r for r in rows if r["kind"] == "rho"]
    rhos = sorted({r["rho"] for r in rr})
    fig = go.Figure()
    for adaptive, name, color in ((False, "ρ fest", "#9aa0a6"), (True, "ρ angepasst", C.COLORS["r3"])):
        ys = [float(np.median([r["iterations"] for r in rr if r["rho"] == rho and r["adaptive"] == adaptive])) for rho in rhos]
        fig.add_trace(go.Bar(x=[str(rho).replace(".", ",") for rho in rhos], y=ys, name=name, marker_color=color))
    fig.update_layout(title=title, barmode="group")
    fig.update_xaxes(title="Start-ρ")
    fig.update_yaxes(type="log", title="Schritte (Median über drei Familien)")
    return _base(fig, height=340)


def build_tol_chart(rows: list, title: str) -> go.Figure:
    """Wert des ADMM (durchgezogen) und sichere Schranke (gestrichelt) minus Optimum je Toleranz, je Familie: positiv heißt „Wert über dem Optimum“ (keine Schranke)."""
    tr = [r for r in rows if r["kind"] == "tol"]
    fig = go.Figure()
    for fm in sorted({r["family"] for r in tr}):
        rs = sorted((r for r in tr if r["family"] == fm), key=lambda r: r["tol"])
        color = C.COLORS["family"][fm]
        fig.add_trace(go.Scatter(x=[r["tol"] for r in rs], y=[r["value"] - r["opt"] for r in rs], mode="lines+markers", name=f"Wert, {C.FAMILY_NAMES[fm]}", line=dict(color=color), legendgroup=fm))
        fig.add_trace(go.Scatter(x=[r["tol"] for r in rs], y=[r["bound"] - r["opt"] for r in rs], mode="lines+markers", name=f"sichere Schranke, {C.FAMILY_NAMES[fm]}", line=dict(color=color, dash="dash"), legendgroup=fm))
    fig.add_hline(y=0, line=dict(color="#333", width=1))
    fig.update_layout(title=title)
    fig.update_xaxes(type="log", title="Toleranz", autorange="reversed")
    fig.update_yaxes(title="minus Optimum (Kosteneinheiten)")
    return _base(fig, height=380)


def build_round_bars(rows_by_n: dict, round_stats, title: str) -> go.Figure:
    """Anteil der Instanzen mit bekanntem Optimum, in denen die gerundete SDP-Lösung (R2, R3) und 200 Abstiege das Optimum treffen."""
    ns = [n for n in sorted(rows_by_n) if round_stats(rows_by_n[n], "r3")["count"]]
    fig = go.Figure()
    for name, color, f in (("Rundung R2", C.COLORS["r2"], lambda n: round_stats(rows_by_n[n], "r2")["hit"]), ("Rundung R3", C.COLORS["r3"], lambda n: round_stats(rows_by_n[n], "r3")["hit"]),
                           ("200 Abstiege", "#9aa0a6", lambda n: round_stats(rows_by_n[n], "r3")["descents_hit"])):
        fig.add_trace(go.Bar(x=[f"n = {n}" for n in ns], y=[100.0 * f(n) / round_stats(rows_by_n[n], "r3")["count"] for n in ns], name=name, marker_color=color))
    fig.update_layout(title=title, barmode="group")
    fig.update_yaxes(title="trifft das Optimum in % der Instanzen", range=[0, 105])
    return _base(fig, height=340)


def build_scale(rows: list, title: str = "Zeit je Schritt und Schritte über die Größe") -> go.Figure:
    """Oben die Zeit je Schritt (logarithmisch), unten die Schritte; nicht konvergierte Läufe sind hohl markiert."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14, subplot_titles=("Zeit je Schritt in ms", "Schritte bis zum Abbruch"))
    for v in C.VARIANTS:
        rs = sorted((r for r in rows if r["variant"] == v), key=lambda r: r["n"])
        if not rs:
            continue
        sym = ["circle" if r["converged"] else "circle-open" for r in rs]
        fig.add_trace(go.Scatter(x=[r["n"] for r in rs], y=[1000 * r["per_iteration"] for r in rs], mode="lines+markers", name=C.VARIANT_SHORT[v], line=dict(color=C.COLORS[v]), marker=dict(symbol=sym, size=10), legendgroup=v), row=1, col=1)
        fig.add_trace(go.Scatter(x=[r["n"] for r in rs], y=[r["iterations"] for r in rs], mode="lines+markers", name=C.VARIANT_SHORT[v], line=dict(color=C.COLORS[v]), marker=dict(symbol=sym, size=10), legendgroup=v, showlegend=False), row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(title="Bauteile n", tickvals=sorted({r["n"] for r in rows}), row=2, col=1)
    fig.update_layout(title=title)
    return _base(fig, height=520)


def build_instance_bounds(v: dict, variant: str, title: str = "Schranken dieser Instanz") -> go.Figure:
    """Waagerechte Balken: die anderen Schranken und die sichere Schranke der SDP; senkrecht der Vergleichswert."""
    keys = list(v["bound"]) + [variant]
    vals = [v["bound"][k] for k in v["bound"]] + [v["res"].bound]
    names = [C.LADDER_NAMES[k] for k in v["bound"]] + [f"{C.VARIANT_SHORT[variant]}, sichere Schranke"]
    fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker_color=[C.COLORS[k] for k in keys], text=[f"{x:.1f}" for x in vals], textposition="outside"))
    fig.add_vline(x=v["ref"], line=dict(color="#c0504d", dash="dot"), annotation_text="Optimum" if v["exact"] else "beste von 200 Abstiegen", annotation_position="top")
    lo = min(vals)
    fig.update_xaxes(range=[lo * 0.85, max(max(vals), v["ref"]) * 1.06], title="Kosten")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, showlegend=False)
    return _base(fig, height=300, legend_below=False)
