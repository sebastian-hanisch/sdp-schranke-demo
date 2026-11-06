"""PDF-Export des aktuellen Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import sdp_constants as C
from sdp_evaluation import Settings, count, de, gap, num, pct, secs

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Δ": "Delta", "↔": "<->", "„": '"', "“": '"', "²": "2", "³": "3", "⁻": "-", "¹": "1", "½": "1/2", "⁴": "4", "λ": "lambda", "ρ": "rho", "ψ": "psi",
            "Ṽ": "V~", "ᵀ": "T", "⪰": ">=", "≈": "~"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, v: dict) -> bytes:
    """`v`: das Ergebnis von `instance_data`."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("SDP-Schranke für das quadratische Zuordnungsproblem"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    inst, res, rel = v["inst"], v["res"], v["rel"]
    how = "alle Anordnungen aufgezählt" if v["exact"] else "nicht aufgezählt, beste von 200 Abstiegen als Vergleich (obere Grenze des Optimums)"
    pdf.multi_cell(0, 5, _s(f"Familie {C.FAMILY_NAMES[s.family]}, {s.n} Bauteile, {inst.rows}x{inst.cols}-Raster, Seed {s.seed}; {C.VARIANT_NAMES[s.variant]}; Toleranz {C.tol_text(s.tol)}, rho {de(s.rho, 1)} "
                            f"{'angepasst' if s.adaptive else 'fest'}; {how}. Eine Instanz, kein Befund über die Familie. Erfundene Familien, eigene Fassung der Relaxation und des ADMM. Zeiten sind Messungen dieser Fassung. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    for w, h in ((70, "Schranke"), (35, "Wert"), (35, "Lücke zum Vergleichswert")):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    rows = [(C.LADDER_NAMES[k], de(x, 2), pct(gap(x, v["ref"]), 2)) for k, x in v["bound"].items()]
    rows.append((f"{C.VARIANT_SHORT[s.variant]}, sichere Schranke", de(res.bound, 3), pct(gap(res.bound, v["ref"]), 2)))
    rows.append((f"{C.VARIANT_SHORT[s.variant]}, Wert des ADMM", de(res.value, 3), pct(gap(res.value, v["ref"]), 2)))
    for name, val, g in rows:
        pdf.cell(70, 5.5, _s(name), border=1)
        pdf.cell(35, 5.5, _s(val), border=1)
        pdf.cell(35, 5.5, _s(g), border=1)
        pdf.ln()
    pdf.ln(3)
    ref_name = "Optimum (aufgezählt)" if v["exact"] else "Beste von 200 Abstiegen (obere Grenze)"
    status = "konvergiert" if res.converged else "NICHT konvergiert (Schrittgrenze)"
    pdf.multi_cell(0, 5, _s(f"{ref_name}: {num(v['ref'])}. ADMM: {count(res.iterations)} Schritte, {status}, Residuen {de(res.primal, 6)} und {de(res.dual, 6)}, {secs(res.seconds)}. "
                            f"Lift N = {rel.N}, PSD-Matrix {rel.r}x{rel.r}, {count(rel.gangster_pairs)} Gangster-Paare. Gerundete Anordnung: Kosten {num(v['round_cost'])}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: sdp-schranke-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
