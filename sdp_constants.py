"""Konstanten: Größen, Raster, Familien der Flussmatrix, die semidefinite Relaxation des QAP (Stufen R2 und R3), der ADMM, feste Reihen der Experimente, Farben und Presets."""

SIZES = (4, 6, 8, 9, 12)                              # Regler „Bauteile“ (n = 4: Raster 2×2; das Handbeispiel „vier Plätze auf einer Zeile“ steht fest in Experiment 1)
EXACT_SIZES = (4, 6, 8, 9)                            # alle n! Anordnungen werden aufgezählt: das Optimum ist bekannt
HEAVY_N = 12                                          # ab hier läuft die Rechnung nur auf Knopfdruck (Median etwa 10 s, größter Fall 70 s)
DEFAULT_N = 8
GRID = {4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3), 10: (2, 5), 12: (3, 4), 14: (2, 7), 16: (4, 4)}   # Plätze als Raster (Zeilen, Spalten), Abstand = Manhattan
SEED_MAX, DEFAULT_SEED = 999999, 48
FIXED_SEED0 = 100000                                  # feste Instanzen: Seeds ab 100000 (nie ein einzelner Seed als Befund)
SEEDS = {6: 5, 8: 5, 9: 3, 12: 3}                     # Seeds je Familie: 30 / 30 / 18 / 18 Instanzen
LADDER_SIZES = (6, 8, 9, 12)                          # Größen der Reihe „Die Leiter“
SCALE_SIZES = (6, 8, 10, 12, 14, 16)                  # Größen der Reihe „Die Größe“ (eine Instanz, Familie „zwei Cluster“, Seed 100000)
SCALE_MAX_ITER = 8000
TREE_CAP = 300_000                                    # Obergrenze der Knoten des Baums, der das Optimum bei n = 12 belegt
STARTS = 200                                          # Abstiege aus Zufallsstarts (obere Schranke, Anfangswert des Baums)
HAND_N = 4

# --- Die Relaxation und der ADMM -----------------------------------------------------------------------------------------------------------------
VARIANTS = ("r2", "r3")
VARIANT_NAMES = {"r2": "R2: Pfeil und Gangster", "r3": "R3: zusätzlich Y ≥ 0"}
VARIANT_SHORT = {"r2": "SDP R2", "r3": "SDP R3"}
VARIANT_HELP = {"r2": "Lift Y mit Y00 = 1, Pfeil (Y0a = Yaa) und Gangster (Einträge zu zwei Plätzen eines Bauteils oder zwei Bauteilen eines Platzes sind null), nach Facial Reduction positiv semidefinit.",
                "r3": "Wie R2, dazu Y ≥ 0 (doppelt nichtnegativ): alle Einträge außerhalb des Gangster-Musters sind nichtnegativ."}
DEFAULT_VARIANT = "r3"
TOL_CHOICES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
DEFAULT_TOL = 1e-5
RHO_CHOICES = (0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_RHO = 1.0
DEFAULT_ADAPTIVE = True
ADAPT_EVERY = 100                                     # alle so viele Schritte: ρ verdoppeln oder halbieren, wenn ein Residuum zehnmal größer ist als das andere
MAX_ITER = {"r2": 6000, "r3": 15000}                  # Schrittgrenze (danach „nicht konvergiert“)
CONVERGED_FACTOR = 2.0                                # „konvergiert“ heißt: beide Residuen höchstens Faktor · Toleranz
RHO_ROWS = ((("uniform", 100001), ("cluster", 100001), ("skewed", 100001)))   # Instanzen der Reihe „Der ADMM“ (n = 8)
RHO_N = 8
RHO_REF_TOL = 1e-8
RHO_REF_MAX_ITER = 30000
TOL_ROWS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

LADDER_KEYS = ("glb", "pb", "da5", "lp", "r2", "r3")
LADDER_NAMES = {"glb": "Gilmore–Lawler", "pb": "Projizierte Eigenwertschranke", "da5": "Dual Ascent (5 Runden)", "lp": "Linearisierung (LP)", "r2": "SDP R2", "r3": "SDP R3"}
LADDER_PIECE = {"glb": "Stück 3", "pb": "Stück 4", "da5": "Stück 5", "lp": "Stück 5", "r2": "dieses Stück", "r3": "dieses Stück"}
LP_MAX_N = 9                                          # die Linearisierung (LP) rechnet nur bis n = 9

FAMILIES = ("netlist", "uniform", "sparse", "cluster", "star", "skewed")
DEFAULT_FAMILY = "cluster"
FAMILY_NAMES = {"netlist": "Netzliste", "uniform": "Gleichverteilt", "sparse": "Dünn besetzt", "cluster": "Zwei Cluster", "star": "Stern", "skewed": "Wenige starke Flüsse"}
FAMILY_HELP = {
    "netlist": "Netzliste aus Dreiergruppen (IC, Kondensator, Widerstand), jedes Paar in einem Netz zählt Fluss 1: die Familie aus Stück 1.",
    "uniform": "Jedes Paar von Bauteilen hat einen zufälligen Fluss von 0 bis 9, gleichverteilt: dichte Matrix ohne Struktur.",
    "sparse": "Nur drei von zehn Paaren haben Fluss (1 bis 9), die übrigen keinen: dünn besetzt ohne weitere Struktur.",
    "cluster": "Zwei Gruppen von Bauteilen: innerhalb einer Gruppe starker Fluss (8 bis 10), zwischen den Gruppen schwacher (0 bis 2).",
    "star": "Ein Bauteil (die Nabe) hat starken Fluss (1 bis 9) zu allen anderen; zwischen den übrigen höchstens 1.",
    "skewed": "Die Flüsse sind schief verteilt: viele Paare ohne oder mit kleinem Fluss, wenige mit sehr großem (bis 60).",
}

# Netzliste (wie in Stück 1 bis 5)
N_NETS_PER_COMPONENT = 1.5
NET_SIZES = (2, 3, 4)
NET_WEIGHTS = (5, 3, 2)
LOCAL_NET_PERCENT = 70
GROUP = 3
# Dünn besetzt, Cluster, Stern, schief (wie in Stück 2)
SPARSE_PERCENT = 30
UNIFORM_MAX = 9
CLUSTER_STRONG, CLUSTER_WEAK_MAX = 8, 2
SKEW_VALUES = (0, 1, 2, 4, 8, 16, 32, 60)
SKEW_WEIGHTS = (45, 22, 13, 8, 6, 3, 2, 1)           # ganzzahlige Gewichte (Summe 100), schweres Ende

HAND_FLOWS = ((0, 1, 5), (1, 2, 1))                   # Handbeispiel 1: drei Bauteile auf einer Zeile, Fluss A–B 5, B–C 1
HAND_GRID = (1, 3)
HAND_GAP_FLOWS = ((0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1))   # Handbeispiel 3: vier Bauteile auf einer Zeile, A und B gegen C und D
HAND_GAP_GRID = (1, 4)
HAND_GRID_FLOWS = ((0, 1, 3), (0, 2, 1), (1, 3, 2), (2, 3, 4))  # Handbeispiel 2 (aus Stück 1): Raster 2×2
HAND_GRID_GRID = (2, 2)

COLORS = {"best": "#2F6B65", "local": "#c0504d", "pad": "#b08a2e", "line": "#4c78a8", "ok": "#2F6B65", "warn": "#b08a2e", "bad": "#b34a4a",
          "r2": "#b08a2e", "r3": "#2F6B65", "glb": "#4c78a8", "pb": "#9c755f", "da5": "#8e6bb0", "lp": "#c0504d",
          "family": {"netlist": "#4c78a8", "uniform": "#9c755f", "sparse": "#b08a2e", "cluster": "#2F6B65", "star": "#8e6bb0", "skewed": "#c0504d"}}

_BASE = {"family": DEFAULT_FAMILY, "n": DEFAULT_N, "seed": DEFAULT_SEED, "variant": DEFAULT_VARIANT, "tol": DEFAULT_TOL, "rho": DEFAULT_RHO, "adaptive": DEFAULT_ADAPTIVE}
PRESETS = {
    "Standardfall: R3, acht Bauteile": dict(_BASE),
    "R2 statt R3": dict(_BASE, variant="r2"),
    "Vier Bauteile": dict(_BASE, n=4, family="uniform"),
    "Dünn besetzt: R2 schwach": dict(_BASE, family="sparse", variant="r2", seed=100000),
    "Schiefe Flüsse: R2 schwach": dict(_BASE, family="skewed", variant="r2", seed=100001),
    "Grobe Toleranz": dict(_BASE, tol=1e-2),
    "Festes ρ, groß": dict(_BASE, rho=10.0, adaptive=False),
    "Zwölf Bauteile": dict(_BASE, n=12),
}
PRESET_HELP = {
    "Standardfall: R3, acht Bauteile": "Acht Bauteile, Familie „zwei Cluster“: die SDP mit Y ≥ 0 liegt fast auf dem Optimum; die sichere Schranke belegt es.",
    "R2 statt R3": "Dieselbe Instanz ohne Y ≥ 0: die Schranke fällt zurück, oft unter die Gilmore–Lawler-Schranke.",
    "Vier Bauteile": "Vier Bauteile auf dem 2×2-Raster: die Relaxation passt auf einen Bildschirm (N = 17).",
    "Dünn besetzt: R2 schwach": "Wenige Flüsse: R2 liegt weit unter dem Optimum und unter Gilmore–Lawler; R3 schließt die Lücke.",
    "Schiefe Flüsse: R2 schwach": "Wenige sehr starke Flüsse: dasselbe Bild wie bei dünnen Flüssen.",
    "Grobe Toleranz": "Toleranz 1e-2: der Wert des ADMM ist keine Schranke mehr, die sichere Schranke bleibt gültig.",
    "Festes ρ, groß": "ρ = 10 ohne Anpassung: der ADMM braucht viele Schritte; mit Anpassung (Standard) deutlich weniger.",
    "Zwölf Bauteile": "Zwölf Bauteile: nicht aufzählbar (Vergleichswert: beste von 200 Abstiegen); die Rechnung läuft nur auf Knopfdruck, im Median etwa zehn Sekunden.",
}


def n_text(n: int) -> str:
    """Anzahl mit Leerzeichen als Tausendertrenner: 362880 → „362 880“."""
    return f"{n:,}".replace(",", " ")


def tol_text(x: float) -> str:
    """Toleranz als Zehnerpotenz: 1e-5."""
    return f"1e-{round(-__import__('math').log10(x))}"


assert set(PRESETS) == set(PRESET_HELP) and set(FAMILY_NAMES) == set(FAMILIES) == set(FAMILY_HELP) and set(VARIANT_NAMES) == set(VARIANTS) == set(VARIANT_HELP) and set(LADDER_NAMES) == set(LADDER_KEYS) == set(LADDER_PIECE)
assert all(p["n"] in SIZES and p["family"] in FAMILIES and 0 <= p["seed"] <= SEED_MAX and p["variant"] in VARIANTS and p["tol"] in TOL_CHOICES and p["rho"] in RHO_CHOICES for p in PRESETS.values())
assert all(r * c == n for n, (r, c) in GRID.items()) and sum(SKEW_WEIGHTS) == 100 and set(SIZES) <= set(GRID) and set(SCALE_SIZES) <= set(GRID) and set(SEEDS) == set(LADDER_SIZES)
