"""Elementos comunes: datos de entrada, catálogos, fórmulas base y estructura de resultados.

Fuentes:
  [C]  Carpio Chávez, A. (2025). Diseño de perforación y voladura de labores de avance. Cusco.
  [LJ] López Jimeno, C. et al. (1987). Manual de perforación y voladura de rocas. IGME, Madrid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Catálogos
# ----------------------------------------------------------------------------
CARTRIDGES = {  # nombre: (diámetro m, longitud m)
    '1 1/4" x 12"': (0.03175, 0.3048),
    '1" x 8"': (0.0254, 0.2032),
    '1" x 12"': (0.0254, 0.3048),
    '7/8" x 7"': (0.0222, 0.1778),
}
# Emulnor: datos del Ejemplo N°06 [C]. 7/8"x7": peso estimado con la densidad del producto.
EXPLOSIVES = {
    "Emulnor 500": dict(vod=4000, rho=0.90, rws=0.70, granel=False,
                        w={'1 1/4" x 12"': 0.200, '1" x 8"': 0.084, '1" x 12"': 0.128}),
    "Emulnor 1000": dict(vod=5800, rho=1.13, rws=0.87, granel=False,
                         w={'1 1/4" x 12"': 0.250, '1" x 8"': 0.108, '1" x 12"': 0.166}),
    "Emulnor 3000": dict(vod=5700, rho=1.14, rws=1.02, granel=False,
                         w={'1 1/4" x 12"': 0.265, '1" x 8"': 0.109, '1" x 12"': 0.173}),
    "Emulnor 5000": dict(vod=5500, rho=1.16, rws=1.12, granel=False,
                         w={'1 1/4" x 12"': 0.265, '1" x 8"': 0.115, '1" x 12"': 0.176}),
    "ANFO (granel)": dict(vod=3000, rho=0.80, rws=1.00, granel=True, w={}),
    "Emulsión a granel": dict(vod=5000, rho=1.15, rws=0.90, granel=True, w={}),
}

# Detonadores no eléctricos Exsanel — Tabla N°05 [C]
EXSANEL_MS = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200]
EXSANEL_LP = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1800, 2400, 3000, 3800, 4600, 5500, 6400,
              7400, 8500, 9600]

# Tiempo mínimo entre taladros del arranque (ms por metro de taladro) — Tabla N°04 [C]
def ms_por_metro(RMR):
    if RMR <= 20:
        return 10
    if RMR <= 40:
        return 20
    if RMR <= 60:
        return 25
    if RMR <= 80:
        return 35
    return 40


# Rangos de P80 que se manejan en minas — sección 6.1.7 [C]
P80_REFERENCIAS = {
    "Libre": None,
    "Cobre gran escala con molino SAG (15–30 cm)": (15, 30),
    "Oro con molino de bolas (10–20 cm)": (10, 20),
    "Cantera de agregados (30–60 cm)": (30, 60),
}

# Predimensionamiento de taladros — Tablas N°06 y N°07 [C]
def dt_y_C(RMR):
    if RMR > 60:
        return 0.525, 2.0
    if RMR > 40:
        return 0.625, 1.5
    return 0.725, 1.0


# Factor de carga por área — Tabla N°08 [C] (kg/m³: mínimo, máximo)
def factor_carga_tabla(area, RMR):
    filas = [(5, (2.60, 3.20), (1.80, 2.30), (1.20, 1.60)),
             (10, (2.0, 2.60), (1.40, 1.80), (0.90, 1.20)),
             (20, (1.65, 2.0), (1.10, 1.40), (0.60, 0.90)),
             (40, (1.20, 1.65), (0.75, 1.10), (0.40, 0.60)),
             (1e9, (0.80, 1.20), (0.50, 0.75), (0.30, 0.40))]
    k = 1 if RMR > 60 else 2 if RMR > 40 else 3
    for lim, *v in filas:
        if area <= lim:
            return v[k - 1]
    return filas[-1][k]


# ----------------------------------------------------------------------------
# Datos de entrada
# ----------------------------------------------------------------------------
@dataclass
class Rock:
    RMR: float = 50
    GSI: float = 45
    RQD: float = 55
    UCS: float = 75          # MPa
    rho: float = 2.75        # t/m³
    Q: float = 1.95          # Barton
    JSF: float = 0.9
    E: float = 35.0          # GPa (Kuz-Ram/Lilly)

    @property
    def CE(self):  # consumo específico [C]
        return 0.56 * self.rho * math.tan(math.radians(5 + self.RMR / 2)) / ((115 - self.RQD) / 3.3) ** (1 / 3)

    @property
    def c(self):   # constante sueca [C]
        return 0.8784 * self.CE + 0.0052

    @property
    def Kv(self):  # factor de volabilidad de Bórquez [C]
        return 1.96 - 0.27 * math.log(max(self.RQD * self.JSF, 1e-3))

    @property
    def Std(self):  # tracción dinámica [C] (MPa)
        return 4.5 * 10.197 * 0.08 * self.UCS

    @property
    def Vp(self):
        return 3500 + 1000 * math.log10(max(self.Q, 1e-3))


@dataclass
class Explosive:
    name: str = "Emulnor 1000"
    vod: float = 5800
    rho: float = 1.13          # g/cm³
    rws: float = 0.87          # potencia relativa en peso referida al ANFO (fracción)
    cart: str = '1 1/4" x 12"'
    w_cart: float = 0.25       # kg/cartucho (0 si es a granel)
    L_cart: float = 0.3048     # m
    granel: bool = False

    def q_lineal(self, d_hole_m: float, acople: float = 1.0) -> float:
        """Densidad lineal de carga (kg/m): cartucho = peso/longitud; granel = ρ·π·D²/4."""
        if self.granel or self.w_cart <= 0:
            return self.rho * 1000 * math.pi * d_hole_m ** 2 / 4 * acople
        return self.w_cart / self.L_cart

    @property
    def PD(self):  # presión de detonación [C] (MPa)
        return self.rho * 1000 * 1e-6 * self.vod ** 2 / 4

    def label(self):
        return self.name if self.granel else f"{self.name} {self.cart}"


def make_explosive(name, cart=None, custom=None) -> Explosive:
    if name == "Personalizado" and custom:
        return Explosive(**custom)
    e = EXPLOSIVES[name]
    if e["granel"]:
        return Explosive(name=name, vod=e["vod"], rho=e["rho"], rws=e["rws"], cart="granel", w_cart=0.0,
                         L_cart=1.0, granel=True)
    cart = cart or '1 1/4" x 12"'
    w = e["w"].get(cart)
    if w is None:  # cartucho no tabulado: estimar con la densidad
        dcar, lcar = CARTRIDGES[cart]
        w = e["rho"] * 1000 * math.pi * dcar ** 2 / 4 * lcar
    return Explosive(name=name, vod=e["vod"], rho=e["rho"], rws=e["rws"], cart=cart, w_cart=w,
                     L_cart=CARTRIDGES[cart][1], granel=False)


# ----------------------------------------------------------------------------
# Fórmulas de burden
# ----------------------------------------------------------------------------
def c_hat_holmberg(c, B, rule="Iterativa"):
    if rule == "c + 0.05":
        return c + 0.05
    return c + 0.05 if B >= 1.4 else c + 0.07 / B


def bmax_holmberg(q, rws, c, f, sb, rule="Iterativa"):
    """B_max = 0.9·√(q·RWS/(Ĉ·f·S/B)) — tajeo/destroza ([C] Ec. 48–78, [LJ] cap. 22)."""
    B = 0.9 * math.sqrt(q * rws / ((c + 0.05) * f * sb))
    for _ in range(60):
        Bn = 0.9 * math.sqrt(q * rws / (c_hat_holmberg(c, B, rule) * f * sb))
        if abs(Bn - B) < 1e-7:
            break
        B = Bn
    return B, c_hat_holmberg(c, B, rule)


def bmax_langefors(D_mm, rho_e, prp, c, f, sb):
    """B_max = D/33·√(ρe·PRP/(ĉ·f·S/B)); ĉ = c+0.75 (1.4–15 m), c+0.07/B (B<1.4) — [LJ] cap. 20 y 24."""
    def B_of(ch):
        return D_mm / 33 * math.sqrt(rho_e * prp / (ch * f * sb))
    B = B_of(c + 0.75)
    if B >= 1.4:
        return B, c + 0.75
    Bs = 1.0
    for _ in range(80):
        Bn = B_of(c + 0.07 / Bs)
        if abs(Bn - Bs) < 1e-7:
            break
        Bs = Bn
    if Bs >= 1.4:  # zona ambigua entre ambas reglas: se adopta el límite
        return 1.4, c + 0.05
    return Bs, c + 0.07 / Bs


def b_pearse_kv(Kv, D_mm, PD, Std):
    """B = Kv·D/1000·√(PD/Std) — Pearse modificado con Kv de Bórquez ([C] Ec. 06)."""
    return Kv * D_mm / 1000 * math.sqrt(PD / Std)


# ----------------------------------------------------------------------------
# Registro de cálculos y resultado
# ----------------------------------------------------------------------------
class Log:
    def __init__(self):
        self.rows = []

    def add(self, fase, par, sym, val, unit="", eq=""):
        v = val
        if isinstance(val, (int, float, np.floating, np.integer)):
            v = round(float(val), 4)
        self.rows.append(dict(Fase=fase, Parámetro=par, Símbolo=sym, Valor=v, Unidad=unit, Ecuación=eq))
        return val

    def df(self):
        d = pd.DataFrame(self.rows, columns=["Fase", "Parámetro", "Símbolo", "Valor", "Unidad", "Ecuación"])
        d["Valor"] = d["Valor"].map(lambda v: f"{v:,.4g}" if isinstance(v, float) else str(v))
        return d


HOLE_COLS = ["id", "zona", "grupo", "cuele", "x", "y", "X0", "Y0", "Z0", "ux", "uy", "uz", "L", "Lt",
             "cargas", "kg", "n_cart", "explosivo", "B", "S", "q", "primer_s", "d_mm", "tacos"]


@dataclass
class Result:
    """Resultado común de cualquier tipo de labor."""
    labor: str
    metodo: str
    log: Log = field(default_factory=Log)
    holes: pd.DataFrame | None = None
    relief: pd.DataFrame | None = None        # x, y, d_mm, X0, Y0, Z0, ux, uy, uz, L
    sources: list = field(default_factory=list)  # caras libres iniciales (x, y) en el plano 2D
    view2d: dict = field(default_factory=dict)
    scene3d: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    frag: dict | None = None    # parámetros para Kuz-Ram/ML (B, S, q, Lc, H, d_mm, W, rws, modo)
    p: dict = field(default_factory=dict)
    zone_colors: dict = field(default_factory=dict)
    zone_order: list = field(default_factory=list)
    simultaneous: set = field(default_factory=set)


def new_hole(zona, grupo, x, y, X0, Y0, Z0, u, L, Lt, cargas, d_mm, explosivo="", B=np.nan, S=np.nan,
             q=np.nan, cuele=False, primer_s=None, w_cart=0.0, tacos=None):
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    kg = float(sum((s1 - s0) * qq for s0, s1, qq in cargas))
    n_cart = int(max(1, round(kg / w_cart))) if (w_cart and kg > 0) else 0
    if primer_s is None:
        primer_s = max(s1 for _, s1, _ in cargas) - 0.12 if cargas else L
    return dict(zona=zona, grupo=int(grupo), cuele=bool(cuele), x=float(x), y=float(y), X0=float(X0),
                Y0=float(Y0), Z0=float(Z0), ux=u[0], uy=u[1], uz=u[2], L=float(L), Lt=float(Lt), cargas=cargas,
                kg=kg, n_cart=n_cart, explosivo=explosivo, B=float(B) if B == B else np.nan,
                S=float(S) if S == S else np.nan, q=float(q) if q == q else np.nan, primer_s=float(primer_s),
                d_mm=float(d_mm), tacos=tacos if tacos is not None else ([(0.0, Lt)] if Lt > 0 else []))


def finalize_holes(rows, order):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=HOLE_COLS)
    rank = {z: i for i, z in enumerate(order)}
    df["_r"] = df.zona.map(lambda z: rank.get(z, 99))
    cx, cy = df.x.mean(), df.y.mean()
    df["_a"] = np.degrees(np.arctan2(df.y - cy, df.x - cx)) % 360
    df = df.sort_values(["_r", "_a"]).drop(columns=["_r", "_a"]).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df


def relief_row(x, y, d_mm, X0, Y0, Z0, u, L):
    u = np.asarray(u, dtype=float) / np.linalg.norm(u)
    return dict(x=x, y=y, d_mm=d_mm, X0=X0, Y0=Y0, Z0=Z0, ux=u[0], uy=u[1], uz=u[2], L=L)


def summary_common(res: Result, vol, rho_r, cost=None, extra=None):
    h = res.holes
    n = len(h)
    kg = float(h.kg.sum()) if n else 0.0
    drill = float(h.L.sum()) + (float(res.relief.L.sum()) if res.relief is not None and len(res.relief) else 0.0)
    out = {"N° taladros cargados": n,
           "N° taladros vacíos": 0 if res.relief is None else len(res.relief),
           "Volumen roto (m³)": vol, "Tonelaje roto (t)": vol * rho_r,
           "Explosivo total (kg)": kg,
           "Factor de carga (kg/m³)": kg / vol if vol > 0 else np.nan,
           "Factor de carga (g/t)": 1000 * kg / (vol * rho_r) if vol > 0 else np.nan,
           "Perforación total (m)": drill,
           "Perforación específica (m/m³)": drill / vol if vol > 0 else np.nan}
    if extra:
        out.update(extra)
    if cost:
        cd, ce, ca = drill * cost["drill"], kg * cost["exp"], n * cost["acc"]
        out.update({"Costo perforación (US$)": cd, "Costo explosivo (US$)": ce, "Costo accesorios (US$)": ca,
                    "Costo total por disparo (US$)": cd + ce + ca,
                    "Costo por tonelada (US$/t)": (cd + ce + ca) / (vol * rho_r) if vol > 0 else np.nan})
        if "Avance por disparo (m)" in out and out["Avance por disparo (m)"]:
            out["Costo por metro de avance (US$/m)"] = (cd + ce + ca) / out["Avance por disparo (m)"]
    return out


PALETTE = ["#d62728", "#ff7f0e", "#e6b800", "#9467bd", "#c51b8a", "#1f77b4", "#17becf", "#8c564b",
           "#e377c2", "#2ca02c", "#7f7f00", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"]


def colors_for(order):
    return {z: PALETTE[i % len(PALETTE)] for i, z in enumerate(order)}
