"""Generadores de cueles (arranques) para frentes de avance, chimeneas y piques.

Cada generador trabaja en el plano del frente (coordenadas locales u, v respecto al centro del
cuele) y devuelve los taladros del cuele con su inclinación lateral, los taladros vacíos, el
polígono de abertura que deja el cuele y el registro de cálculos.

Fuentes: [C] Carpio (2025) cap. 3.2; [LJ] López Jimeno (1987) cap. 22 (Figs. 22.5–22.27).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

CUT_TYPES = [
    "Cilíndrico 4 secciones — Holmberg (Carpio)",
    "Cilíndrico 4 secciones — Holmberg (López Jimeno, con desviación)",
    "Quemado simple",
    "Quemado Sarrois",
    "Quemado Sueco",
    "Fagersta (vacío central 64–76 mm)",
    "Cilíndrico en espiral / doble espiral",
    "Coromant (dos vacíos secantes 57 mm)",
    "Cráter (Livingston, barreno central)",
    "En V / cuña",
    "En abanico",
    "Instantáneo piramidal",
]
PARALLEL_CUTS = CUT_TYPES[:9]
ANGLED_CUTS = CUT_TYPES[9:]
CUT_NOTES = {
    "Cilíndrico 4 secciones — Holmberg (Carpio)": "Ec. 23–46 del libro de Carpio (abertura A_h = √2·X).",
    "Cilíndrico 4 secciones — Holmberg (López Jimeno, con desviación)":
        "Fig. 22.5 y ejemplo F de LJ: la abertura efectiva descuenta la desviación, A_h = √2·(X − E_p).",
    "Quemado simple": "Fig. 22.20 LJ. Todos los taladros del mismo diámetro; avance ≤ 2.5 m.",
    "Quemado Sarrois": "Fig. 22.21 LJ. 8 cargados + 1 vacío Ø38; ejes a 10 cm (dura) – 20 cm (blanda); avance 80–95 %.",
    "Quemado Sueco": "Fig. 22.22 LJ. Filas verticales a 20 cm; laterales cada 30 cm; cargado–vacío 10–15 cm; "
                     "avance 90–100 %; menor consumo; perforación precisa.",
    "Fagersta (vacío central 64–76 mm)": "Fig. 22.14 LJ: mixto entre cuatro secciones y doble espiral; "
                                         "'adecuado para pequeñas galerías con perforación manual'. "
                                         "Geometría aproximada con las cotas a=100, b=150, c=210, d=250 mm.",
    "Cilíndrico en espiral / doble espiral": "Figs. 22.11–22.12 LJ: cargas en espiral alrededor de un vacío de "
                                             "75–200 mm (geometría aproximada).",
    "Coromant (dos vacíos secantes 57 mm)": "Fig. 22.13 LJ: hueco en forma de '8' con plantilla; aquí se usa su "
                                            "diámetro equivalente en la recursión de Holmberg.",
    "Cráter (Livingston, barreno central)": "§4.3 y Fig. 22.23 LJ: carga esférica (L = 6D) a la profundidad óptima "
                                            "D_o en un barreno central de gran diámetro; avance por cráter 4/3·D_o.",
    "En V / cuña": "§4.4-A LJ: H_c = 46D, B = 34D, q_f = 990D², l_f = 0.3L, q_c = 0.5q_f, T = 12D; "
                   "contracuele B = 24D, q_c = 0.4q_f; avance 45–50 % del ancho; B ≤ 0.5L − 0.2.",
    "En abanico": "§4.4-B LJ: B = 23D (B < L − 0.4), H_c = 42D, q_f = 990D², l_f = 0.3L, q_c = 0.4q_f.",
    "Instantáneo piramidal": "§4.4-C LJ: haz cerrado de taladros iniciados simultáneamente; avance ≈ 80 % del ancho; "
                             "gran proyección de escombro.",
}


@dataclass
class CutCtx:
    d1: float            # diámetro de producción (m)
    d2: float            # diámetro de vacío (m)
    NB: int              # n° de vacíos
    H: float             # longitud de perforación efectiva (m)
    F: float             # error de perforación (m)
    c: float             # constante sueca
    rws: float           # RWS del explosivo del cuele
    q_avail: float       # densidad lineal disponible (cartucho) kg/m
    UCS: float
    width: float         # dimensión útil del frente en u (m)
    height: float        # dimensión útil del frente en v (m)
    factor: float = 1.7  # B_max arranque = factor·Ø_eq
    pearse_B: float | None = None  # B_max Pearse si se elige
    n_sections: int | None = None  # None = automático
    max_open: float | None = None  # abertura máxima permitida (labor estrecha)
    spacing: float | None = None   # espaciamiento impuesto en cueles quemados
    vertex_deg: float = 60.0       # ángulo interior de la cuña
    adv_factor: float = 0.475      # avance / ancho en cueles en ángulo
    rho_e: float = 1.13            # g/cm³ (cráter)
    Et: float = 1.5                # factor energía-tensión (cráter)
    delta_o: float = 0.5           # relación óptima de profundidades (cráter)
    d_crater: float = 0.2          # diámetro del barreno central del cráter (m)


@dataclass
class CutResult:
    holes: list = field(default_factory=list)   # dict(zona, u, v, tu, tv, B, S, q, lf, qf, qc, L, T)
    relief: list = field(default_factory=list)  # (u, v, d)
    opening: Polygon | None = None
    sections: list = field(default_factory=list)
    log: list = field(default_factory=list)
    L: float | None = None        # longitud de taladro impuesta por el cuele
    advance: float | None = None  # avance esperado
    q0: float | None = None
    simultaneous: set = field(default_factory=set)
    warnings: list = field(default_factory=list)


def _sq(dist, rot):
    return [(dist * math.cos(math.radians(rot + 90 * k)), dist * math.sin(math.radians(rot + 90 * k))) for k in range(4)]


def _relief_positions(NB, d2):
    if NB == 1:
        return [(0.0, 0.0)]
    if NB == 2:
        r = (d2 + 0.02) / 2
        return [(-r, 0.0), (r, 0.0)]
    r = (d2 + 0.02) / (2 * math.sin(math.pi / NB))
    return [(r * math.cos(math.pi / 2 + 2 * math.pi * k / NB), r * math.sin(math.pi / 2 + 2 * math.pi * k / NB))
            for k in range(NB)]


def burn_spacing(UCS):
    """Distancia entre ejes en cueles quemados Ø38 mm: 10 cm (roca dura) a 20 cm (blanda) — [LJ] cuele Sarrois."""
    if UCS > 180:
        return 0.10
    if UCS > 120:
        return 0.12
    if UCS > 70:
        return 0.15
    return 0.20


def q0_holmberg(d1, Bmax, deq, c, rws):
    return 55 * d1 * (Bmax / deq) ** 1.5 * (Bmax - deq / 2) * (c / 0.4) / rws


def _hull(points, pad=0.0):
    hull = MultiPoint([Point(p) for p in points]).convex_hull
    return hull.buffer(pad, join_style=2) if pad else hull


# ----------------------------------------------------------------------------
def cylindrical(ctx: CutCtx, variant="Carpio", deq_override=None, max_auto=4):
    r = CutResult()
    deq = deq_override if deq_override else ctx.d2 * math.sqrt(ctx.NB)
    lg = r.log
    lg.append(("Diámetro equivalente de alivio", "Ø_equiv", deq, "m", "(25) Ø2·√NB"))
    Lmax = 0.15 + 34.1 * deq - 39.4 * deq ** 2
    lg.append(("Longitud máxima alcanzable", "L", Lmax, "m", "(24) 0.15 + 34.1·Ø − 39.4·Ø²"))
    if ctx.H > Lmax:
        r.warnings.append(f"H' = {ctx.H:.2f} m supera la longitud alcanzable con el alivio (L = {Lmax:.2f} m).")
    if ctx.pearse_B:
        Bmax0, eq = ctx.pearse_B, "Pearse: Kv·Ø1/1000·√(PD/Std)"
    else:
        Bmax0, eq = ctx.factor * deq, f"(28–29) {ctx.factor}·Ø_equiv"
    lg.append(("Burden máximo del arranque", "B_max", Bmax0, "m", eq))
    Bp0 = Bmax0 - ctx.F
    lg.append(("Burden práctico del arranque", "B_p", Bp0, "m", "(30) B_max − F"))
    if Bp0 <= deq / 2 + ctx.d1 / 2:
        r.warnings.append("El burden práctico del arranque casi toca el alivio: aumente Ø2/NB o mejore la precisión (F).")
    q0 = q0_holmberg(ctx.d1, Bmax0, deq, ctx.c, ctx.rws)
    lg.append(("Densidad de carga lineal del arranque", "q₀", q0, "kg/m", "(26)"))
    r.q0 = q0
    secs = []
    sqrtH = math.sqrt(ctx.H)
    max_open = ctx.max_open or 0.8 * min(ctx.width, ctx.height)
    if variant == "Carpio":
        Ah = math.sqrt(2) * Bp0
        lg.append(("Ancho de abertura del arranque", "A_h0", Ah, "m", "(33) √2·B_p"))
        secs.append(dict(name="Arranque", dist=Bp0, rot=45.0, B=Bp0, Ah=Ah, q=q0))
        nmax = ctx.n_sections if ctx.n_sections is not None else max_auto - 1
        rot = 0.0
        for N in range(1, nmax + 1):
            Bm = min(8.8e-2 * math.sqrt(Ah * q0 * ctx.rws / (ctx.d1 * ctx.c)), 2 * Ah)
            Bp = Bm - ctx.F
            X = Ah / 2 + Bp
            Ahn = math.sqrt(2) * X
            if ctx.n_sections is None and N > 1 and Ahn > 1.05 * sqrtH:
                break
            if Ahn > max_open:
                r.warnings.append(f"El cuadrante {N} no cabe en el frente (A_h = {Ahn:.2f} m > {max_open:.2f} m); "
                                  f"el cuele se limita a {N - 1} cuadrante(s).")
                break
            lg += [(f"Burden máximo cuadrante {N}", f"B_max{N}", Bm, "m", "(34/38/42)"),
                   (f"Burden práctico cuadrante {N}", f"B_p{N}", Bp, "m", "(35/39/43)"),
                   (f"Ancho de abertura cuadrante {N}", f"A_h{N}", Ahn, "m", "(37/41/45) √2·(A_h/2 + B_p)")]
            secs.append(dict(name=f"Cuadrante {N}", dist=X, rot=rot, B=Bp, Ah=Ahn, q=q0))
            rot = 45.0 - rot
            Ah = Ahn
        lg.append(("Verificación A_h último ≤ √H'", "√H'", sqrtH, "m",
                   "cumple" if Ah <= 1.05 * sqrtH else "A_h > √H'"))
    else:  # López Jimeno: descuenta la desviación en la abertura efectiva
        Ep = ctx.F
        B1 = Bmax0 - Ep
        Ahp = math.sqrt(2) * B1
        Ah_eff = math.sqrt(2) * (B1 - Ep)
        lg.append(("Abertura nominal 1ª sección", "A_h'", Ahp, "m", "√2·B1"))
        lg.append(("Abertura efectiva (con desviación)", "A_h", Ah_eff, "m", "√2·(B1 − E_p)"))
        secs.append(dict(name="Arranque", dist=B1, rot=45.0, B=B1, Ah=Ahp, q=q0))
        nmax = ctx.n_sections if ctx.n_sections is not None else max_auto - 1
        rot, X = 0.0, B1
        for N in range(1, nmax + 1):
            if Ah_eff <= 0.01:
                r.warnings.append("La desviación anula la abertura efectiva: reduzca F o aumente el alivio.")
                break
            q = ctx.q_avail
            Bm = 8.8e-2 * math.sqrt(Ah_eff * q * ctx.rws / (ctx.d1 * ctx.c))
            note = "8.8·10⁻²·√(A_h·q·PRP/(Ø1·c))"
            if Bm - Ep > 2 * Ah_eff:  # deformación plástica: se reduce la carga
                q = 540 * ctx.d1 * ctx.c * Ah_eff / ctx.rws
                Bm = 8.8e-2 * math.sqrt(Ah_eff * q * ctx.rws / (ctx.d1 * ctx.c))
                note = "B > 2A_h → q = 540·Ø1·c·A_h/PRP"
            B2 = Bm - Ep
            Xn = Ahp / 2 + B2
            Ahpn = math.sqrt(2) * Xn
            if ctx.n_sections is None and N > 1 and Ahp >= math.sqrt(ctx.H):
                break
            if Ahpn > max_open:
                r.warnings.append(f"La sección {N + 1} no cabe en el frente; el cuele se limita a {N} sección(es).")
                break
            if B2 < 0.5 * Ah_eff:
                r.warnings.append(f"Sección {N + 1}: B < 0.5·A_h, el cuele pierde su carácter de cuatro secciones.")
            lg += [(f"Burden sección {N + 1}", f"B{N + 1}", B2, "m", note + " − E_p"),
                   (f"Carga lineal sección {N + 1}", f"q{N + 1}", q, "kg/m", ""),
                   (f"Abertura nominal sección {N + 1}", f"A_h'{N + 1}", Ahpn, "m", "√2·(B + A_h'/2)")]
            secs.append(dict(name=f"Cuadrante {N}", dist=Xn, rot=rot, B=B2, Ah=Ahpn, q=q))
            Ah_eff = math.sqrt(2) * (Xn - Ep)
            Ahp, X = Ahpn, Xn
            rot = 45.0 - rot
    pts_all = []
    for s in secs:
        for (u, v) in _sq(s["dist"], s["rot"]):
            r.holes.append(dict(zona=s["name"], u=u, v=v, tu=0.0, tv=0.0, B=s["B"], S=s["Ah"], q=s["q"]))
            pts_all.append((u, v))
    r.relief = [(u, v, ctx.d2) for u, v in _relief_positions(ctx.NB, ctx.d2)] if deq_override is None else r.relief
    last = secs[-1]
    r.opening = Polygon(_sq(last["dist"], last["rot"]))
    r.sections = secs
    return r


def burn(ctx: CutCtx, kind):
    r = CutResult()
    a = ctx.spacing or burn_spacing(ctx.UCS)
    d = ctx.d1
    q = ctx.q_avail
    if kind == "simple":
        nv = max(1, min(ctx.NB, 3))
        rel = _relief_positions(nv, ctx.d2)
        ext = max(abs(u) for u, _ in rel) + ctx.d2 / 2
        dist = max(a, ext + d / 2 + 0.05)
        pts = [(dist, 0), (-dist, 0), (0, dist), (0, -dist)]
        names = ["Arranque"] * 4
        r.relief = [(u, v, ctx.d2) for u, v in rel]
        r.log.append(("Distancia vacío–cargado", "a", dist, "m", "10–20 cm según dureza (LJ)"))
    elif kind == "sarrois":
        pts, names = [], []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    continue
                pts.append((i * a, j * a))
                names.append("Arranque" if (i == 0 or j == 0) else "Cuadrante 1")
        r.relief = [(0.0, 0.0, ctx.d1)]
        r.log.append(("Distancia entre ejes", "a", a, "m", "10 cm (dura) – 20 cm (blanda), Ø38 (Fig. 22.21)"))
    elif kind == "sueco":
        av = 0.10 if ctx.UCS > 120 else 0.125 if ctx.UCS > 70 else 0.15
        pts, names = [], []
        for x in (-0.20, 0.20):
            for y in (0.30, 0.0, -0.30):
                pts.append((x, y))
                names.append("Cuadrante 1")
        for y in (0.30, 0.0, -0.30):
            pts.append((0.0, y))
            names.append("Arranque")
        r.relief = [(0.0, av, ctx.d1), (0.0, -av, ctx.d1)]
        r.log += [("Distancia entre filas verticales", "—", 0.20, "m", "Fig. 22.22"),
                  ("Separación vertical de laterales", "—", 0.30, "m", "Fig. 22.22"),
                  ("Distancia cargado–vacío", "—", av, "m", "10–15 cm según dureza")]
    elif kind == "fagersta":
        A, Bd, Cd, Dd = 0.10, 0.15, 0.21, 0.25
        pts = [(0, A), (0, -A), (-A, 0), (Cd / math.sqrt(2), Cd / math.sqrt(2)), (Bd, Bd + 0.10),
               (-Cd / math.sqrt(2), -Cd / math.sqrt(2)), (-Bd, Bd + 0.10), (Dd / math.sqrt(2), -Dd / math.sqrt(2)),
               (Bd, 0.42), (-Bd, -0.40)]
        names = ["Arranque"] * 3 + ["Cuadrante 1"] * 3 + ["Cuadrante 2"] * 4
        dc = max(ctx.d2, 0.064)
        r.relief = [(0.0, 0.0, dc)]
        r.log.append(("Vacío central", "Ø", dc, "m", "64–76 mm (Fig. 22.14)"))
    elif kind == "espiral":
        dc = max(ctx.d2, 0.075)
        r.relief = [(0.0, 0.0, dc)]
        pts, names = [], []
        r0 = 1.5 * dc
        for k in range(8):
            rad = r0 * (1 + 0.28 * k)
            ang = math.radians(-90 + 110 * k)
            pts.append((rad * math.cos(ang), rad * math.sin(ang)))
            names.append("Arranque" if k < 2 else "Cuadrante 1" if k < 4 else "Cuadrante 2")
        r.log.append(("Radio inicial de la espiral", "1.5·Ø", r0, "m", "Figs. 22.11–22.12 (aprox.)"))
    else:
        raise ValueError(kind)
    for (u, v), z in zip(pts, names):
        r.holes.append(dict(zona=z, u=u, v=v, tu=0.0, tv=0.0, B=a, S=a, q=q))
    r.opening = _hull(pts, 0.02)
    r.q0 = q
    if ctx.H > 2.5:
        r.warnings.append("Los cueles quemados no suelen superar 2.5 m por disparo (LJ §4.2).")
    return r


def coromant(ctx: CutCtx):
    s = 0.8 * 0.057
    rr = 0.057 / 2
    # área de la unión de dos círculos secantes → diámetro equivalente
    th = 2 * math.acos(s / (2 * rr))
    lens = rr ** 2 * (th - math.sin(th))
    area = 2 * math.pi * rr ** 2 - lens
    deq = math.sqrt(4 * area / math.pi)
    r = cylindrical(ctx, "López Jimeno", deq_override=deq, max_auto=3)
    r.relief = [(0.0, s / 2, 0.057), (0.0, -s / 2, 0.057)]
    r.log.insert(0, ("Diámetro equivalente del '8'", "Ø_eq", deq, "m", "área de dos vacíos secantes de 57 mm"))
    return r


def crater(ctx: CutCtx):
    r = CutResult()
    D = ctx.d_crater
    Q = 1.5 * math.pi * D ** 3 * ctx.rho_e * 1000
    Dc = ctx.Et * Q ** (1 / 3)
    Do = ctx.delta_o * Dc
    X = 4 / 3 * Do
    n = max(1, min(3, int(ctx.H // X)))
    if X > ctx.H:
        r.warnings.append(f"El avance de un cráter (4/3·D_o = {X:.2f} m) supera H' = {ctx.H:.2f} m: "
                          "reduzca el diámetro del barreno central o alargue la perforación.")
    r.log += [("Carga esférica (L = 6D)", "Q", Q, "kg", "3πD³ρe/2"),
              ("Profundidad crítica", "D_c", Dc, "m", "E_t·Q^(1/3) (Livingston)"),
              ("Profundidad óptima", "D_o", Do, "m", "Δ_o·D_c"),
              ("Avance por cráter", "X", X, "m", "4/3·D_o (Tabla 24.2)"),
              ("N° de cráteres en el barreno", "n", n, "", "seccionado y retacado")]
    r.holes.append(dict(zona="Arranque", u=0.0, v=0.0, tu=0.0, tv=0.0, B=Do, S=Do, q=Q / (6 * D),
                        decks=[(k * X + Do - 3 * D, k * X + Do + 3 * D) for k in range(n)], d_mm=D * 1000))
    rad = max(0.6 * Do, 0.35)
    for k in range(6):
        ang = math.radians(60 * k + 30)
        r.holes.append(dict(zona="Cuadrante 1", u=rad * math.cos(ang), v=rad * math.sin(ang), tu=0, tv=0,
                            B=rad, S=rad, q=ctx.q_avail))
    r.opening = Point(0, 0).buffer(rad * 1.05, 8)
    r.L = max(min(n * X, ctx.H), Do + 3 * D + 0.1)
    r.advance = min(n * X, ctx.H)
    r.q0 = ctx.q_avail
    return r


def angled(ctx: CutCtx, kind):
    r = CutResult()
    D = ctx.d1
    alpha = math.radians(ctx.vertex_deg / 2)  # ángulo de cada taladro con el eje
    W = ctx.width
    qf = 990 * D ** 2
    T = 12 * D
    if kind == "piramidal":
        X = min(0.8 * W, ctx.H)  # avance ≈ 80 % del ancho
        L = X / math.cos(alpha)
        half = X * math.tan(alpha)
        if half > 0.45 * W:
            alpha = math.atan(0.45 * W / X)
            L = X / math.cos(alpha)
            half = X * math.tan(alpha)
        r.log += [("Avance esperado", "X", X, "m", "≈ 80 % del ancho (LJ §4.4-C)"),
                  ("Longitud de taladro", "L", L, "m", "X / cos α")]
        for (u, v) in _sq(half * math.sqrt(2), 45):
            nrm = math.hypot(u, v)
            s = math.sin(alpha)
            r.holes.append(dict(zona="Arranque", u=u, v=v, tu=-u / nrm * s, tv=-v / nrm * s, B=half, S=half,
                                q=0.5 * qf, lf=0.3 * L, qf=qf, qc=0.5 * qf, T=T, L=L))
        r.simultaneous.add("Arranque")
        r.opening = Polygon(_sq(half * math.sqrt(2), 45)).buffer(0.05)
        r.L, r.advance = L, X
        r.warnings.append("Cuele instantáneo: gran proyección de escombro; prevea la distancia de seguridad.")
        return r
    X = min(ctx.adv_factor * W, ctx.H)
    L = X / math.cos(alpha) + 0.10
    gap = 0.10
    half = X * math.tan(alpha) + gap / 2
    if half > 0.48 * W:
        alpha = math.atan((0.48 * W - gap / 2) / X)
        L = X / math.cos(alpha) + 0.10
        half = 0.48 * W
        r.warnings.append("El ángulo de la cuña se cerró para que los emboquilles quepan en el ancho; "
                          "si el ángulo interior baja de 60° las cargas quedan muy confinadas.")
    Hc = min((46 if kind == "V" else 42) * D, 0.7 * ctx.height)
    Bc = (34 if kind == "V" else 23) * D
    r.log += [("Avance esperado", "X", X, "m", f"{ctx.adv_factor:.2f}·ancho (45–50 %, LJ)"),
              ("Longitud de taladro del cuele", "L", L, "m", "X / cos α"),
              ("Altura total del cuele", "H_c", Hc, "m", f"{46 if kind == 'V' else 42}·D"),
              ("Piedra del cuele", "B", Bc, "m", f"{34 if kind == 'V' else 23}·D"),
              ("Carga de fondo", "q_f", qf, "kg/m", "990·D²"),
              ("Longitud de carga de fondo", "l_f", 0.3 * L, "m", "0.3·L"),
              ("Retacado", "T", T, "m", "12·D")]
    lim = 0.5 * L - 0.2 if kind == "V" else L - 0.4
    if Bc > lim:
        r.warnings.append(f"B = {Bc:.2f} m no cumple B ≤ {'0.5L − 0.2' if kind == 'V' else 'L − 0.4'} "
                          f"= {lim:.2f} m: en voladuras cortas la piedra debe reducirse.")
        Bc = max(lim, 0.25)
    nrow = 3
    vs = np.linspace(-Hc / 2 + Hc / (2 * nrow), Hc / 2 - Hc / (2 * nrow), nrow)
    s = math.sin(alpha)
    if kind == "V":
        for v in vs:
            for sgn in (-1, 1):
                r.holes.append(dict(zona="Arranque", u=sgn * half, v=v, tu=-sgn * s, tv=0.0, B=Bc, S=Hc / nrow,
                                    q=0.5 * qf, lf=0.3 * L, qf=qf, qc=0.5 * qf, T=T, L=L))
        # contracuele
        a2 = alpha * 0.55
        u2 = half + min(24 * D, 0.45 * W - half) if half < 0.45 * W else None
        if u2 is not None and u2 > half + 0.15:
            L2 = X / math.cos(a2)
            for v in vs:
                for sgn in (-1, 1):
                    r.holes.append(dict(zona="Cuadrante 1", u=sgn * u2, v=v, tu=-sgn * math.sin(a2), tv=0.0,
                                        B=24 * D, S=Hc / nrow, q=0.4 * qf, lf=0.3 * L2, qf=qf, qc=0.4 * qf, T=T,
                                        L=L2))
            r.log.append(("Piedra del contracuele", "B", 24 * D, "m", "24·D"))
            ext = u2
        else:
            ext = half
    else:  # abanico horizontal desde el hastial izquierdo
        u0 = -half
        angs = np.radians(np.linspace(15, math.degrees(alpha) * 1.6, 4))
        for v in vs:
            for k, a in enumerate(angs):
                Lk = min(X / math.cos(a), L * 1.15)
                r.holes.append(dict(zona="Arranque" if k < 2 else "Cuadrante 1", u=u0 + 0.06 * k, v=v,
                                    tu=math.sin(a), tv=0.0, B=Bc, S=Hc / nrow, q=0.4 * qf, lf=0.3 * Lk, qf=qf,
                                    qc=0.4 * qf, T=T, L=Lk))
        ext = half
        r.warnings.append("El cuele en abanico está prácticamente en desuso por su complejidad de perforación (LJ).")
    r.opening = Polygon([(-ext, -Hc / 2), (ext, -Hc / 2), (ext, Hc / 2), (-ext, Hc / 2)])
    r.L, r.advance = L, X
    return r


def make_cut(cut_type: str, ctx: CutCtx) -> CutResult:
    if cut_type.startswith("Cilíndrico 4 secciones — Holmberg (Carpio)"):
        return cylindrical(ctx, "Carpio")
    if cut_type.startswith("Cilíndrico 4 secciones — Holmberg (López"):
        return cylindrical(ctx, "López Jimeno")
    if cut_type == "Quemado simple":
        return burn(ctx, "simple")
    if cut_type == "Quemado Sarrois":
        return burn(ctx, "sarrois")
    if cut_type == "Quemado Sueco":
        return burn(ctx, "sueco")
    if cut_type.startswith("Fagersta"):
        return burn(ctx, "fagersta")
    if cut_type.startswith("Cilíndrico en espiral"):
        return burn(ctx, "espiral")
    if cut_type.startswith("Coromant"):
        return coromant(ctx)
    if cut_type.startswith("Cráter"):
        return crater(ctx)
    if cut_type == "En V / cuña":
        return angled(ctx, "V")
    if cut_type == "En abanico":
        return angled(ctx, "abanico")
    if cut_type == "Instantáneo piramidal":
        return angled(ctx, "piramidal")
    raise ValueError(cut_type)
