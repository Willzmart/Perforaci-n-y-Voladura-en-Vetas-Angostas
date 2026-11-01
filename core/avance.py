"""Labores de avance horizontal: método Holmberg (Carpio cap. 3.2 / López Jimeno cap. 22) con
selección de cuele, método de carga y modo para labores estrechas."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry.polygon import orient

from .common import (Explosive, Log, Result, Rock, bmax_holmberg, colors_for, dt_y_C, factor_carga_tabla,
                     finalize_holes, new_hole, relief_row, summary_common)
from .cuts import ANGLED_CUTS, CUT_NOTES, CutCtx, make_cut
from .facelayout import fill_helpers, sample_chain, tunnel_chains

ZONES = ["Arranque", "Cuadrante 1", "Cuadrante 2", "Cuadrante 3", "Cuadrante 4", "Ayuda de hastiales",
         "Ayuda de corona", "Ayuda de arrastre", "Hastiales", "Corona", "Arrastre"]
GROUP = {"Arranque": 0, "Cuadrante 1": 1, "Cuadrante 2": 2, "Cuadrante 3": 3, "Cuadrante 4": 4,
         "Ayuda de hastiales": 5, "Ayuda de corona": 5, "Ayuda de arrastre": 5, "Hastiales": 6, "Corona": 7,
         "Arrastre": 8}
CUT_Z = {"Arranque", "Cuadrante 1", "Cuadrante 2", "Cuadrante 3", "Cuadrante 4"}
CONTOUR_Z = {"Hastiales", "Corona", "Arrastre"}
CHARGE_METHODS = ["Carpio — q₀ en todas las zonas (cap. 3.2)",
                  "Carpio — segundo método con factores (cap. 3.4)",
                  "López Jimeno — carga de fondo y columna (cap. 22)"]
FACTORS_C2 = {"Arranque": 1.0, "Cuadrante 1": 0.95, "Cuadrante 2": 0.90, "Cuadrante 3": 0.85,
              "Cuadrante 4": 0.85, "Ayuda de hastiales": 0.80, "Ayuda de corona": 0.80, "Ayuda de arrastre": 0.80,
              "Hastiales": 0.65, "Arrastre": 0.98}


@dataclass
class Drill:
    d1_mm: float = 45
    d2_mm: float = 102
    NB: int = 2
    bar_m: float = 3.35
    eff: float = 0.95
    F_method: str = "Dato directo"
    F_value: float = 0.04
    alpha_mm_m: float = 10.0
    e_mm: float = 20.0

    def F(self, Hp):
        if self.F_method == "Ec. 26":
            return self.alpha_mm_m / 1000 * Hp + self.e_mm / 1000, "(26) α·H' + e"
        if self.F_method == "Ec. 27":
            return 0.0013 * Hp ** 2 + 0.078 * Hp + 0.0014, "(27)"
        return self.F_value, "dato"


@dataclass
class AvanceOptions:
    cut_type: str = "Cilíndrico 4 secciones — Holmberg (Carpio)"
    cut_method: str = "Holmberg"     # B_max arranque: Holmberg | Pearse
    cut_factor: float = 1.7
    n_quadrants: int | None = None
    burn_spacing: float | None = None
    vertex_deg: float = 60.0
    adv_factor: float = 0.475
    d_crater_mm: float = 200.0
    Et: float = 1.5
    delta_o: float = 0.5
    gamma_deg: float = 3.0
    K_corona: float = 15.0
    contour_mode: str = "Voladura controlada (S = K·Ø1, q = 90·Ø1²)"
    c_hat_rule: str = "Iterativa"
    contour_offset: float = 0.10
    cut_height: float | None = None
    cut_x: float = 0.0               # desplazamiento horizontal del cuele (ubicarlo en la veta)
    blast_eff: float = 0.95
    charge_method: str = CHARGE_METHODS[0]
    narrow: bool = False
    B_prod_override: float | None = None
    SB_prod: float = 1.25


def run_avance(section, rock: Rock, drill: Drill, exm: Explosive, exc: Explosive, o: AvanceOptions,
               cost=None) -> Result:
    res = Result(labor="Avance", metodo=o.cut_type)
    lg = res.log
    P = res.p
    s = section
    A, Ht = s.width, s.height
    d1, d2 = drill.d1_mm / 1000, drill.d2_mm / 1000
    F0 = "1. Parámetros generales"
    lg.add(F0, "Área de la sección", "S", s.area, "m²")
    lg.add(F0, "Perímetro de la sección", "P", s.perimeter, "m")
    Hp = lg.add(F0, "Longitud de perforación efectiva", "H'", drill.bar_m * drill.eff, "m", "(23) L_barra·η")
    F, eqF = drill.F(Hp)
    lg.add(F0, "Error de perforación", "F", F, "m", eqF)
    lg.add(F0, "Factor de volabilidad", "K_v", rock.Kv, "", "1.96 − 0.27·ln(RQD·JSF)")
    lg.add(F0, "Presión de detonación", "PD", exm.PD, "MPa", "ρe·10⁻⁶·VOD²/4")
    lg.add(F0, "Resistencia a tracción dinámica", "S_td", rock.Std, "MPa", "4.5·10.197·0.08·σci")
    lg.add(F0, "Consumo específico", "C.E", rock.CE, "kg/m³", "0.56·ρr·tan(5+RMR/2)/∛((115−RQD)/3.3)")
    c = lg.add(F0, "Constante sueca de roca", "c", rock.c, "kg/m³", "0.8784·C.E + 0.0052")
    lg.add(F0, "Impedancia explosivo/roca", "n_z", exm.rho * exm.vod / (rock.rho * rock.Vp), "",
           "ρe·VOD/(ρr·Vp)")
    Lt = lg.add(F0, "Longitud de taco", "Lt", 10 * d1, "m", "(31) 10·Ø1")
    dt, Cr = dt_y_C(rock.RMR)
    Npred = s.perimeter / dt + Cr * s.area
    lg.add(F0, "Predimensionamiento de taladros", "N°t", Npred, "", f"P/dt + C·S (dt={dt}, C={Cr}; Tablas 06–07)")
    q_avail = exm.q_lineal(d1)

    # ---- Cuele ----
    F2 = "2. Cuele: " + o.cut_type
    margin = o.contour_offset + 0.15
    max_open = (0.55 if o.narrow else 0.8) * min(A, Ht)
    pearse = rock.Kv * drill.d1_mm / 1000 * math.sqrt(exm.PD / rock.Std) if o.cut_method == "Pearse" else None
    ctx = CutCtx(d1=d1, d2=d2, NB=drill.NB, H=Hp, F=F, c=c, rws=exm.rws, q_avail=q_avail, UCS=rock.UCS,
                 width=(A - 2 * o.contour_offset) if o.cut_type in ANGLED_CUTS else A - 2 * margin,
                 height=Ht - 2 * margin, factor=o.cut_factor, pearse_B=pearse,
                 n_sections=o.n_quadrants, max_open=max_open, spacing=o.burn_spacing, vertex_deg=o.vertex_deg,
                 adv_factor=o.adv_factor, rho_e=exm.rho, Et=o.Et, delta_o=o.delta_o, d_crater=o.d_crater_mm / 1000)
    cut = make_cut(o.cut_type, ctx)
    for row in cut.log:
        lg.add(F2, *row)
    res.warnings += cut.warnings
    q0 = cut.q0 or q_avail
    if not o.cut_type.startswith("Cilíndrico 4 secciones — Holmberg (Carpio)"):
        q0 = q_avail  # LJ: la destroza usa la concentración del cartucho disponible
    angled = o.cut_type in ANGLED_CUTS or cut.L is not None
    if angled:
        adv = cut.advance
        L_face = min(Hp, adv + 0.10)
    else:
        L_face = Hp
        adv = Hp * o.blast_eff
    Lc = L_face - Lt
    lg.add(F2, "Longitud de carga (taladros del frente)", "Lc", Lc, "m", "(32) H' − Lt")
    P["cut_note"] = CUT_NOTES.get(o.cut_type, "")

    # ---- Tajeo (Holmberg) ----
    gs = math.sin(math.radians(o.gamma_deg))
    zp = {}
    for fase, z, f, sb in [("4. Ayudas", "Ayuda de hastiales", 1.45, 1.25), ("4. Ayudas", "Ayuda de corona", 1.20, 1.25),
                           ("4. Ayudas", "Ayuda de arrastre", 1.45, 1.25), ("5. Contorno", "Hastiales", 1.45, 1.25),
                           ("5. Contorno", "Arrastre", 1.45, 1.00), ("5. Contorno", "Contorno sin control", 1.20, 1.25)]:
        Bm, ch = bmax_holmberg(q0, exm.rws, c, f, sb, o.c_hat_rule)
        zp[z] = Bm
        if z != "Contorno sin control" or not o.contour_mode.startswith("Voladura controlada"):
            lg.add(fase, f"Burden máximo — {z.lower()}", "B_max", Bm, "m", f"0.9·√(q·RWS/(Ĉ·{f}·{sb})), Ĉ={ch:.3f}")
    Ah = cut.opening.bounds[2] - cut.opening.bounds[0]
    lg.add("4. Ayudas", "Burden práctico ayuda hastiales (libro)", "B_p", (A - Ah) / 4, "m", "(49) (A − A_h)/4")
    lg.add("4. Ayudas", "Altura disponible hacia abajo", "H_disp(↓)", 0.64 * (Ht - Ah), "m", "(53)")
    lg.add("4. Ayudas", "Altura disponible hacia arriba", "H_disp(↑)", 0.36 * (Ht - Ah), "m", "(60)")
    Bp_h = zp["Hastiales"] - L_face * gs - F
    controlled = o.contour_mode.startswith("Voladura controlada")
    if controlled:
        S_cor = lg.add("5. Contorno", "Espaciamiento de corona", "S", o.K_corona * d1, "m", "(71) K·Ø1")
        Bm_cor = S_cor / 0.8
        Bp_cor = lg.add("5. Contorno", "Burden práctico de corona", "B_pract", Bm_cor - L_face * gs - F, "m",
                        "(72–73) S/0.8 − H'·senγ − F")
        q_cor = lg.add("5. Contorno", "Carga de corona", "q_corona", 90 * d1 ** 2, "kg/m", "(75) 90·Ø1²")
    else:
        Bp_cor = zp["Contorno sin control"] - L_face * gs - F
        S_cor = 1.25 * Bp_cor
        q_cor = None
        lg.add("5. Contorno", "Burden de contorno sin voladura controlada", "B", Bp_cor, "m",
               "f = 1.2, S/B = 1.25 (LJ §4.1-E)")
    lg.add("5. Contorno", "Burden práctico de hastiales", "B_p", Bp_h, "m", "B_max − H'·senγ − F")
    Bm_arr = zp["Arrastre"]
    Bp_arr = lg.add("5. Contorno", "Burden práctico de arrastre", "B_pract", Bm_arr - L_face * gs - F, "m", "(79)")
    N_arr = int((s.floor_width + 2 * L_face * gs) / Bm_arr + 2)
    lg.add("5. Contorno", "N° taladros de arrastre", "N°", N_arr, "", "(80)")
    Bp_limit = zp["Ayuda de hastiales"] - F
    Bprod = Bp_limit
    if o.B_prod_override:
        if o.B_prod_override > Bp_limit:
            res.warnings.append(f"El burden impuesto ({o.B_prod_override:.2f} m) supera el límite de Holmberg "
                                f"({Bp_limit:.2f} m); se usa el límite.")
        else:
            Bprod = o.B_prod_override
        lg.add("6. Predicción P80", "Burden de producción impuesto", "B_prod", Bprod, "m", "optimización por P80")
    P.update(Hp=Hp, F=F, c=c, q0=q0, Lt=Lt, Lc=Lc, L_face=L_face, adv=adv, Bp_limit=Bp_limit, Bp_prod=Bprod,
             d1=d1, d2=d2, Npred=Npred, gs=gs)

    # ---- Geometría ----
    Ahl = max(cut.opening.bounds[3] - cut.opening.bounds[1], 0.2)
    yc = o.cut_height if o.cut_height else 0.36 * (Ht - Ahl) + Ahl / 2
    yc = float(np.clip(yc, Ahl / 2 + 0.25, Ht - Ahl / 2 - 0.25)) if Ht > Ahl + 0.5 else Ht / 2
    half_free = A / 2 - (cut.opening.bounds[2] - cut.opening.bounds[0]) / 2 - 0.25
    xc = float(np.clip(o.cut_x, -max(half_free, 0.0), max(half_free, 0.0)))
    P["cut_center"] = (xc, yc)
    from shapely import affinity
    opening = affinity.translate(cut.opening, xc, yc)
    rows = []
    gam = math.radians(o.gamma_deg)

    def charges(zona, L, B, qz=None, spec=None):
        """Devuelve (Lt, cargas[(s0,s1,q)], q_ref)."""
        if spec and "qf" in spec:  # cuele en ángulo con fondo/columna propios
            T = spec["T"]
            lf = spec["lf"]
            return T, [(T, L - lf, spec["qc"]), (L - lf, L, spec["qf"])], spec["qf"]
        if spec and "decks" in spec:
            return 0.0, [(a, b, spec["q"]) for a, b in spec["decks"]], spec["q"]
        if zona == "Corona" and controlled:
            return Lt, [(Lt, L, q_cor)], q_cor
        m = o.charge_method
        if m.startswith("Carpio — q₀"):
            q = qz if (qz and zona in CUT_Z) else q0
            return Lt, [(Lt, L, q)], q
        if m.startswith("Carpio — segundo"):
            dcar = d1 if exm.granel else _cart_diam(exm)
            base = exm.rho * 1000 * math.pi * dcar ** 2 / 4
            fz = FACTORS_C2.get(zona, 0.65)
            return Lt, [(Lt, L, base * fz)], base * fz
        # López Jimeno: fondo + columna
        qf = qz if (qz and zona in CUT_Z) else q_avail
        if zona in CUT_Z:
            return Lt, [(Lt, L, qf)], qf
        lf = min(1.25 * B if B == B else 0.4 * L, 0.6 * (L - Lt))
        ratio = 0.7 if zona == "Arrastre" else 0.5
        return Lt, [(Lt, L - lf, ratio * qf), (L - lf, L, qf)], qf

    def add_face_hole(zona, x, y, ux_lat, uz_lat, B, S, qz=None, spec=None, Lh=None, d_mm=None):
        L = Lh if Lh else L_face
        T, cargas, qref = charges(zona, L, B, qz, spec)
        uy = math.sqrt(max(1e-6, 1 - ux_lat ** 2 - uz_lat ** 2))
        ex = exc if (zona == "Corona" and controlled) else exm
        rows.append(new_hole(zona, GROUP[zona], x, y, x, 0.0, y, (ux_lat, uy, uz_lat), L, T, cargas,
                             d_mm or drill.d1_mm, ex.label(), B, S, qref, cuele=zona in CUT_Z,
                             w_cart=ex.w_cart))

    for h in cut.holes:
        add_face_hole(h["zona"], xc + h["u"], yc + h["v"], h.get("tu", 0.0), h.get("tv", 0.0), h.get("B"),
                      h.get("S"), h.get("q"), spec=h if ("qf" in h or "decks" in h) else None, Lh=h.get("L"),
                      d_mm=h.get("d_mm"))
    rel = [relief_row(xc + u, yc + v, dd * 1000, xc + u, 0.0, yc + v, (0, 1, 0), L_face + 0.1)
           for (u, v, dd) in cut.relief]
    res.relief = pd.DataFrame(rel, columns=["x", "y", "d_mm", "X0", "Y0", "Z0", "ux", "uy", "uz", "L"])
    res.sources = [(r["x"], r["y"]) for r in rel] if rel else [(xc, yc)]

    # contorno
    off = o.contour_offset
    cpoly = orient(s.polygon.buffer(-off, join_style=2), 1.0) if off > 0 else s.polygon
    spring_c = min(s.springline, cpoly.bounds[3] - 0.02)
    chains = tunnel_chains(cpoly, spring_c)
    contour_pts = []
    if "corona" in chains:
        ch = chains["corona"]
        n = max(2, math.ceil(ch.length / S_cor) + 1)
        for x, y, tx, ty in sample_chain(ch, n, True):
            add_face_hole("Corona", x, y, ty * math.sin(gam), -tx * math.sin(gam), Bp_cor, ch.length / (n - 1))
            contour_pts.append((x, y))
    if "piso" in chains:
        ch = chains["piso"]
        n = max(3, N_arr) if not o.narrow else max(2, min(N_arr, math.ceil(ch.length / max(Bp_arr, 0.3)) + 1))
        for x, y, tx, ty in sample_chain(ch, n, True):
            add_face_hole("Arrastre", x, y, 0.0, -math.sin(gam), Bp_arr, ch.length / max(n - 1, 1))
            contour_pts.append((x, y))
    S_h = 1.25 * max(Bp_h, 0.3)
    for side in ("der", "izq"):
        if side in chains:
            ch = chains[side]
            n_int = max(0, math.ceil(ch.length / S_h) - 1)
            for x, y, tx, ty in sample_chain(ch, n_int, False):
                add_face_hole("Hastiales", x, y, ty * math.sin(gam), -tx * math.sin(gam), Bp_h,
                              ch.length / (n_int + 1))
                contour_pts.append((x, y))
    existing = [(r["x"], r["y"]) for r in rows]
    d_ring = (max(Bp_h, 0.25) + max(Bp_cor, 0.25)) / 2
    helpers = fill_helpers(s.polygon, cpoly, opening, existing, [(r["x"], r["y"]) for r in rel], Bprod,
                           o.SB_prod, d_ring, floor_cut=max(Bp_arr, 0.25) + off, center=(xc, yc), mirror_axis=0.0)
    for (x, y) in helpers:
        dx, dy = x - xc, y - yc
        z = "Ayuda de corona" if dy >= abs(dx) * 0.8 else "Ayuda de arrastre" if -dy >= abs(dx) * 0.8 \
            else "Ayuda de hastiales"
        add_face_hole(z, x, y, 0.0, 0.0, Bprod, o.SB_prod * Bprod)
    res.holes = finalize_holes(rows, ZONES)
    res.zone_order = [z for z in ZONES if z in set(res.holes.zona)]
    res.zone_colors = colors_for(ZONES)
    res.simultaneous = set(cut.simultaneous) | ({"Corona"} if controlled else set())
    n_tot = len(res.holes) + len(res.relief)
    if o.narrow or s.area < 6:
        if n_tot > 1.4 * Npred:
            res.warnings.append(f"La malla tiene {n_tot} perforaciones y el predimensionamiento del libro da "
                                f"≈ {Npred:.0f}: revise el tipo de cuele o el burden.")
    fmin, fmax = factor_carga_tabla(s.area, rock.RMR)
    vol = s.area * adv
    fc = res.holes.kg.sum() / vol
    lg.add("7. Verificación", "Factor de carga resultante", "FC", fc, "kg/m³", f"Tabla 08 (Carpio): {fmin}–{fmax}")
    if fc > 1.3 * fmax:
        res.warnings.append(f"Factor de carga {fc:.2f} kg/m³ por encima del rango de la Tabla 08 ({fmin}–{fmax}).")
    res.view2d = dict(kind="face", outline=list(s.polygon.exterior.coords),
                      openings=[(sname, list(affinity.translate(_sec_poly(sec), xc, yc).exterior.coords))
                                for sname, sec in [(sc["name"], sc) for sc in cut.sections]]
                      if cut.sections else [("Cuele", list(opening.exterior.coords))],
                      xlabel="Ancho (m)", ylabel="Altura (m)", dims=(A, s.height))
    res.scene3d = dict(kind="tunnel", outline=list(s.polygon.exterior.coords), advance=adv)
    res.summary = summary_common(res, vol, rock.rho, cost, extra={"Avance por disparo (m)": adv,
                                                                   "Área de la sección (m²)": s.area})
    res.frag = dict(B=Bprod, S=o.SB_prod * Bprod, q=q0, Lc=Lc, H=L_face, d_mm=drill.d1_mm, W=F,
                    rws=exm.rws * 100, modo="optimizar", B_limit=Bp_limit, zona="ayudas (producción)")
    return res


def _sec_poly(sec):
    from shapely.geometry import Polygon
    import math as m
    return Polygon([(sec["dist"] * m.cos(m.radians(sec["rot"] + 90 * k)),
                     sec["dist"] * m.sin(m.radians(sec["rot"] + 90 * k))) for k in range(4)])


def _cart_diam(ex: Explosive):
    from .common import CARTRIDGES
    return CARTRIDGES.get(ex.cart, (0.03175, 0.3))[0]
