"""Envoltorio para vetas angostas y labores artesanales: avance en veta y producción en veta."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from shapely.geometry import Point, Polygon

from .avance import AvanceOptions, Drill, run_avance
from .produccion import NOTES, PROD_METHODS, ProdOptions, run_produccion
from .veta import Vein, axis_points, balance, band_inclined, band_manto, band_plan, classify, tonnage

VETA_CUTS = ["Quemado simple", "Quemado Sarrois", "Quemado Sueco", "Fagersta (vacío central 64–76 mm)",
             "Cilíndrico 4 secciones — Holmberg (López Jimeno, con desviación)",
             "Cilíndrico 4 secciones — Holmberg (Carpio)", "En V / cuña"]
VETA_PROD = {
    "Corte y relleno ascendente — realce": PROD_METHODS[6],
    "Almacenamiento provisional (shrinkage) — realce vertical": PROD_METHODS[6],
    "Corte y relleno ascendente — breasting": PROD_METHODS[7],
    "Taladros largos en vetas angostas (subniveles)": PROD_METHODS[1],
    "Subniveles con taladros en abanico (veta angosta)": PROD_METHODS[2],
    "Cámaras y pilares en mantos angostos (perforación horizontal)": PROD_METHODS[5],
}
SHRINK_NOTE = ("Almacenamiento provisional: se vuela el techo con barrenos de realce casi verticales y el mineral "
               "roto queda en el tajeo como piso de trabajo; en cada voladura sólo se extrae el esponjamiento. "
               "Los libros no desarrollan este método: el esquema usa la geometría y las fórmulas de realce "
               "(LJ §6) y la carga máxima por retardo del secuenciamiento.")

# Equipos de perforación típicos (diámetros y barras)
EQUIPOS = {
    "Perforadora manual tipo jackleg (Ø 32–38 mm)": dict(d1=38, bars=[2, 3, 4, 5, 6, 8], bar=5),
    "Stoper para realce (Ø 32–38 mm)": dict(d1=38, bars=[2, 3, 4, 5, 6], bar=4),
    "Martillo manual y empujador (Ø 29–33 mm, LJ §6)": dict(d1=32, bars=[2, 3, 4, 5, 6], bar=4),
    "Equipo de taladros largos (Ø 51–64 mm)": dict(d1=64, bars=[4, 5, 6, 8, 10, 12], bar=5),
}


def run_avance_veta(section, rock, drill: Drill, exm, exc, o: AvanceOptions, vein: Vein, cut_in_vein: bool,
                    two_stage: bool, cost=None):
    H = section.height

    def vein_x(z):
        return vein.x_center + (z - H / 2) / math.tan(math.radians(max(vein.dip_deg, 5)))

    res = run_avance(section, rock, drill, exm, exc, o, cost)
    if cut_in_vein:
        yc = res.p["cut_center"][1]
        res = run_avance(section, rock, drill, exm, exc, replace(o, cut_x=vein_x(yc)), cost)
    band = band_inclined(vein, vein.x_center, H / 2, 0, H)
    band_s = band.intersection(section.polygon)
    A_ore = band_s.area
    adv = res.p["adv"]
    h = res.holes
    inv = classify(h, band_s, margin=0.05)
    h["en_veta"] = inv.values
    cut_in = band_s.buffer(0.05).contains(Point(*res.p["cut_center"]))
    if two_stage:
        h["etapa"] = np.where(h.en_veta | h.cuele, 1, 2)
        if not cut_in:
            res.warnings.append("Voladura en dos etapas con el cuele fuera de la veta: ubique el cuele en la veta para "
                                "que la primera etapa tenga cara libre dentro del mineral.")
    else:
        h["etapa"] = 1
    res.holes = h
    T_o, T_w = tonnage(section.area, A_ore, adv, vein)
    bal = balance(T_o, T_w, vein, two_stage)
    lg = res.log
    F = "8. Veta y dilución"
    lg.add(F, "Ancho horizontal aparente de la veta", "w_h", vein.horiz_width, "m", "potencia / sen(buzamiento)")
    lg.add(F, "Área de mineral en el frente", "A_veta", A_ore, "m²", "intersección franja–sección")
    lg.add(F, "Área de desmonte (caja)", "A_caja", section.area - A_ore, "m²")
    lg.add(F, "Mineral in situ por disparo", "T_min", T_o, "t", "A_veta·avance·ρ_mineral")
    lg.add(F, "Desmonte por disparo", "T_des", T_w, "t", "A_caja·avance·ρ_desmonte")
    lg.add(F, "Modo de voladura", "—", "dos etapas (veta + desquinche)" if two_stage else "frente completo", "", "")
    lg.add(F, "Material enviado como mineral", "T", bal["T_ore_pile"], "t",
           f"mineral × (1 + {vein.overbreak:.0%} de sobrerrotura)" if two_stage else "todo el frente se mezcla")
    lg.add(F, "Dilución", "D", bal["dilucion"] * 100, "%", "desmonte / material enviado como mineral")
    lg.add(F, "Ley diluida", "ley_d", bal["ley_diluida"], vein.unit, "ley·T_min / T_enviado")
    lg.add(F, "Contenido metálico por disparo", "—", bal["metal_txt"], "", "")
    if two_stage:
        for e in (1, 2):
            g = h[h.etapa == e]
            lg.add(F, f"Etapa {e}: taladros y explosivo", "—", f"{len(g)} taladros · {g.kg.sum():.2f} kg", "", "")
    res.p["veta"] = dict(band=band_s, A_ore=A_ore, T_o=T_o, T_w=T_w, cut_in=cut_in, two_stage=two_stage, **bal)
    res.p["stage2_sources"] = axis_points(band, section.polygon)
    coords = list(band_s.exterior.coords) if band_s.geom_type == "Polygon" and not band_s.is_empty else []
    res.view2d["bands"] = [(coords, f"Veta ({vein.width:.2f} m, {vein.dip_deg:.0f}°)", "rgba(230,180,30,0.35)")]
    _summary_vein(res, bal, T_o, T_w, cost)
    A, Hh = section.width, section.height
    if A < 1.5 or Hh < 1.5:
        res.warnings.append(f"Sección de {A:.2f} × {Hh:.2f} m, menor al estándar de 1.5 × 1.5 m: revise el "
                            "sostenimiento, la ventilación y la evacuación de heridos (camilla ≈ 0.55 m de ancho).")
    return res


def run_prod_veta(rock, ex, po: ProdOptions, vein: Vein, label: str, swell=0.35, cost=None):
    po = replace(po, method=VETA_PROD[label])
    res = run_produccion(rock, ex, po, cost)
    m = po.method
    W = po.width
    wh = vein.horiz_width
    if label.startswith("Almacenamiento"):
        res.p["note"] = SHRINK_NOTE
    else:
        res.p["note"] = NOTES.get(m, "")
    v2 = res.view2d
    outline = Polygon(v2["outline"]) if v2.get("outline") else None
    if m in (PROD_METHODS[1], PROD_METHODS[6]):  # vistas en planta
        x1 = max(c[0] for c in v2["outline"])
        band = band_plan(vein, 0, x1).intersection(outline)
        f_ore = min(wh, W) / W
    elif m == PROD_METHODS[5]:  # manto horizontal, vista frontal
        ys = [c[0] for c in v2["outline"]]
        band = band_manto(vein, min(ys), max(ys)).intersection(outline)
        f_ore = min(vein.width, po.height) / po.height
    else:  # breasting o anillo de abanico: franja inclinada en la vista frontal/sección
        zs = [c[1] for c in v2["outline"]]
        band = band_inclined(vein, 0.0, (min(zs) + max(zs)) / 2, min(zs), max(zs)).intersection(outline)
        f_ore = band.area / outline.area if outline.area else 1.0
        if m == PROD_METHODS[2]:
            f_ore = band.area / max(outline.area - po.drift_w * po.drift_h, 1e-6)
    f_ore = float(min(max(f_ore, 0.0), 1.0))
    vol = res.p.get("vol", 0.0)
    T_o = vol * f_ore * vein.rho_ore
    T_w = vol * (1 - f_ore) * vein.rho_waste
    bal = balance(T_o, T_w, vein, selective=False)
    lg = res.log
    F = "4. Veta y dilución"
    lg.add(F, "Ancho horizontal aparente de la veta", "w_h", wh, "m", "potencia / sen(buzamiento)")
    lg.add(F, "Ancho de minado", "W", W if m != PROD_METHODS[5] else po.height, "m")
    lg.add(F, "Fracción de mineral en el volumen volado", "f", f_ore * 100, "%")
    lg.add(F, "Mineral por voladura", "T_min", T_o, "t")
    lg.add(F, "Desmonte por voladura (dilución)", "T_des", T_w, "t")
    lg.add(F, "Dilución", "D", bal["dilucion"] * 100, "%", "desmonte / total extraído")
    lg.add(F, "Ley diluida", "ley_d", bal["ley_diluida"], vein.unit)
    lg.add(F, "Contenido metálico por voladura", "—", bal["metal_txt"], "", "")
    if label.startswith("Almacenamiento"):
        lg.add(F, "Esponjamiento", "e", swell * 100, "%")
        lg.add(F, "Material a extraer por voladura", "T_ext", vol * swell * (f_ore * vein.rho_ore +
                                                                          (1 - f_ore) * vein.rho_waste), "t",
               "e·V_in situ (el resto queda como piso de trabajo)")
    if m != PROD_METHODS[5] and W < wh - 1e-6:
        res.warnings.append(f"El ancho de minado ({W:.2f} m) es menor que el ancho aparente de la veta "
                            f"({wh:.2f} m): queda mineral en las cajas.")
    if m in (PROD_METHODS[6], PROD_METHODS[7]) and W < 0.8:
        res.warnings.append("Ancho de minado menor a 0.8 m: difícil para el trabajo manual y el sostenimiento.")
    coords = list(band.exterior.coords) if band.geom_type == "Polygon" and not band.is_empty else []
    v2["bands"] = [(coords, f"Veta ({vein.width:.2f} m, {vein.dip_deg:.0f}°)", "rgba(230,180,30,0.35)")]
    res.p["veta"] = dict(f_ore=f_ore, T_o=T_o, T_w=T_w, **bal)
    _summary_vein(res, bal, T_o, T_w, cost)
    return res


def _summary_vein(res, bal, T_o, T_w, cost):
    S = res.summary
    S["Mineral in situ (t)"] = T_o
    S["Desmonte (t)"] = T_w
    S["Enviado como mineral (t)"] = bal["T_ore_pile"]
    S["Dilución (%)"] = bal["dilucion"] * 100
    S["Ley diluida"] = bal["ley_diluida"]
    if "Costo total por disparo (US$)" in S and bal["T_ore_pile"] > 0:
        S["Costo por tonelada de mineral (US$/t)"] = S["Costo total por disparo (US$)"] / bal["T_ore_pile"]
