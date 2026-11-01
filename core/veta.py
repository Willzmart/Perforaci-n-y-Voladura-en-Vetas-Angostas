"""Geometría de la veta en el frente o en el tajeo, clasificación veta/caja, voladura en dos etapas
(veta y desquinche de caja) y dilución.

Criterios: el área de mineral y de desmonte se obtiene por intersección geométrica de la franja de
veta con la sección; la dilución es la fracción de desmonte en el material enviado como mineral.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon


@dataclass
class Vein:
    width: float = 0.5          # potencia real (m)
    dip_deg: float = 75.0       # buzamiento (°)
    x_center: float = 0.0       # posición del eje de la veta a media altura del frente (m)
    grade: float = 8.0          # ley
    unit: str = "g/t"           # "g/t" o "%"
    rho_ore: float = 2.8        # t/m³
    rho_waste: float = 2.65     # t/m³
    overbreak: float = 0.10     # dilución externa en voladura selectiva (fracción del mineral)

    @property
    def horiz_width(self):
        """Ancho horizontal aparente de la veta en una sección perpendicular al rumbo."""
        return self.width / max(math.sin(math.radians(self.dip_deg)), 0.05)


def band_inclined(v: Vein, x_center, z_mid, zmin, zmax):
    """Franja inclinada (buzamiento) en un plano vertical perpendicular al rumbo (x horizontal, z vertical)."""
    t = 1 / math.tan(math.radians(max(v.dip_deg, 5)))
    h = v.horiz_width / 2
    z0, z1 = zmin - 1, zmax + 1
    xa, xb = x_center + (z0 - z_mid) * t, x_center + (z1 - z_mid) * t
    return Polygon([(xa - h, z0), (xa + h, z0), (xb + h, z1), (xb - h, z1)])


def band_plan(v: Vein, x0, x1, y_center=0.0):
    """Franja de veta en planta (x a lo largo del rumbo, y transversal): ancho horizontal aparente."""
    h = v.horiz_width / 2
    return Polygon([(x0, y_center - h), (x1, y_center - h), (x1, y_center + h), (x0, y_center + h)])


def band_manto(v: Vein, ymin, ymax, z_floor=0.0):
    """Manto horizontal o poco inclinado visto de frente: franja de potencia constante desde el piso."""
    return Polygon([(ymin - 1, z_floor), (ymax + 1, z_floor), (ymax + 1, z_floor + v.width),
                    (ymin - 1, z_floor + v.width)])


def classify(holes, band: Polygon, margin=0.05):
    """Devuelve una serie booleana: True si la boca del taladro está en la veta (con un margen)."""
    b = band.buffer(margin)
    return holes.apply(lambda r: b.contains(Point(r.x, r.y)), axis=1)


def tonnage(area_section, area_ore, advance, v: Vein):
    T_o = max(area_ore, 0) * advance * v.rho_ore
    T_w = max(area_section - area_ore, 0) * advance * v.rho_waste
    return T_o, T_w


def balance(T_o, T_w, v: Vein, selective: bool):
    """Balance de mineral para una voladura completa (mezcla) o en dos etapas (selectiva)."""
    if selective:
        T_ore_pile = T_o * (1 + v.overbreak)
        T_dil = T_o * v.overbreak
        T_waste_pile = max(T_w - T_dil, 0)
    else:
        T_ore_pile = T_o + T_w
        T_dil = T_w
        T_waste_pile = 0.0
    grade_d = v.grade * T_o / T_ore_pile if T_ore_pile > 0 else 0.0
    dil = T_dil / T_ore_pile if T_ore_pile > 0 else 0.0
    if v.unit == "g/t":
        metal = T_o * v.grade
        metal_txt = f"{metal:,.0f} g ({metal / 31.1035:,.1f} oz)"
    else:
        metal = T_o * v.grade / 100
        metal_txt = f"{metal:,.2f} t de metal"
    return dict(T_ore_pile=T_ore_pile, T_waste_pile=T_waste_pile, dilucion=dil, ley_diluida=grade_d,
                metal=metal, metal_txt=metal_txt)


def oversize_pct(X50, n, opening_cm):
    """% retenido sobre la parrilla (Rosin-Rammler): 100·exp(−0.693·(x/X50)^n)."""
    return 100 * math.exp(-0.693 * (opening_cm / X50) ** n)


def axis_points(band: Polygon, section: Polygon, n=7):
    """Puntos sobre el eje de la franja dentro de la sección (cara libre de la segunda etapa)."""
    inter = band.intersection(section)
    if inter.is_empty:
        return []
    c = inter.centroid
    minx, miny, maxx, maxy = inter.bounds
    pts = []
    for z in np.linspace(miny + 0.1, maxy - 0.1, n):
        line = LineString([(minx - 1, z), (maxx + 1, z)]).intersection(inter)
        if not line.is_empty:
            pts.append((line.centroid.x, z))
    return pts or [(c.x, c.y)]
