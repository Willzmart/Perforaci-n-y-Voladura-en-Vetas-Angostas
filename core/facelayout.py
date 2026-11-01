"""Distribución geométrica de taladros en un frente: contorno y ayudas (producción)."""
from __future__ import annotations

import math

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient


def sample_chain(line: LineString, n_pts: int, include_ends=True):
    if n_pts <= 0:
        return []
    L = line.length
    if include_ends:
        ts = [0.5 * L] if n_pts == 1 else np.linspace(0, L, n_pts)
    else:
        ts = np.linspace(0, L, n_pts + 2)[1:-1]
    out = []
    for t in ts:
        p = line.interpolate(t)
        p2 = line.interpolate(min(L, t + 0.01))
        p1 = line.interpolate(max(0, t - 0.01))
        tx, ty = p2.x - p1.x, p2.y - p1.y
        nrm = math.hypot(tx, ty) or 1.0
        out.append((p.x, p.y, tx / nrm, ty / nrm))
    return out


def tunnel_chains(poly: Polygon, springline: float, floor_tol=0.03):
    """Divide el contorno (antihorario) en piso, hastial derecho, corona y hastial izquierdo."""
    ring = LineString(poly.exterior.coords)
    L = ring.length
    n = max(400, int(L / 0.01))
    ts = np.linspace(0, L, n, endpoint=False)
    pts = [ring.interpolate(t) for t in ts]
    ymin = min(p.y for p in pts)

    def cls(p):
        if p.y <= ymin + floor_tol:
            return "piso"
        if p.y > springline:
            return "corona"
        return "der" if p.x > 0 else "izq"

    labels = [cls(p) for p in pts]
    k0 = next((i for i in range(n) if labels[i] != labels[i - 1]), 0)
    labels = labels[k0:] + labels[:k0]
    pts = pts[k0:] + pts[:k0]
    groups = []
    cur, cur_pts = labels[0], [pts[0]]
    for lab, p in zip(labels[1:], pts[1:]):
        if lab == cur:
            cur_pts.append(p)
        else:
            groups.append((cur, cur_pts + [p]))
            cur, cur_pts = lab, [p]
    groups.append((cur, cur_pts + [pts[0]]))
    chains = {}
    for lab, gp in groups:
        if len(gp) < 2:
            continue
        ls = LineString([(p.x, p.y) for p in gp])
        if lab not in chains or chains[lab].length < ls.length:
            chains[lab] = ls
    return chains


def uniform_contour(poly: Polygon, offset: float, S: float):
    """Taladros de contorno equiespaciados (labores verticales). Devuelve (x, y, nx, ny)."""
    cp = orient(poly.buffer(-offset, join_style=2), 1.0) if offset > 0 else poly
    ls = LineString(cp.exterior.coords)
    n = max(4, math.ceil(ls.length / S))
    out = []
    for i in range(n):
        t = i * ls.length / n
        p = ls.interpolate(t)
        p2 = ls.interpolate((t + 0.01) % ls.length)
        tx, ty = p2.x - p.x, p2.y - p.y
        nr = math.hypot(tx, ty) or 1
        out.append((p.x, p.y, ty / nr, -tx / nr))
    return out, cp


def fill_helpers(poly: Polygon, cpoly: Polygon, opening: Polygon, existing: list, relief_xy: list,
                 Bprod: float, SB: float, d_ring: float, floor_cut: float | None = None, center=(0.0, 0.0),
                 mirror_axis: float | None = 0.0, max_new=120):
    """Fila de ayudas paralela al contorno + relleno simétrico de máxima distancia.

    Devuelve lista de (x, y). Cada punto de roca debe quedar a ≤ ½·√(B² + S²) de algún taladro.
    """
    new = []
    S_ring = SB * Bprod
    ring1 = cpoly.buffer(-d_ring, join_style=2)
    if floor_cut is not None:
        ring1 = ring1.intersection(box(-1e3, floor_cut, 1e3, 1e3))
    if ring1.geom_type == "MultiPolygon":
        ring1 = max(ring1.geoms, key=lambda g: g.area)
    pts = list(existing)
    if not ring1.is_empty and ring1.area > 0.05:
        excl = opening.buffer(0.6 * Bprod, join_style=2)
        ls = LineString(orient(ring1, 1.0).exterior.coords)
        L = ls.length
        n = max(4, math.ceil(L / S_ring))
        if n % 2:
            n += 1
        t0 = ls.project(Point(center[0], ring1.bounds[1] - 1))
        for i in range(n):
            pt = ls.interpolate((t0 + i * L / n) % L)
            if excl.contains(pt):
                continue
            if pts and min(math.hypot(pt.x - ex, pt.y - ey) for ex, ey in pts) < 0.6 * Bprod:
                continue
            new.append((pt.x, pt.y))
            pts.append((pt.x, pt.y))
    R_cov = 0.5 * math.hypot(Bprod, S_ring)
    minx, miny, maxx, maxy = poly.bounds
    step = max(0.03, min(0.06, Bprod / 15))
    gx, gy = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    G = np.c_[gx.ravel(), gy.ravel()]
    inner = cpoly.buffer(-0.35 * d_ring)
    mask = contains_xy(inner, G[:, 0], G[:, 1]) & ~contains_xy(opening, G[:, 0], G[:, 1])
    G = G[mask]
    if len(G) and (pts or relief_xy):
        H = np.array(pts + list(relief_xy))
        dmin = np.min(np.hypot(G[:, None, 0] - H[None, :, 0], G[:, None, 1] - H[None, :, 1]), axis=1)
        for _ in range(max_new):
            k = int(np.argmax(dmin))
            if dmin[k] <= R_cov:
                break
            x, y = G[k]
            if mirror_axis is not None and abs(x - mirror_axis) >= 0.25 * Bprod:
                cand = [(x, y), (2 * mirror_axis - x, y)]
            elif mirror_axis is not None:
                cand = [(mirror_axis, y)]
            else:
                cand = [(x, y)]
            excl2 = opening.buffer(0.4 * Bprod)
            allp = pts + new + list(relief_xy)
            for j, (px, py) in enumerate(cand):
                if j > 0:  # punto espejo: validar que no caiga en el cuele ni junto a otro taladro
                    if excl2.contains(Point(px, py)) or not inner.contains(Point(px, py)):
                        continue
                    if allp and min(math.hypot(px - ex, py - ey) for ex, ey in allp) < 0.5 * Bprod:
                        continue
                new.append((px, py))
                allp.append((px, py))
                dmin = np.minimum(dmin, np.hypot(G[:, 0] - px, G[:, 1] - py))
    return new
