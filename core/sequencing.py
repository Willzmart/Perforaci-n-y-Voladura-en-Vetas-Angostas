"""Secuenciamiento de salida para cualquier tipo de labor.

- Árbol de alivio (Dijkstra con peso d²) desde las caras libres iniciales (vacíos, chimenea, corte).
- Asignación de retardos con precedencia, orden por grupos, un taladro por retardo en el cuele,
  carga máxima por retardo y zonas simultáneas (corona en voladura controlada, cuele instantáneo).
  Métodos: heurística, MILP (HiGHS) o por filas (Du Pont, producción).
- Conversión a tiempos con las series Exsanel MS/LP (Carpio, Tabla N°05), respetando el tiempo mínimo
  entre taladros del arranque (Carpio, Tabla N°04: ms/m × longitud del taladro).
- Daño al shotcrete (Carpio cap. V).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack
from scipy.sparse.csgraph import dijkstra

from .common import EXSANEL_LP, EXSANEL_MS

SYSTEMS = ["Carmex (mecha de seguridad + fulminante)", "No eléctrico — Exsanel periodo corto (MS)",
           "No eléctrico — Exsanel periodo largo (LP)", "No eléctrico — tabla propia", "Electrónico"]


def relief_graph(h: pd.DataFrame, sources, link=1.6):
    n = len(h)
    src = np.asarray(sources, dtype=float).reshape(-1, 2)
    m = len(src)
    XY = np.vstack([src, h[["x", "y"]].values])
    Bv = np.concatenate([np.full(m, np.nanmin(h.B) if h.B.notna().any() else 1.0),
                         h.B.fillna(h.B.median() if h.B.notna().any() else 1.0).values])
    D = np.hypot(XY[:, None, 0] - XY[None, :, 0], XY[:, None, 1] - XY[None, :, 1])
    for _ in range(14):
        R = link * Bv[None, :]
        W = np.where((D <= R) & (D > 1e-9), D ** 2, 0.0)
        W[:, :m] = 0.0
        dist, pred, _ = dijkstra(W, directed=True, indices=list(range(m)), min_only=True, return_predecessors=True)
        if np.all(np.isfinite(dist[m:])):
            break
        link += 0.3
    dist = np.where(np.isfinite(dist), dist, np.nanmax(dist[np.isfinite(dist)]) if np.isfinite(dist).any() else 0)
    parent = np.array([int(pred[j] - m) if pred[j] >= m else -1 for j in range(m, m + n)])
    return dist[m:], parent, link


def schedule_heuristic(h, parent, dist, opt):
    n = len(h)
    period = np.zeros(n, dtype=int)
    load, count, cut_used = {}, {}, set()
    grp = h.grupo.values if opt["classic"] else np.where(h.grupo == h.grupo.max(), 1, 0)
    order = np.lexsort((h.x.values, dist, grp))
    group_end = {}
    sim = opt.get("simultaneous", set())
    blocks = {z: [i for i in range(n) if h.zona.iat[i] == z] for z in sim}
    done = set()

    def fits(p, idxs):
        if load.get(p, 0) + sum(h.kg.iat[i] for i in idxs) > opt["qmax"] + 1e-9:
            return False
        block = len(idxs) > 1
        if not block and count.get(p, 0) + 1 > opt["nmax"]:
            return False
        if block and count.get(p, 0) > 0:
            return False
        if opt["cut_single"] and not block and h.cuele.iat[idxs[0]] and p in cut_used:
            return False
        return True

    def assign(p, idxs):
        for i in idxs:
            period[i] = p
            load[p] = load.get(p, 0) + h.kg.iat[i]
            count[p] = count.get(p, 0) + 1
            if h.cuele.iat[i]:
                cut_used.add(p)
            group_end[grp[i]] = max(group_end.get(grp[i], 0), p)

    for i in order:
        if period[i]:
            continue
        g = grp[i]
        earliest = max([group_end[k] for k in group_end if k < g], default=0) + 1
        if parent[i] >= 0 and period[parent[i]]:
            earliest = max(earliest, period[parent[i]] + 1)
        z = h.zona.iat[i]
        if z in blocks and z not in done:
            done.add(z)
            idxs = blocks[z]
            e = max([earliest] + [period[parent[j]] + 1 for j in idxs if parent[j] >= 0 and period[parent[j]]])
            p = e
            while p < e + 300 and not fits(p, idxs):
                p += 1
            if fits(p, idxs):
                assign(p, idxs)
                continue
        p = earliest
        while not fits(p, [i]):
            p += 1
            if p > 3000:
                raise RuntimeError("No es posible asignar retardos: aumente la carga máxima por retardo.")
        assign(p, [i])
    return period


def schedule_milp(h, parent, opt, P, time_limit=20):
    n = len(h)
    grp = h.grupo.values if opt["classic"] else np.where(h.grupo == h.grupo.max(), 1, 0)
    groups = sorted(set(grp))
    G = len(groups)
    gidx = {g: k for k, g in enumerate(groups)}
    nx = n * P
    nv = nx + G + 1
    X = lambda i, p: i * P + p
    rows, lb, ub = [], [], []
    sim = opt.get("simultaneous", set())

    def add_row(coefs, lo, hi):
        r = lil_matrix((1, nv))
        for j, v in coefs:
            r[0, j] = v
        rows.append(r)
        lb.append(lo)
        ub.append(hi)

    pr = np.arange(1, P + 1)
    T = lambda i: [(X(i, p), pr[p]) for p in range(P)]
    for i in range(n):
        add_row([(X(i, p), 1) for p in range(P)], 1, 1)
    for i in range(n):
        j = parent[i]
        same_block = j >= 0 and h.zona.iat[i] in sim and h.zona.iat[j] == h.zona.iat[i]
        if j >= 0 and grp[j] <= grp[i] and not same_block:
            add_row(T(i) + [(c, -v) for c, v in T(j)], 1, np.inf)
    for i in range(n):
        k = gidx[grp[i]]
        add_row(T(i) + [(nx + k, -1)], -np.inf, 0)
        add_row(T(i) + [(nv - 1, -1)], -np.inf, 0)
        if k > 0:
            add_row(T(i) + [(nx + k - 1, -1)], 1, np.inf)
    cut = [i for i in range(n) if h.cuele.iat[i] and h.zona.iat[i] not in sim]
    cnt = [i for i in range(n) if h.zona.iat[i] not in sim]
    for p in range(P):
        add_row([(X(i, p), h.kg.iat[i]) for i in range(n)], -np.inf, opt["qmax"])
        add_row([(X(i, p), 1) for i in cnt], -np.inf, opt["nmax"])
        if opt["cut_single"] and cut:
            add_row([(X(i, p), 1) for i in cut], -np.inf, 1)
    for z in sim:
        idx = [i for i in range(n) if h.zona.iat[i] == z]
        for i in idx[1:]:
            add_row(T(i) + [(c, -v) for c, v in T(idx[0])], 0, 0)
    A = vstack(rows).tocsr()
    c = np.zeros(nv)
    c[nv - 1] = 1000.0
    for i in range(n):
        for p in range(P):
            c[X(i, p)] = pr[p]
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.r_[np.ones(nx), np.zeros(G + 1)],
               bounds=Bounds(np.zeros(nv), np.r_[np.ones(nx), np.full(G + 1, P)]),
               options=dict(time_limit=time_limit, disp=False))
    if res.x is None:
        return None, res
    return res.x[:nx].reshape(n, P).argmax(axis=1) + 1, res


def schedule_rows(h):
    """Producción por filas: un número por fila; dentro de la fila se escalona 10 ms (Du Pont)."""
    period = h.grupo.values.astype(int) + 1
    return period


def detonator_times(period, system, cfg, cut_mask=None, min_gap_ms=0.0):
    """Devuelve (tiempo ms, número de serie) por período, respetando el tiempo mínimo entre los
    períodos del cuele (Carpio, Tabla 04)."""
    p = np.asarray(period)
    per = sorted(set(p))
    if system.startswith("Carmex"):
        tmap = {k: (k - 1) * cfg["dt_s"] * 1000.0 for k in per}
        return np.array([tmap[k] for k in p]), np.array(p)
    if system == "Electrónico":
        tmap = {k: cfg["t0"] + (k - 1) * cfg["dt_ms"] for k in per}
        return np.array([tmap[k] for k in p]), np.array(p)
    tab = EXSANEL_MS if "MS" in system else EXSANEL_LP if "LP" in system else list(cfg["table"])
    cut_per = set(p[cut_mask]) if cut_mask is not None else set()
    tmap, nmap = {}, {}
    idx, t_prev = -1, -1e9
    for k in per:
        gap = min_gap_ms if (k in cut_per and (k - 1) in cut_per) else 1e-6
        j = idx + 1
        while j < len(tab) and tab[j] - t_prev < gap:
            j += 1
        if j >= len(tab):  # se agota la serie: se extrapola
            step = tab[-1] - tab[-2]
            t = max(t_prev + gap, (tab[-1] + step * (j - len(tab) + 1)))
        else:
            t = tab[j]
        tmap[k], nmap[k] = t, j + 1
        idx, t_prev = j, t
    return np.array([tmap[k] for k in p]), np.array([nmap[k] for k in p])


def fuse_lengths(period, cfg, L):
    base = cfg["tail"] + (np.asarray(L) if cfg["primer"] == "Fondo" else cfg["Lt"] + 0.2)
    return base + (np.asarray(period) - 1) * cfg["dt_s"] / cfg["burn"]


def qmax_from_ppv(ppv_lim, D, K, beta):
    return D ** 2 * (ppv_lim / K) ** (2 / beta)


def shotcrete(fc_mpa, E_gpa=None, nu=0.2, rho=2400.0, K=267.57, alpha=1.62, Q=5.0):
    """Carpio cap. V: E = 15000·√f'c (kg/cm²); Vp = √(E(1−ν)/(ρ(1+ν)(1−2ν))); Vcrit = 0.1·f'c·Vp/E;
    d = (K·Q^α / Vcrit)^(1/α)."""
    fc_kg = fc_mpa * 10.197
    if not E_gpa:
        E_gpa = 15000 * math.sqrt(fc_kg) / 10197.0
    Vp = math.sqrt(E_gpa * 1e9 * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))
    Vc = 0.1 * fc_mpa * Vp / E_gpa
    d = (K * Q ** alpha / Vc) ** (1 / alpha)
    return dict(E=E_gpa, Vp=Vp, Vcrit=Vc, d=d)


def deck_max(cargas):
    """Carga máxima de un tramo continuo de explosivo (las cargas seccionadas salen por separado)."""
    if not cargas:
        return 0.0
    segs = sorted(cargas)
    best, cur, end = 0.0, 0.0, None
    for s0, s1, q in segs:
        if end is not None and s0 - end > 1e-6:
            best, cur = max(best, cur), 0.0
        cur += (s1 - s0) * q
        end = s1
    return max(best, cur)


def run_sequencing(holes, sources, opt, system, cfg, RMR=50):
    h = holes.reset_index(drop=True)
    dist, parent, link = relief_graph(h, sources, opt.get("link", 1.6))
    info = {}
    if opt.get("method") == "Por filas":
        period = schedule_rows(h)
        method = "Por filas (Du Pont: 75 ms entre filas, 10 ms dentro de la fila)"
        period_h = period
    else:
        period_h = schedule_heuristic(h, parent, dist, opt)
        period = period_h
        method = "Heurística (listas con restricciones)"
        if opt.get("method") == "MILP" and len(h) <= 140:
            per_m, res = schedule_milp(h, parent, opt, int(period_h.max()), opt.get("time_limit", 20))
            info["milp_status"] = res.message
            if per_m is not None:
                used = sorted(set(per_m))
                remap = {p: k + 1 for k, p in enumerate(used)}
                period = np.array([remap[p] for p in per_m])
                method = "MILP (HiGHS)" + (" — óptimo" if res.status == 0 else " — mejor solución en el tiempo límite")
            else:
                method += " (MILP sin solución en el tiempo límite)"
        elif opt.get("method") == "MILP":
            method += " (MILP omitido: más de 140 taladros)"
    out = h.copy()
    out["predecesor"] = [int(h.id.iat[j]) if j >= 0 else 0 for j in parent]
    out["dist_alivio"] = dist
    out["periodo"] = period
    gap = cfg.get("ms_m", 25) * float(h.L.median())
    t, num = detonator_times(period, system, cfg, cut_mask=out.cuele.values, min_gap_ms=gap if opt.get("carpio_gap") else 0)
    if opt.get("method") == "Por filas":
        rank = out.groupby("periodo").x.rank(method="first").values - 1
        t = (period - 1) * cfg.get("row_ms", 75) + rank * cfg.get("hole_ms", 10)
        num = period
    out["t_ms"] = t
    out["n_detonador"] = num
    out["kg_carga"] = [deck_max(c) for c in out.cargas]
    if system.startswith("Carmex"):
        out["mecha_m"] = fuse_lengths(period, cfg, h.L.values)
    out = out.sort_values(["periodo", "dist_alivio"]).reset_index(drop=True)
    out["orden"] = np.arange(1, len(out) + 1)
    info.update(method=method, link=link, n_periods=int(out.periodo.max()), heur_periods=int(period_h.max()),
                gap_ms=gap)
    return out, info
