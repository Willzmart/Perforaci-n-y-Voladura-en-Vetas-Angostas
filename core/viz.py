"""Figuras Plotly genéricas: vista 2D (frente o planta), vista 3D, columnas de carga y secuencia."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STEM = "#9c8f7a"
FUSE = "#f2c200"
PRIMER = "#ff2d2d"
ROCK_FILL = "rgba(160,160,160,0.18)"


def _xy(coords):
    a = np.asarray(coords)
    return list(a[:, 0]), list(a[:, 1])


def view_2d(res, label="id", seq=None, show_vectors=False, show_openings=True, height=700, color_by="zona"):
    v = res.view2d
    h = res.holes.copy()
    if res.p.get("view_filter"):
        h = h[h.zona == res.p["view_filter"]]
    fig = go.Figure()
    if v.get("outline"):
        x, y = _xy(v["outline"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=ROCK_FILL,
                                 line=dict(color="#333", width=2.5), name="Contorno", hoverinfo="skip"))
    for coords, name, color in v.get("bands", []):
        if coords:
            x, y = _xy(coords)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=color, name=name,
                                     line=dict(color="rgba(160,120,0,0.8)", width=1), hoverinfo="skip"))
    for xs, ys in v.get("lines", []):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#888", width=1), showlegend=False,
                                 hoverinfo="skip"))
    if show_openings:
        for name, coords in v.get("openings", []):
            x, y = _xy(coords)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=res.zone_colors.get(name, "#999"),
                                     width=1, dash="dot"), showlegend=False, hoverinfo="skip"))
    rel = res.relief
    if rel is not None and len(rel):
        for _, r in rel.iterrows():
            rr = r.d_mm / 2000
            fig.add_shape(type="circle", x0=r.x - rr, x1=r.x + rr, y0=r.y - rr, y1=r.y + rr,
                          line=dict(color="#222", width=1.5), fillcolor="white")
        fig.add_trace(go.Scatter(x=rel.x, y=rel.y, mode="markers", name="Vacío / alivio",
                                 marker=dict(size=11, color="white", line=dict(color="#222", width=2)),
                                 hovertemplate="Vacío Ø%{customdata:.0f} mm<extra></extra>", customdata=rel.d_mm))
    if seq is not None:
        h = h.merge(seq[["id", "periodo", "t_ms", "predecesor"]], on="id", how="left")
        if show_vectors:
            pos = h.set_index("id")[["x", "y"]]
            for _, r in h.iterrows():
                if r.predecesor and r.predecesor in pos.index:
                    px, py = pos.loc[r.predecesor]
                    fig.add_annotation(x=r.x, y=r.y, ax=px, ay=py, xref="x", yref="y", axref="x", ayref="y",
                                       showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="rgba(0,0,0,0.35)")
    groups = [(z, h[h.zona == z], res.zone_colors.get(z, "#555")) for z in res.zone_order]
    if color_by == "etapa" and "etapa" in h:
        groups = [(f"Etapa {e} — {'veta' if e == 1 else 'caja (desquinche)'}", h[h.etapa == e],
                   "#c8102e" if e == 1 else "#1f4e9c") for e in sorted(h.etapa.unique())]
    for z, g, zc in groups:
        if g.empty:
            continue
        if label == "id":
            txt = g.id.astype(str)
        elif label == "periodo" and "periodo" in g:
            txt = g.periodo.map(lambda v: "" if pd.isna(v) else str(int(v)))
        elif label == "kg":
            txt = g.kg.round(1).astype(str)
        else:
            txt = None
        cd = np.c_[g.id, g.B.fillna(0), g.kg, g.n_cart, g.L]
        hov = "N° %{customdata[0]} — " + z + "<br>B = %{customdata[1]:.2f} m · L = %{customdata[4]:.2f} m" \
              "<br>%{customdata[2]:.2f} kg (%{customdata[3]} cart.)"
        if "periodo" in g:
            cd = np.c_[cd, g.periodo, g.t_ms]
            hov += "<br>Retardo %{customdata[5]} — %{customdata[6]:.0f} ms"
        fig.add_trace(go.Scatter(x=g.x, y=g.y, mode="markers+text" if txt is not None else "markers", text=txt,
                                 textposition="top center", textfont=dict(size=9), name=f"{z} ({len(g)})",
                                 marker=dict(size=10, color=zc, line=dict(color="#111", width=0.7)),
                                 customdata=cd, hovertemplate=hov + "<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title=v.get("ylabel", "y (m)"), zeroline=False)
    fig.update_xaxes(title=v.get("xlabel", "x (m)"), zeroline=False)
    ttl = {"face": "Vista frontal del frente", "plan": "Vista en planta de la malla",
           "front": "Vista frontal / sección de la malla"}.get(v.get("kind"), "Malla")
    if res.p.get("view_filter"):
        ttl += " (primera sección del abanico)"
    fig.update_layout(height=height, plot_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10), title=ttl,
                      legend=dict(x=1.02, y=1))
    return fig


# ----------------------------------------------------------------------------
def _poly_mesh(coords, plane, c, color, opacity, name):
    pts = np.asarray(coords)[:-1]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    a = np.r_[cx, pts[:, 0]]
    b = np.r_[cy, pts[:, 1]]
    k = np.full_like(a, c)
    n = len(pts)
    i, j, kk = np.zeros(n, int), np.arange(1, n + 1), np.r_[np.arange(2, n + 1), 1]
    X, Y, Z = (a, k, b) if plane == "xz" else (a, b, k)
    return go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=kk, color=color, opacity=opacity, name=name, hoverinfo="skip",
                     showlegend=True, flatshading=True)


def _prism(coords, plane, c0, c1, opacity, name):
    pts = np.asarray(coords)
    n = len(pts)
    a = np.r_[pts[:, 0], pts[:, 0]]
    b = np.r_[pts[:, 1], pts[:, 1]]
    k = np.r_[np.full(n, c0), np.full(n, c1)]
    I, J, K = [], [], []
    for t in range(n - 1):
        I += [t, t + 1]
        J += [t + 1, n + t + 1]
        K += [n + t, n + t]
    X, Y, Z = (a, k, b) if plane == "xz" else (a, b, k)
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color="#8a8a8a", opacity=opacity, name=name, hoverinfo="skip",
                     showlegend=True)


def _ring(coords, plane, c, name, dash=None, color="#444"):
    a, b = _xy(coords)
    k = [c] * len(a)
    X, Y, Z = (a, k, b) if plane == "xz" else (a, b, k)
    return go.Scatter3d(x=X, y=Y, z=Z, mode="lines", name=name, line=dict(color=color, width=4, dash=dash),
                        hoverinfo="skip")


def _box(x0, x1, y0, y1, z0, z1, name, color="#555", shear=0.0):
    """Caja (o paralelepípedo inclinado según el buzamiento: y se desplaza (z1 − z)·cot(β))."""
    P = [(x0, y0 - (z1 - z0) * shear, z0), (x1, y0 - (z1 - z0) * shear, z0), (x1, y1 - (z1 - z0) * shear, z0),
         (x0, y1 - (z1 - z0) * shear, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in E:
        xs += [P[a][0], P[b][0], None]
        ys += [P[a][1], P[b][1], None]
        zs += [P[a][2], P[b][2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=3), name=name, hoverinfo="skip")


def view_3d(res, system, primer="Fondo", show_shell=True, show_init=True, show_stem=True, zones=None, seq=None,
            height=760, max_holes=900):
    h = res.holes if not zones else res.holes[res.holes.zona.isin(zones)]
    if len(h) > max_holes:
        h = h.iloc[:max_holes]
    sc = res.scene3d
    fig = go.Figure()
    kind = sc.get("kind")
    if kind == "tunnel":
        fig.add_trace(_poly_mesh(sc["outline"], "xz", 0.0, "#c9b79c", 0.35, "Frente"))
        fig.add_trace(_ring(sc["outline"], "xz", 0.0, "Contorno del frente"))
        fig.add_trace(_ring(sc["outline"], "xz", sc["advance"], f"Avance esperado ({sc['advance']:.2f} m)", "dash",
                            "#1f77b4"))
        if show_shell:
            fig.add_trace(_prism(sc["outline"], "xz", -2.0, 0.0, 0.07, "Labor excavada"))
        eye = dict(x=-1.25, y=-2.1, z=0.75)
    elif kind == "vertical":
        up = sc["up"]
        if sc.get("length"):
            Lr = sc["length"]
            fig.add_trace(_ring(sc["outline"], "xy", 0.0, "Nivel inferior (cara libre)"))
            fig.add_trace(_ring(sc["outline"], "xy", Lr, "Nivel superior (perforación)"))
            if show_shell:
                fig.add_trace(_prism(sc["outline"], "xy", 0.0, Lr, 0.06, "Contorno de la chimenea"))
        else:
            fig.add_trace(_poly_mesh(sc["outline"], "xy", 0.0, "#c9b79c", 0.35, "Frente"))
            fig.add_trace(_ring(sc["outline"], "xy", 0.0, "Contorno del frente"))
            adv = sc["advance"] if up else -sc["advance"]
            fig.add_trace(_ring(sc["outline"], "xy", adv, f"Avance esperado ({abs(adv):.2f} m)", "dash", "#1f77b4"))
            if show_shell:
                fig.add_trace(_prism(sc["outline"], "xy", -2.0 if up else 0.0, 0.0 if up else 2.0, 0.07,
                                     "Labor excavada"))
        eye = dict(x=1.6, y=-1.6, z=0.9 if up else 1.4)
    else:
        if "box" in sc:
            shr = sc.get("shear", 0.0)
            fig.add_trace(_box(*sc["box"], "Bloque / tajeo", shear=shr))
            if sc.get("slot"):
                x0, x1, y0, y1, z0, z1 = sc["box"]
                fig.add_trace(_box(x0 - 2.0, x0, y0, y1, z0, z1, "Corte lateral (cara libre)", "#d62728", shear=shr))
            if sc.get("drift"):
                w, hh = sc["drift"]
                x0, x1 = sc["box"][0], sc["box"][1]
                fig.add_trace(_box(x0, x1, -w / 2, w / 2, 0, hh, "Galería de perforación", "#1f77b4"))
        eye = dict(x=1.7, y=-2.0, z=1.1)

    def pt(r, s):
        return np.array([r.X0 + r.ux * s, r.Y0 + r.uy * s, r.Z0 + r.uz * s])

    stem, fuse, prim, coll = [], [], [], []
    charge = {z: [] for z in res.zone_order}
    for _, r in h.iterrows():
        for s0, s1 in (r.tacos or []):
            stem += [pt(r, s0), pt(r, s1), [None] * 3]
        for s0, s1, qq in r.cargas:
            charge.setdefault(r.zona, [])
            charge[r.zona] += [pt(r, s0), pt(r, s1), [None] * 3]
        if r.cargas:
            ends = [s1 for _, s1, _ in r.cargas]
            starts = [s0 for s0, _, _ in r.cargas]
            sp = max(ends) - 0.12 if primer == "Fondo" else min(starts) + 0.12
            p = pt(r, sp)
            prim.append(p)
            off = np.array([0.03, 0.0, 0.03]) if abs(r.uy) > 0.5 else np.array([0.03, 0.03, 0.0])
            c0 = pt(r, 0)
            tail = 0.35
            if "mecha_m" in h.columns and not pd.isna(r.get("mecha_m", np.nan)):
                tail = max(0.2, float(r.mecha_m) - sp)
            fuse += [p + off, c0 + off, c0 + off - np.array([r.ux, r.uy, r.uz]) * tail, [None] * 3]
        coll.append(pt(r, 0))
    for z, seg in charge.items():
        if seg:
            a = np.array(seg, dtype=object)
            fig.add_trace(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="lines",
                                       line=dict(color=res.zone_colors.get(z, "#555"), width=8), name=f"Carga — {z}",
                                       hoverinfo="skip"))
    if show_stem and stem:
        a = np.array(stem, dtype=object)
        fig.add_trace(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="lines", line=dict(color=STEM, width=8),
                                   name="Taco / retacado", hoverinfo="skip"))
    if show_init and fuse:
        a = np.array(fuse, dtype=object)
        lbl = "Carmex (mecha de seguridad)" if system.startswith("Carmex") else (
            "Cable del detonador electrónico" if system.startswith("Elec") else "Tubo de choque (no eléctrico)")
        fig.add_trace(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="lines", line=dict(color=FUSE, width=3),
                                   name=lbl, hoverinfo="skip"))
        pr = np.array(prim)
        fig.add_trace(go.Scatter3d(x=pr[:, 0], y=pr[:, 1], z=pr[:, 2], mode="markers",
                                   marker=dict(size=4, color=PRIMER, symbol="diamond"),
                                   name=f"Cebo + detonador ({primer.lower()})", hoverinfo="skip"))
    if coll:
        co = np.array(coll)
        hs = h.merge(seq[["id", "periodo", "t_ms"]], on="id", how="left") if seq is not None else h
        txt = [f"N° {r.id} — {r.zona}<br>L = {r.L:.2f} m · {r.kg:.2f} kg"
               + (f"<br>Retardo {int(r.periodo)} — {r.t_ms:.0f} ms" if "periodo" in hs and r.periodo == r.periodo else "")
               for _, r in hs.iterrows()]
        fig.add_trace(go.Scatter3d(x=co[:, 0], y=co[:, 1], z=co[:, 2], mode="markers", marker=dict(size=2.5, color="#111"),
                                   name="Boca de taladro", text=txt, hovertemplate="%{text}<extra></extra>"))
    rel = res.relief
    if rel is not None and len(rel):
        xs, ys, zs = [], [], []
        for _, r in rel.iterrows():
            xs += [r.X0, r.X0 + r.ux * r.L, None]
            ys += [r.Y0, r.Y0 + r.uy * r.L, None]
            zs += [r.Z0, r.Z0 + r.uz * r.L, None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#555", width=12),
                                   name="Vacío / alivio", hoverinfo="skip"))
    # proporciones: escenas muy alargadas (chimeneas, tajeos) se comprimen para que se vean completas
    pts = []
    for tr in fig.data:
        if hasattr(tr, "x") and tr.x is not None and len(tr.x):
            pts.append(np.c_[[v if v is not None else np.nan for v in tr.x],
                             [v if v is not None else np.nan for v in tr.y],
                             [v if v is not None else np.nan for v in tr.z]].astype(float))
    scene = dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data",
                 camera=dict(eye=eye, up=dict(x=0, y=0, z=1)))
    if pts:
        A = np.vstack(pts)
        ext = np.nanmax(A, axis=0) - np.nanmin(A, axis=0)
        ext = np.where(ext > 1e-6, ext, 1e-6)
        if ext.max() / max(min(ext), 1e-6) > 4:
            r = np.clip(ext / ext.max(), 0.3, 1.0) * 1.4
            scene.update(aspectmode="manual", aspectratio=dict(x=r[0], y=r[1], z=r[2]))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0),
                      title="Vista 3D: taladros, retacado, carga e iniciación", scene=scene,
                      legend=dict(orientation="h", x=0, y=-0.02, yanchor="top", font=dict(size=11)))
    if len(res.holes) > len(h) and not zones:
        fig.add_annotation(text=f"Se muestran {len(h)} de {len(res.holes)} taladros", x=0.5, y=1.0, xref="paper",
                           yref="paper", showarrow=False)
    return fig


def hole_profiles(res, primer="Fondo"):
    h = res.holes
    zones = [z for z in res.zone_order if z in set(h.zona)]
    fig = go.Figure()
    Lmax = float(h.L.max())
    for k, z in enumerate(zones):
        r = h[h.zona == z].iloc[0]
        y = -k
        fig.add_shape(type="rect", x0=0, x1=r.L, y0=y - 0.18, y1=y + 0.18, line=dict(color="#333"),
                      fillcolor="#f4f4f4", layer="below")
        for s0, s1 in (r.tacos or []):
            fig.add_shape(type="rect", x0=s0, x1=s1, y0=y - 0.14, y1=y + 0.14, line=dict(width=0), fillcolor=STEM,
                          layer="below")
        qmax = max(qq for _, _, qq in r.cargas) if r.cargas else 1
        for s0, s1, qq in r.cargas:
            fig.add_shape(type="rect", x0=s0, x1=s1, y0=y - 0.12, y1=y + 0.12, line=dict(width=0.5, color="#fff"),
                          fillcolor=res.zone_colors.get(z, "#555"), opacity=0.45 + 0.55 * qq / qmax, layer="below")
        if r.cargas:
            ends = [s1 for _, s1, _ in r.cargas]
            starts = [s0 for s0, _, _ in r.cargas]
            px = max(ends) - 0.15 if primer == "Fondo" else min(starts) + 0.15
            fig.add_trace(go.Scatter(x=[px], y=[y], mode="markers", showlegend=(k == 0), name="Cebo + detonador",
                                     marker=dict(symbol="diamond", size=10, color=PRIMER, line=dict(color="#000", width=1))))
            fig.add_trace(go.Scatter(x=[px, -0.03 * Lmax - 0.2], y=[y + 0.05, y + 0.05], mode="lines",
                                     line=dict(color=FUSE, width=3), showlegend=(k == 0), name="Línea de iniciación"))
        qtxt = " / ".join(f"{qq:.2f}" for _, _, qq in r.cargas)
        fig.add_annotation(x=-0.05 * Lmax - 0.3, y=y, text=f"<b>{z}</b>", xanchor="right", showarrow=False,
                           font=dict(size=11))
        fig.add_annotation(x=r.L + 0.02 * Lmax + 0.05, y=y, xanchor="left", showarrow=False, font=dict(size=10),
                           text=f"{r.kg:.2f} kg · q = {qtxt} kg/m")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=12, color=STEM),
                             name="Taco / retacado"))
    fig.update_xaxes(range=[-0.45 * Lmax - 1.5, Lmax * 1.45 + 1.0], title="Distancia desde la boca (m)",
                     zeroline=False)
    fig.update_yaxes(visible=False, range=[-len(zones) + 0.4, 0.6])
    fig.update_layout(height=70 * len(zones) + 170, margin=dict(l=10, r=10, t=40, b=60), plot_bgcolor="white",
                      title="Columna de carga por zona (boca → fondo; el color más intenso es la carga de fondo)",
                      legend=dict(orientation="h", y=-0.15, x=0.25))
    return fig


def sequence_animation(res, seq, height=680):
    per = sorted(seq.periodo.unique())
    v = res.view2d
    base = []
    if v.get("outline"):
        x, y = _xy(v["outline"])
        base.append(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=ROCK_FILL,
                               line=dict(color="#333", width=2.5), showlegend=False, hoverinfo="skip"))
    rel = res.relief
    if rel is not None and len(rel):
        base.append(go.Scatter(x=rel.x, y=rel.y, mode="markers", showlegend=False, hoverinfo="skip",
                               marker=dict(size=11, color="white", line=dict(color="#222", width=2))))
    s = seq if not res.p.get("view_filter") else seq[seq.zona == res.p["view_filter"]]
    if s.empty:
        s = seq

    def state(p):
        f, nw, w = s.periodo < p, s.periodo == p, s.periodo > p
        return [go.Scatter(x=s.x[w], y=s.y[w], mode="markers+text", text=s.periodo[w].astype(str),
                           textposition="top center", marker=dict(size=10, color="#d9d9d9",
                                                                  line=dict(color="#777", width=1)),
                           name="Por salir", hoverinfo="skip"),
                go.Scatter(x=s.x[f], y=s.y[f], mode="markers", marker=dict(size=12, color="#444", symbol="x"),
                           name="Detonado", hoverinfo="skip"),
                go.Scatter(x=s.x[nw], y=s.y[nw], mode="markers+text", text=s.periodo[nw].astype(str),
                           textposition="top center", marker=dict(size=17, color="#ff4b00",
                                                                  line=dict(color="#000", width=1)),
                           name="Detona ahora", hoverinfo="skip")]

    tms = seq.groupby("periodo").t_ms.min()
    fig = go.Figure(data=base + state(per[0]),
                    frames=[go.Frame(data=base + state(p), name=str(p),
                                     layout=go.Layout(title=f"Retardo {p} — t = {tms[p]:.0f} ms")) for p in per])
    fig.update_layout(height=height, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10),
                      title=f"Retardo {per[0]}",
                      updatemenus=[dict(type="buttons", x=0.0, y=-0.07, direction="left", buttons=[
                          dict(label="▶ Reproducir", method="animate",
                               args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                          dict(label="⏸ Pausa", method="animate",
                               args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
                      sliders=[dict(active=0, y=-0.02, currentvalue=dict(prefix="Retardo "),
                                    steps=[dict(method="animate", label=str(p),
                                                args=[[str(p)], dict(mode="immediate",
                                                                     frame=dict(duration=0, redraw=True))])
                                           for p in per])])
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def charge_per_delay(seq, qmax):
    g = seq.groupby("periodo").agg(kg=("kg", "sum"), n=("id", "count")).reset_index()
    fig = go.Figure(go.Bar(x=g.periodo, y=g.kg, text=g.n.astype(str) + " tal.", textposition="outside",
                           marker_color="#1f77b4", hovertemplate="Retardo %{x}<br>%{y:.1f} kg<extra></extra>"))
    fig.add_hline(y=qmax, line=dict(color="#d62728", dash="dash"), annotation_text=f"Q máx = {qmax:.1f} kg")
    fig.update_layout(height=340, xaxis_title="N° de retardo", yaxis_title="kg por retardo", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig
