"""Voladura en vetas angostas y labores artesanales — avance horizontal y producción.

Fuentes: Carpio Chávez (2025) y López Jimeno et al. (IGME, 1987).
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import fragmentation as fr
from core import sequencing as sq
from core import viz
from core.avance import CHARGE_METHODS, AvanceOptions, Drill
from core.common import CARTRIDGES, EXPLOSIVES, EXSANEL_LP, EXSANEL_MS, P80_REFERENCIAS, Rock, make_explosive, ms_por_metro
from core.cuts import CUT_NOTES
from core.produccion import CE0_TABLE, CRITERIA, ProdOptions
from core.sections import GALLERY_TYPES, build_section
from core.veta import Vein, oversize_pct
from core.vetas import (ACARREO, DESMONTE_TIPOS, EQUIPOS, VETA_CUTS, VETA_PROD, run_avance_desmonte,
                        run_avance_veta, run_prod_veta)

st.set_page_config(page_title="Voladura en vetas angostas", layout="wide")
ASSETS = Path(__file__).parent / "assets"
ss = st.session_state
ss.setdefault("overrides", {})
ss.setdefault("seq", None)
ss.setdefault("seq_sig", None)

SECCIONES = {"1.00 × 1.50 (artesanal)": (1.0, 1.5), "1.20 × 1.80": (1.2, 1.8), "1.50 × 1.50 (estándar mínimo)": (1.5, 1.5),
             "1.50 × 1.80": (1.5, 1.8), "1.80 × 2.10": (1.8, 2.1), "2.10 × 2.40": (2.1, 2.4), "2.40 × 2.70": (2.4, 2.7),
             "Personalizada": None}

# =============================================================================
st.sidebar.title("Datos de entrada")
labor = st.sidebar.radio("Labor", ["Avance en veta", "Avance en desmonte", "Producción en veta"])
DESM = labor == "Avance en desmonte"


def explosive_picker(label, default, cart_default, key):
    names = list(EXPLOSIVES) + ["Personalizado"]
    name = st.selectbox(label, names, index=names.index(default), key=key + "n")
    if name == "Personalizado":
        c1, c2 = st.columns(2)
        custom = dict(name="Personalizado", vod=c1.number_input("VOD (m/s)", 1500, 8000, 4500, key=key + "v"),
                      rho=c2.number_input("Densidad (g/cm³)", 0.6, 1.6, 1.10, key=key + "r"),
                      rws=c1.number_input("RWS (% ANFO)", 40, 180, 90, key=key + "w") / 100,
                      cart="—", w_cart=c2.number_input("Peso cartucho (kg)", 0.02, 1.0, 0.08, key=key + "c"),
                      L_cart=c1.number_input("Long. cartucho (m)", 0.1, 0.8, 0.178, key=key + "l"), granel=False)
        return make_explosive("Personalizado", custom=custom)
    if EXPLOSIVES[name]["granel"]:
        return make_explosive(name)
    cart = st.selectbox("Cartucho", list(CARTRIDGES), index=list(CARTRIDGES).index(cart_default), key=key + "k")
    return make_explosive(name, cart)


if DESM:
    w_v, dip, unit, grade, rho_o, rho_w, xv, ob = 0.4, 70.0, "g/t", 0.0, 2.8, 2.65, 0.0, 0.1
else:
    with st.sidebar.expander("Veta", expanded=True):
        c1, c2 = st.columns(2)
        w_v = c1.number_input("Potencia real (m)", 0.05, 5.0, 0.40, 0.05)
        dip = c2.number_input("Buzamiento (°)", 10.0, 90.0, 70.0, 1.0)
        unit = c1.selectbox("Unidad de ley", ["g/t", "%"])
        grade = c2.number_input("Ley de la veta", 0.0, 1000.0, 10.0 if unit == "g/t" else 2.0, 0.1)
        rho_o = c1.number_input("Densidad mineral (t/m³)", 2.0, 5.0, 2.80, 0.05)
        rho_w = c2.number_input("Densidad caja (t/m³)", 2.0, 4.0, 2.65, 0.05)
        xv = st.number_input("Posición del eje de la veta en el frente (m)", -2.0, 2.0, 0.0, 0.05,
                             help="Distancia horizontal desde el eje de la labor, medida a media altura.") \
            if labor.startswith("Avance") else 0.0
        ob = st.slider("Sobrerrotura de la voladura selectiva (%)", 0, 40, 10) / 100
vein = Vein(width=w_v, dip_deg=dip, x_center=xv, grade=grade, unit=unit, rho_ore=rho_o, rho_waste=rho_w, overbreak=ob)

with st.sidebar.expander("Macizo rocoso", expanded=False):
    c1, c2 = st.columns(2)
    RMR = c1.number_input("RMR", 5, 100, 50)
    GSI = c2.number_input("GSI", 5, 100, 45)
    RQD = c1.number_input("RQD (%)", 1, 100, 55)
    UCS = c2.number_input("σci (MPa)", 5.0, 350.0, 75.0, 5.0)
    rho_r = c1.number_input("Densidad (t/m³)", 1.8, 4.5, 2.70, 0.01)
    Qb = c2.number_input("Q de Barton", 0.001, 1000.0, 1.95)
    Emod = c1.number_input("E (GPa)", 1.0, 120.0, 35.0)
    JSF = c2.number_input("JSF", 0.5, 1.0, 1.0 if RMR > 60 else 0.9 if RMR > 40 else 0.8 if RMR > 20 else 0.7, 0.05)
rock = Rock(RMR=RMR, GSI=GSI, RQD=RQD, UCS=UCS, rho=rho_r, Q=Qb, JSF=JSF, E=Emod)

with st.sidebar.expander("Iniciación y costos", expanded=False):
    system = st.selectbox("Sistema de iniciación", sq.SYSTEMS, index=0)
    primer = st.radio("Posición del cebo", ["Fondo", "Boca"], horizontal=True)
    cost = dict(drill=st.number_input("Perforación (US$/m)", 0.0, 50.0, 1.5, 0.1),
                exp=st.number_input("Explosivo (US$/kg)", 0.0, 50.0, 4.5, 0.1),
                acc=st.number_input("Accesorios por taladro (US$)", 0.0, 20.0, 1.2, 0.1))

res, err = None, None
try:
    if labor.startswith("Avance"):
        if DESM:
            with st.sidebar.expander("Labor en desmonte", expanded=True):
                tipo_d = st.selectbox("Tipo de labor", list(DESMONTE_TIPOS))
                perm = DESMONTE_TIPOS[tipo_d]
        with st.sidebar.expander("Sección y cuele", expanded=True):
            kind = st.selectbox("Tipo de labor", GALLERY_TYPES, index=3)
            pres = st.selectbox("Sección", list(SECCIONES), index=4, key=f"sec{DESM}")
            dd = SECCIONES[pres] or (1.8, 2.1)
            c1, c2 = st.columns(2)
            A = c1.number_input("Ancho (m)", 0.8, 3.5, float(dd[0]), 0.05)
            Hs = c2.number_input("Alto (m)", 1.0, 3.5, float(dd[1]), 0.05)
            r_arch = st.number_input("Radio del arco (m)", 0.1, A / 2, min(0.5, A / 2), 0.05) if kind.startswith("Baúl") else None
            top_w = st.number_input("Ancho en el techo (m)", 0.3, A, round(0.8 * A, 2), 0.05) if kind.startswith("Trap") else None
            cut_type = st.selectbox("Cuele", VETA_CUTS, index=2)
            st.caption(CUT_NOTES.get(cut_type, ""))
            cut_x, cut_h = 0.0, None
            if DESM:
                cut_in_vein = two_stage = False
                if st.checkbox("Ubicar el cuele manualmente", False,
                               help="Por defecto el cuele va centrado, a ≈ 1/3 de la altura. Desplazarlo hacia un "
                                    "hastial o hacia abajo ayuda a orientar la salida y a reducir el lanzamiento."):
                    c1, c2 = st.columns(2)
                    cut_x = c1.number_input("Desplazamiento lateral (m)", -1.5, 1.5, 0.0, 0.05,
                                            help="Se limita para que el cuele no quede pegado al hastial.")
                    cut_h = c2.number_input("Altura del centro del cuele (m)", 0.3, 3.0, round(0.4 * Hs, 2), 0.05)
            else:
                cut_in_vein = st.checkbox("Ubicar el cuele en la veta", True,
                                          help="La veta, más fracturada o blanda, funciona como alivio natural.")
                two_stage = st.checkbox("Voladura en dos etapas (veta + desquinche de caja)", False,
                                        help="Primero se vuela y limpia la veta; luego la caja. Reduce la dilución.")
            burn_sp = st.number_input("Distancia entre ejes del cuele (m)", 0.06, 0.30, 0.12, 0.01) \
                if cut_type in ("Quemado simple", "Quemado Sarrois") and st.checkbox("Fijar distancia del cuele") else None
        with st.sidebar.expander("Perforación", expanded=False):
            eq = st.selectbox("Equipo", list(EQUIPOS)[:3])
            E = EQUIPOS[eq]
            c1, c2 = st.columns(2)
            d1 = c1.number_input("Ø de perforación (mm)", 22.0, 51.0, float(E["d1"]), 1.0)
            d2 = c2.number_input("Ø del vacío (mm)", 22.0, 102.0, float(E["d1"]) + 7, 1.0)
            NB = c1.number_input("N° de vacíos", 1, 4, 2)
            bar = c2.selectbox("Barra (pies)", E["bars"], index=E["bars"].index(E["bar"]))
            eff = st.slider("Eficiencia de perforación (%)", 70, 100, 90)
            Fv = st.number_input("Error de perforación F (m)", 0.0, 0.3, 0.03, 0.005)
        with st.sidebar.expander("Explosivo y carga", expanded=False):
            exm = explosive_picker("Explosivo", "Emulnor 1000", '7/8" x 7"', "am")
            exc = explosive_picker("Explosivo de corona", "Emulnor 500", '1" x 8"', "ac")
            charge_method = st.selectbox("Método de carga", CHARGE_METHODS, index=1)
            contour_mode = st.selectbox("Contorno", ["Sin voladura controlada (f = 1.2, S/B = 1.25)",
                                                     "Voladura controlada (S = K·Ø1, q = 90·Ø1²)"],
                                        index=1 if (DESM and perm) else 0, key=f"cont{DESM and perm}")
            beff = st.slider("Eficiencia de voladura (%)", 70, 100, 90) / 100
        if DESM:
            with st.sidebar.expander("Acarreo y programa de avance", expanded=False):
                c1, c2 = st.columns(2)
                ob_d = c1.slider("Sobreexcavación (%)", 0, 40, 5 if contour_mode.startswith("Voladura") else 15,
                                 key=f"obd{contour_mode[:3]}") / 100
                sw_d = c2.slider("Esponjamiento (%)", 20, 80, 50) / 100
                hv = st.selectbox("Medio de acarreo", list(ACARREO), index=1)
                cap = ACARREO[hv] if ACARREO[hv] else st.number_input("Capacidad (m³)", 0.02, 10.0, 0.5, 0.01)
                fill = c1.slider("Factor de llenado", 0.5, 1.0, 0.9, 0.05)
                L_lab = c2.number_input("Longitud de la labor (m)", 1.0, 5000.0, 100.0, 5.0)
                spg = c1.number_input("Disparos por guardia", 1, 3, 1)
                gpd = c2.number_input("Guardias por día", 1, 3, 2)
        key = f"{labor}|{cut_type}"
        ov = ss.overrides.get(key, (None, None))
        section = build_section(kind, A, Hs, r_arch, top_w)
        drill = Drill(d1_mm=d1, d2_mm=d2, NB=int(NB), bar_m=bar * 0.3048, eff=eff / 100, F_value=Fv)
        opts = AvanceOptions(cut_type=cut_type, burn_spacing=burn_sp, contour_mode=contour_mode, blast_eff=beff,
                             charge_method=charge_method, narrow=True, B_prod_override=ov[0], SB_prod=ov[1] or 1.25,
                             cut_x=cut_x, cut_height=cut_h)
        if DESM:
            res = run_avance_desmonte(section, rock, drill, exm, exc, opts, tipo_d,
                                      dict(swell=sw_d, overbreak=ob_d, cap_m3=cap, fill=fill),
                                      dict(length=L_lab, shots_shift=int(spg), shifts_day=int(gpd)), cost)
        else:
            res = run_avance_veta(section, rock, drill, exm, exc, opts, vein, cut_in_vein, two_stage, cost)
        main_exp = exm
    else:
        with st.sidebar.expander("Método de explotación", expanded=True):
            label = st.selectbox("Método", list(VETA_PROD))
            is_long = "largos" in label or "abanico" in label
            wh = vein.horiz_width
            po = ProdOptions()
            c1, c2 = st.columns(2)
            if "mantos" in label:
                po.room = po.width = c1.number_input("Ancho de la cámara (m)", 2.0, 20.0, 6.0, 0.5)
                po.height = c2.number_input("Altura de minado (m)", 0.8, 6.0, max(1.5, round(vein.width + 0.3, 1)), 0.1)
                po.length = c1.number_input("Longitud de barreno (m)", 0.6, 4.0, 1.5, 0.1)
            else:
                po.width = c1.number_input("Ancho de minado (m)", 0.4, 10.0, float(round(max(wh + 0.3, 0.9), 2)), 0.05)
                if "abanico" in label:
                    po.up_h = po.height = c2.number_input("Altura del abanico (m)", 3.0, 30.0, 10.0, 0.5)
                    po.drift_w = c1.number_input("Ancho del subnivel (m)", 1.5, 4.0, 2.4, 0.1)
                    po.drift_h = c2.number_input("Alto del subnivel (m)", 1.5, 4.0, 2.4, 0.1)
                    po.ce0_key = st.selectbox("Tipo de roca (Tabla 24.4)", list(CE0_TABLE), index=1)
                    po.length = c1.number_input("Longitud del tajeo (m)", 3.0, 100.0, 15.0, 0.5)
                elif "largos" in label:
                    po.height = c2.number_input("Altura entre subniveles (m)", 4.0, 30.0, 12.0, 0.5)
                    po.length = c1.number_input("Longitud del tajeo (m)", 3.0, 100.0, 15.0, 0.5)
                    po.dip_deg = dip
                elif "breasting" in label:
                    po.height = c2.number_input("Altura de la tajada (m)", 1.0, 4.5, 2.0, 0.1)
                    po.length = c1.number_input("Longitud de barreno (m)", 0.6, 4.0, 1.5, 0.1)
                else:
                    po.height = c2.number_input("Altura de la tajada (m)", 1.0, 4.5, 2.0, 0.1)
                    po.length = c1.number_input("Longitud del tajeo (m)", 2.0, 100.0, 10.0, 0.5)
                    po.incl_deg = st.slider("Inclinación de los barrenos (°)", 50, 90, 80 if "shrinkage" in label else 60)
            swell = st.slider("Esponjamiento (%)", 20, 60, 35) / 100 if "shrinkage" in label else 0.35
        with st.sidebar.expander("Perforación y esquema", expanded=False):
            eqs = list(EQUIPOS)[3:] if is_long else list(EQUIPOS)[:3]
            eq = st.selectbox("Equipo", eqs, index=0 if is_long else (1 if "realce" in label or "shrinkage" in label else 0))
            E = EQUIPOS[eq]
            c1, c2 = st.columns(2)
            po.D_mm = c1.number_input("Ø de perforación (mm)", 22.0, 89.0, float(E["d1"]), 1.0)
            crit_def = "Consumo específico (LJ §4, subniveles)" if "abanico" in label else (
                "Langefors (LJ cap. 20 y 24)" if "largos" in label else "Holmberg — tajeo (Carpio / LJ cap. 22)")
            po.criterion = st.selectbox("Criterio de burden", CRITERIA, index=CRITERIA.index(crit_def))
            fdef = 1.45 if ("realce" in label or "shrinkage" in label or "mantos" in label) else (1.20 if "breasting" in label else 1.0)
            po.f = c1.number_input("Factor de fijación f", 0.8, 1.6, fdef, 0.05)
            po.SB = c2.number_input("S/B", 1.0, 2.0, 2.0 if "abanico" in label else 1.25, 0.05)
            po.rows_blast = c1.number_input("Filas por voladura", 1, 30, 2 if not is_long else 3)
            po.col_ratio = c2.number_input("Columna / fondo", 0.3, 1.0, 0.5, 0.05)
            po.q_deck_max = c1.number_input("Carga máx. por tramo (kg)", 5.0, 300.0, 50.0, 5.0)
            po.alpha = c2.number_input("Desviación (m/m)", 0.0, 0.05, 0.01, 0.005)
            po.e_coll = c1.number_input("Error de emboquille (m)", 0.0, 0.3, 0.03, 0.01)
            po.k_T = c2.number_input("Factor de retacado", 0.5, 2.0, 1.0, 0.1)
            po.staggered = st.checkbox("Malla al tresbolillo", True)
            pexp = explosive_picker("Explosivo", "Emulsión a granel" if po.D_mm >= 51 else "Emulnor 1000", '1" x 8"',
                                    f"pe{list(VETA_PROD).index(label)}")
        key = f"P|{label}"
        ov = ss.overrides.get(key, (None, None))
        po.B_override = ov[0]
        if ov[1]:
            po.SB = ov[1]
        res = run_prod_veta(rock, pexp, po, vein, label, swell, cost)
        main_exp = pexp
except Exception as e:  # noqa: BLE001
    err = e

if res is None:
    st.error(f"No se pudo generar el diseño con estos datos: {err}")
    st.stop()

H = res.holes
sig = (labor, res.metodo, len(H), round(float(H.x.sum()), 3), round(float(H.kg.sum()), 3), system)
if ss.seq_sig != sig:
    ss.seq = None
S = res.summary
V = res.p.get("veta", {})
DM = res.p.get("desmonte", {})

# =============================================================================
st.title("Voladura en vetas angostas — " + {"Avance en veta": "avance en veta", "Avance en desmonte": "avance en desmonte",
                                             "Producción en veta": "producción"}[labor])
st.caption((tipo_d + " · " + res.metodo) if DESM else (res.metodo if labor.startswith("Avance") else label))
m = st.columns(6)
m[0].metric("Taladros + vacíos", f"{len(H)} + {0 if res.relief is None else len(res.relief)}")
if "Avance por disparo (m)" in S:
    m[1].metric("Avance", f"{S['Avance por disparo (m)']:.2f} m")
else:
    m[1].metric("Volumen por voladura", f"{S['Volumen roto (m³)']:.1f} m³")
m[2].metric("Explosivo", f"{S['Explosivo total (kg)']:.1f} kg")
if DESM:
    m[3].metric("Desmonte/disparo", f"{DM['T']:.1f} t")
    m[4].metric("Carga (kg/m³)", f"{S['Factor de carga (kg/m³)']:.2f}")
    m[5].metric("Costo (US$/m)", f"{S.get('Costo por metro de avance (US$/m)', float('nan')):.0f}")
else:
    m[3].metric("Mineral enviado", f"{V['T_ore_pile']:.1f} t")
    m[4].metric("Dilución", f"{V['dilucion'] * 100:.0f} %")
    m[5].metric(f"Ley diluida ({vein.unit})", f"{V['ley_diluida']:.2f}")
if ss.overrides.get(key, (None, None))[0]:
    st.info(f"Burden impuesto por la optimización de P80: B = {ss.overrides[key][0]:.2f} m.")
for w in dict.fromkeys(res.warnings):
    st.warning(w)

tabs = st.tabs(["Memoria de cálculo", "Acarreo y programa" if DESM else "Veta y dilución", "Fragmentación (P80)", "Vista 2D", "Vista 3D y carguío",
                "Secuenciamiento", "Resumen y exportación"])

with tabs[0]:
    if res.p.get("note"):
        st.info(res.p["note"])
    if res.p.get("cut_note"):
        st.caption("Cuele: " + res.p["cut_note"])
    if labor.startswith("Avance") and (ASSETS / "cueles_1x1.5.png").exists():
        with st.expander("Cueles quemados para labores de 1.0 × 1.5 m (López Jimeno, cap. 22)"):
            st.image(str(ASSETS / "cueles_1x1.5.png"), width="stretch")
    log = res.log.df()
    for fase in log.Fase.unique():
        with st.expander(fase, expanded=True):
            st.dataframe(log[log.Fase == fase].drop(columns="Fase"), hide_index=True, width="stretch")

with tabs[1]:
    if DESM:
        c = st.columns(4)
        c[0].metric("Desmonte por disparo", f"{DM['T']:.1f} t")
        c[1].metric("Volumen esponjado", f"{DM['V_esp']:.2f} m³")
        c[2].metric("Viajes por disparo", f"{DM['trips']}")
        c[3].metric("Explosivo por metro", f"{DM['kg_m']:.2f} kg/m")
        c = st.columns(4)
        c[0].metric("Disparos para la labor", f"{DM['n_shots']}")
        c[1].metric("Días de avance", f"{DM['days']:.1f}")
        c[2].metric("Explosivo total", f"{S['Explosivo total de la labor (kg)']:.0f} kg")
        c[3].metric("Costo PyV de la labor", f"US$ {S.get('Costo de perforación y voladura de la labor (US$)', 0):,.0f}")
        adv_ = S["Avance por disparo (m)"]
        n_ = DM["n_shots"]
        if n_:
            dist = np.minimum(np.arange(1, n_ + 1) * adv_, L_lab)
            days_ = np.arange(1, n_ + 1) / (spg * gpd)
            fig = go.Figure(go.Scatter(x=days_, y=dist, mode="lines", line=dict(width=3, color="#1f4e9c"),
                                       name="Avance acumulado"))
            fig.update_layout(height=320, xaxis_title="Días", yaxis_title="Avance acumulado (m)", plot_bgcolor="white",
                              margin=dict(l=10, r=10, t=30, b=10), title="Programa de avance")
            st.plotly_chart(fig, width="stretch")
        cst = {k.replace("Costo ", "").replace(" (US$)", ""): S[k] for k in
               ["Costo perforación (US$)", "Costo explosivo (US$)", "Costo accesorios (US$)"] if k in S}
        if cst:
            fig = go.Figure(go.Bar(x=list(cst), y=[v / adv_ for v in cst.values()], marker_color="#8c8c8c"))
            fig.update_layout(height=280, yaxis_title="US$ por metro de avance", plot_bgcolor="white",
                              margin=dict(l=10, r=10, t=30, b=10), title="Costo de perforación y voladura por metro")
            st.plotly_chart(fig, width="stretch")
        st.caption("El tiempo de ciclo real depende también de la limpieza, el sostenimiento y la ventilación; los "
                   "días calculados sólo consideran el número de disparos por guardia.")
    else:
        c = st.columns(4)
        c[0].metric("Mineral in situ", f"{V['T_o']:.1f} t")
        c[1].metric("Desmonte", f"{V['T_w']:.1f} t")
        c[2].metric("Dilución", f"{V['dilucion'] * 100:.1f} %")
        c[3].metric("Contenido metálico", V["metal_txt"])
        if labor.startswith("Avance"):
            from core.veta import balance
            one = balance(V["T_o"], V["T_w"], vein, False)
            two = balance(V["T_o"], V["T_w"], vein, True)
            comp = pd.DataFrame({
                "Frente completo": [one["T_ore_pile"], one["dilucion"] * 100, one["ley_diluida"], len(H), H.kg.sum()],
                "Dos etapas (veta + desquinche)": [two["T_ore_pile"], two["dilucion"] * 100, two["ley_diluida"],
                                                   len(H), H.kg.sum()]},
                index=["Enviado como mineral (t)", "Dilución (%)", f"Ley diluida ({vein.unit})", "Taladros", "Explosivo (kg)"])
            st.subheader("Comparación: frente completo vs voladura en dos etapas")
            st.dataframe(comp.round(2), width="stretch")
            st.markdown("En dos etapas se vuela primero la veta (con el cuele en el mineral), se limpia el mineral y luego "
                        "se dispara la caja (desquinche) hacia el hueco dejado. La dilución queda limitada a la "
                        "sobrerrotura; a cambio el ciclo toma más tiempo y se manipulan dos disparos.")
            if V.get("two_stage"):
                g = H.groupby("etapa").agg(Taladros=("id", "count"), kg=("kg", "sum"))
                g.index = ["Etapa 1 — veta", "Etapa 2 — caja"][:len(g)]
                st.dataframe(g.round(2), width="stretch")
        fig = go.Figure(go.Pie(labels=["Mineral", "Desmonte"], values=[V["T_o"], V["T_w"]], hole=0.5,
                               marker=dict(colors=["#e6b41e", "#8c8c8c"]), sort=False))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

with tabs[2]:
    fp = res.frag
    route = st.radio("Ruta", ["Ruta A — Kuz-Ram", "Ruta B — Modelo ML híbrido"], horizontal=True)
    c1, c2, c3, c4 = st.columns(4)
    ref = c1.selectbox("Referencia (Carpio 6.1.7)", list(P80_REFERENCIAS))
    rng = P80_REFERENCIAS[ref]
    P80_t = c2.number_input("P80 objetivo (cm)", 2.0, 100.0, float(np.mean(rng)) if rng else 15.0, 0.5)
    grizz = c3.number_input("Abertura de la parrilla (cm)", 5.0, 100.0, 20.0, 1.0,
                            help="D.S. 024-2016-EM, Art. 292 e): parrillas de echaderos con aberturas ≤ 20 cm "
                                 "(≤ 50 cm con equipos de bajo perfil). Verifique el texto vigente.")
    B_lim = float(fp.get("B_limit") or fp["B"] * 1.5)
    SB_now = fp["S"] / fp["B"] if fp["B"] > 0 else 1.25
    B_min = c4.number_input("Burden mínimo (m)", 0.1, max(0.15, B_lim - 0.05), float(min(max(0.1, 0.4 * B_lim), B_lim - 0.1)),
                            0.05)
    best_B = best_SB = None
    if route.startswith("Ruta A"):
        with st.expander("Factor de roca A (Lilly)"):
            cc = st.columns(3)
            rmd = cc[0].selectbox("RMD", list(fr.RMD_OPTS), index=1)
            jps = cc[1].selectbox("JPS", list(fr.JPS_OPTS), index=1)
            jpa = cc[2].selectbox("JPA", list(fr.JPA_OPTS), index=1)
            Arf, parts = fr.rock_factor(rmd, jps, jpa, rock.rho, rock.E, rock.UCS)
            st.caption(f"A = {Arf:.2f} · " + " · ".join(f"{k} = {v:.1f}" for k, v in parts.items()))
        k = fr.KuzRamInput(A=Arf, q=fp["q"], Lc=fp["Lc"], H=fp["H"], d_mm=fp["d_mm"], W=fp["W"], rws_pct=fp["rws"],
                           SB=SB_now, K_ref=fp.get("K_override"), B_ref=fp["B"] if fp.get("K_override") else None)
        cur = fr.kuzram(fp["B"], k, fp["S"])
        ovs = oversize_pct(cur["X50"], cur["n"], grizz)
        c = st.columns(5)
        c[0].metric("B actual", f"{fp['B']:.2f} m")
        c[1].metric("X50", f"{cur['X50']:.1f} cm")
        c[2].metric("n", f"{cur['n']:.2f}")
        c[3].metric("P80", f"{cur['P80']:.1f} cm", f"{cur['P80'] - P80_t:+.1f}", delta_color="inverse")
        c[4].metric(f"Sobretamaño > {grizz:.0f} cm", f"{ovs:.1f} %",
                    help="Bancos que no pasan la parrilla y requieren voladura secundaria o rotura manual.")
        opt_r, curve = fr.optimize_burden_kuzram(k, P80_t, B_min, B_lim)
        curve["over"] = [oversize_pct(r.X50, r.n, grizz) for r in curve.itertuples()]
        g1, g2 = st.columns(2)
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=curve.B, y=curve.P80, name="P80 (cm)", line=dict(width=3, color="#c2416f")))
        f1.add_trace(go.Scatter(x=curve.B, y=curve.over, name=f"% > {grizz:.0f} cm", yaxis="y2",
                                line=dict(dash="dot", color="#1f4e9c")))
        f1.add_hline(y=P80_t, line=dict(dash="dash", color="#2ca02c"), annotation_text="Objetivo")
        f1.add_trace(go.Scatter(x=[opt_r["B"]], y=[opt_r["P80"]], mode="markers", name="Óptimo",
                                marker=dict(size=14, symbol="star", color="#ff7f0e")))
        f1.update_layout(height=360, xaxis_title="Burden B (m)", yaxis_title="P80 (cm)", plot_bgcolor="white",
                         yaxis2=dict(title="% sobretamaño", overlaying="y", side="right"),
                         margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
        g1.plotly_chart(f1, width="stretch")
        f2 = go.Figure()
        for lab, rr, col in [("Actual", cur, "#888"), ("Óptimo", opt_r, "#c2416f")]:
            x, R = fr.rosin_rammler_curve(rr["X50"], rr["n"], xmax=max(cur["P80"], opt_r["P80"], grizz) * 2)
            f2.add_trace(go.Scatter(x=x, y=R, name=f"{lab}: P80 = {rr['P80']:.1f} cm", line=dict(color=col, width=3)))
        f2.add_vline(x=grizz, line=dict(dash="dash", color="#1f4e9c"), annotation_text="Parrilla")
        f2.update_layout(height=360, xaxis_title="Tamaño (cm)", yaxis_title="% pasante", plot_bgcolor="white",
                         margin=dict(l=10, r=10, t=30, b=10))
        g2.plotly_chart(f2, width="stretch")
        if opt_r["converged"]:
            st.success(f"B óptimo = {opt_r['B']:.2f} m → P80 = {opt_r['P80']:.1f} cm, sobretamaño "
                       f"{oversize_pct(opt_r['X50'], opt_r['n'], grizz):.1f} %.")
        else:
            st.warning(f"El objetivo no se alcanza en [{B_min:.2f}, {B_lim:.2f}] m; lo más cercano es "
                       f"B = {opt_r['B']:.2f} m (P80 = {opt_r['P80']:.1f} cm).")
        best_B, best_SB = opt_r["B"], SB_now
    else:
        up = st.file_uploader("Modelo (.joblib / .pkl) — sólo archivos de confianza", type=["joblib", "pkl", "pickle"])
        demo = st.checkbox("Usar el sustituto de DEMOSTRACIÓN (no es un modelo real)", value=up is None)
        model = None
        if up is not None and not demo:
            try:
                import joblib
                model = joblib.load(io.BytesIO(up.getvalue()))
            except Exception:  # noqa: BLE001
                try:
                    model = pickle.loads(up.getvalue())
                except Exception as e:  # noqa: BLE001
                    st.error(f"No se pudo cargar el modelo: {e}")
        elif demo:
            model = fr.DemoSurrogate()
        if model is not None:
            cc = st.columns(3)
            target_kind = cc[0].radio("Predice", ["P80", "X50"], horizontal=True)
            unit_m = cc[1].selectbox("Unidad", ["cm", "mm", "m"])
            scale = {"cm": 1.0, "mm": 0.1, "m": 100.0}[unit_m]
            n_u = cc[2].number_input("n (si predice X50)", 0.5, 3.0, 1.5, 0.05)
            base = dict(d_mm=fp["d_mm"], q=fp["q"], H=fp["H"], Lc=fp["Lc"], Lt=fp["H"] - fp["Lc"], RMR=RMR, GSI=GSI,
                        RQD=RQD, UCS=UCS, rho_r=rho_r, VOD=main_exp.vod, rho_e=main_exp.rho, RWS=main_exp.rws * 100,
                        A_sec=S.get("Área de la sección (m²)", 0.0))
            names = fr.model_feature_names(model)
            if names and [f for f in names if f not in fr.ML_VARIABLES]:
                st.error("Variables no disponibles: " + ", ".join(f for f in names if f not in fr.ML_VARIABLES))
                names = None
            if not names:
                names = st.multiselect("Orden de variables", list(fr.ML_VARIABLES),
                                       default=["B", "S", "d_mm", "q", "Lc", "H", "RMR", "RWS"])
            if names:
                pnow, _ = fr.ml_predict(model, names, fr.ml_features(fp["B"], SB_now, base), scale, target_kind, n_u)
                st.metric("P80 con el diseño actual", f"{pnow:.1f} cm")
                SBr = st.slider("Rango S/B", 1.0, 2.0, (1.0, 1.6), 0.05)
                mkey = (labor, res.metodo, P80_t, B_min, round(B_lim, 3), SBr, tuple(names), unit_m, target_kind)
                if st.button("Optimizar con Evolución Diferencial"):
                    ss.ml_res = (mkey,) + fr.optimize_ml(model, names, base, P80_t, (B_min, B_lim), SBr, scale,
                                                         target_kind, n_u, 0.15, 0)
                if ss.get("ml_res") and ss.ml_res[0] == mkey:
                    _, rr, _h = ss.ml_res
                    st.metric("B óptimo", f"{rr['B']:.2f} m  ·  S/B {rr['SB']:.2f}  ·  P80 {rr['P80']:.1f} cm")
                    best_B, best_SB = rr["B"], rr["SB"]
    c1, c2 = st.columns(2)
    if c1.button("Aplicar burden óptimo", type="primary", disabled=best_B is None):
        ss.overrides[key] = (float(best_B), float(best_SB))
        st.rerun()
    if c2.button("Restablecer burden del método", disabled=key not in ss.overrides):
        ss.overrides.pop(key, None)
        st.rerun()

with tabs[3]:
    c1, c2, c3 = st.columns(3)
    lab2 = c1.selectbox("Etiqueta", ["id", "periodo", "kg", "ninguna"],
                        format_func={"id": "N° de taladro", "periodo": "N° de retardo", "kg": "Carga (kg)",
                                     "ninguna": "Sin etiqueta"}.get)
    cb = c2.radio("Color", ["zona", "etapa"], horizontal=True, format_func={"zona": "Por zona", "etapa": "Por etapa"}.get) \
        if "etapa" in H else "zona"
    vec = c3.checkbox("Vectores de salida (con secuencia)")
    st.plotly_chart(viz.view_2d(res, lab2, ss.seq, vec and ss.seq is not None, True, color_by=cb), width="stretch")

with tabs[4]:
    c1, c2, c3 = st.columns(3)
    sh = c1.checkbox("Labor / bloque", True)
    si = c2.checkbox("Iniciación", True)
    stt = c3.checkbox("Taco", True)
    st.plotly_chart(viz.view_3d(res, system, primer, sh, si, stt, None, ss.seq), width="stretch")
    st.plotly_chart(viz.hole_profiles(res, primer), width="stretch")

with tabs[5]:
    prod = not labor.startswith("Avance")
    c1, c2, c3 = st.columns(3)
    with c1:
        stage_opts = ["Completa"] + (["Etapa 1 — veta", "Etapa 2 — caja"] if V.get("two_stage") else [])
        target = st.radio("Voladura a secuenciar", stage_opts, horizontal=True)
        meth = st.radio("Método", (["Por filas", "MILP", "Heurística"] if prod else ["MILP", "Heurística"]),
                        format_func={"Por filas": "Por filas", "MILP": "MILP exacto (HiGHS)", "Heurística": "Heurística"}.get)
        cut_single = st.checkbox("Un taladro por retardo en el cuele", not prod)
    with c2:
        nmax = st.number_input("Máx. taladros por retardo", 1, 50, 3 if not prod else 6)
        qmax = st.number_input("Q máx por retardo (kg)", 0.5, 500.0, 8.0 if not prod else 60.0, 0.5)
        link = st.slider("Radio de alivio (× B)", 1.2, 3.0, 1.6, 0.1)
    with c3:
        cfg = dict(ms_m=ms_por_metro(RMR))
        carpio_gap = False
        if system.startswith("Carmex"):
            cfg.update(burn=st.number_input("Combustión de la mecha (s/m)", 80.0, 250.0, 150.0, 5.0,
                                            help="Valor referencial: use el del lote del fabricante."),
                       dt_s=st.number_input("Intervalo entre salidas (s)", 0.5, 10.0, 2.0, 0.5),
                       tail=st.number_input("Mecha fuera del taladro (m)", 0.1, 1.5, 0.40, 0.05),
                       primer=primer, Lt=float(H.Lt.median()))
            Lmin = st.number_input("Longitud mínima de carmex (m)", 0.5, 5.0, 1.50, 0.05,
                                   help="D.S. 024-2016-EM, Art. 237 c): longitud mínima de 1.50 m (verifique el texto vigente).")
        elif system == "Electrónico":
            cfg.update(t0=0.0, dt_ms=st.number_input("Intervalo (ms)", 1.0, 1000.0, 50.0))
        elif system.endswith("tabla propia"):
            txt = st.text_area("Tiempos (ms)", ", ".join(str(t) for t in EXSANEL_LP))
            cfg["table"] = [float(t) for t in txt.replace(";", ",").split(",") if t.strip()]
        if not prod and not system.startswith(("Carmex", "Electrónico")):
            carpio_gap = st.checkbox(f"Tiempo mínimo en el arranque (Carpio: {cfg['ms_m']} ms/m × L)", True)
        if prod:
            cfg.update(row_ms=st.number_input("Entre filas (ms)", 1.0, 500.0, 75.0),
                       hole_ms=st.number_input("Dentro de la fila (ms)", 0.0, 200.0, 10.0))
    if st.button("Calcular secuencia", type="primary"):
        Hs_ = H if target == "Completa" else H[H.etapa == (1 if target.startswith("Etapa 1") else 2)]
        src = res.sources if not target.startswith("Etapa 2") else res.p.get("stage2_sources", res.sources)
        qu = Hs_.cargas.map(sq.deck_max) if meth == "Por filas" else Hs_.kg
        if qu.max() > qmax:
            st.error(f"Una carga de {qu.max():.2f} kg supera Q máx ({qmax:.1f} kg).")
        else:
            opt = dict(classic=True, qmax=qmax, nmax=int(nmax), cut_single=cut_single and not target.startswith("Etapa 2"),
                       method=meth, time_limit=20, link=link, carpio_gap=carpio_gap, simultaneous=res.simultaneous)
            seq, info = sq.run_sequencing(Hs_, src, opt, system, cfg, RMR)
            if system.startswith("Carmex"):
                seq["mecha_m"] = np.maximum(seq["mecha_m"], Lmin)
            ss.seq, ss.seq_sig, ss.seq_info, ss.seq_qmax = seq, sig, info, qmax
            st.rerun()
    if ss.seq is not None:
        seq, info = ss.seq, ss.seq_info
        c = st.columns(4)
        c[0].metric("Retardos", info["n_periods"])
        c[1].metric("Duración", f"{seq.t_ms.max() / 1000:.1f} s")
        per_delay = seq.groupby("t_ms").kg_carga.sum() if "Por filas" in info["method"] else seq.groupby("periodo").kg.sum()
        c[2].metric("Carga máx./retardo", f"{per_delay.max():.2f} kg")
        c[3].metric("Taladros", len(seq))
        st.caption("Método: " + info["method"])
        g1, g2 = st.columns([1.1, 1])
        g1.plotly_chart(viz.sequence_animation(res, seq), width="stretch")
        g2.plotly_chart(viz.view_2d(res, "periodo", seq, True, False, height=600), width="stretch")
        cols = ["orden", "id", "zona", "periodo", "n_detonador", "t_ms", "kg", "x", "y"] + (["mecha_m"] if "mecha_m" in seq else [])
        if "mecha_m" in seq:
            st.info("Con carmex el orden de chispeo es la columna 'orden'. La combustión de la mecha tiene una dispersión "
                    "de alrededor de ±10 %, así que el escalonamiento por longitud es sólo referencial.")
        st.dataframe(seq[cols].round(3), hide_index=True, width="stretch")
        st.download_button("Descargar secuencia (CSV)", seq[cols].to_csv(index=False).encode("utf-8"), "secuencia.csv")

with tabs[6]:
    c1, c2 = st.columns([1, 1.2])
    c1.dataframe(pd.DataFrame({"Indicador": list(S), "Valor": [round(float(v), 3) for v in S.values()]}),
                 hide_index=True, width="stretch")
    zt = H.groupby("zona", sort=False).agg(Taladros=("id", "count"), Cartuchos=("n_cart", "sum"),
                                          Explosivo=("explosivo", "first"), kg=("kg", "sum"))
    c2.dataframe(zt.round(2), width="stretch")
    cols = ["id", "zona", "x", "y", "L", "B", "S", "explosivo", "n_cart", "kg"] + (["etapa"] if "etapa" in H else [])
    st.dataframe(H[cols].round(3), hide_index=True, width="stretch")
    d1c, d2c, d3c = st.columns(3)
    d1c.download_button("Taladros (CSV)", H[cols].to_csv(index=False).encode("utf-8"), "taladros.csv")
    d2c.download_button("Memoria de cálculo (CSV)", res.log.df().to_csv(index=False).encode("utf-8"), "memoria.csv")
    d3c.download_button("Resumen (JSON)", json.dumps({k: float(v) for k, v in S.items()}, ensure_ascii=False,
                                                     indent=2).encode("utf-8"), "resumen.json")
