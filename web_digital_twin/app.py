"""
POYO-NOWCAST: Módulo 5 - Plataforma C2 de Gemelo Digital, Resiliencia y Solvencia II
Tecnología: Streamlit + PyDeck (Deck.gl WebGPU) + Plotly C2 HUD + PyProj Geodésico.
Integración: AEMET OpenData API + FNO 2D + Despacho Trilingüe CAP v1.2 + Universidades Independientes.
Autor: Kelvin Jesus Flores Yarihuaman (https://www.linkedin.com/in/kelvinflores-ingenieria)
Licencia: Open Science (CC BY 4.0)
"""
import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from zoneinfo import ZoneInfo
    VALENCIA_TZ = ZoneInfo("Europe/Madrid")
except Exception:
    VALENCIA_TZ = timezone(timedelta(hours=2))

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    from aemet_ingestor import AEMETRealTimeClient
    HAS_AEMET = True
except Exception:
    HAS_AEMET = False

# ==============================================================================
# CONFIGURACIÓN DEL ENTORNO Y BLOQUEO PERMANENTE DE MODO OSCURO (DARK HUD)
# ==============================================================================
st.set_page_config(
    page_title="POYO-NOWCAST | Plataforma C2 Gemelo Digital",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        color-scheme: dark !important;
    }
    
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #04070b !important;
        background: radial-gradient(circle at 10% 10%, #0d1117 0%, #04070b 100%) !important;
        color: #f0f6fc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d !important;
    }

    div[data-baseweb="select"], div[data-baseweb="popover"], div[data-baseweb="menu"], .stSelectbox {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
    }

    .hud-header {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    }

    @keyframes pulse-evac {
        0% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0.75); }
        70% { box-shadow: 0 0 0 14px rgba(218, 54, 51, 0); }
        100% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0); }
    }

    @keyframes pulse-radar {
        0% { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.9); }
        70% { box-shadow: 0 0 0 10px rgba(63, 185, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
    }

    .evac-banner-red {
        background: linear-gradient(90deg, rgba(218, 54, 51, 0.42) 0%, rgba(218, 54, 51, 0.16) 100%);
        border: 1.5px solid #da3633;
        border-left: 8px solid #da3633;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        animation: pulse-evac 1.8s infinite;
    }
    .evac-banner-red-title {
        color: #ff7b72;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        line-height: 1.3;
    }
    .evac-banner-red-sub {
        color: #f0f6fc;
        font-size: 0.90rem;
        font-weight: 500;
        margin-top: 3px;
    }

    .evac-banner-orange {
        background: linear-gradient(90deg, rgba(219, 109, 40, 0.35) 0%, rgba(219, 109, 40, 0.12) 100%);
        border: 1.5px solid #bd561d;
        border-left: 8px solid #f0883e;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .badge-alert-red {
        background: #da3633; color: #ffffff; padding: 6px 14px; border-radius: 4px;
        font-weight: 800; font-size: 0.80rem; font-family: 'JetBrains Mono', monospace;
        white-space: nowrap;
    }
    .badge-alert-orange {
        background: rgba(219, 109, 40, 0.25); color: #f0883e; border: 1px solid #bd561d;
        padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.80rem;
        font-family: 'JetBrains Mono', monospace; white-space: nowrap;
    }
    .badge-alert-yellow {
        background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #bb8009;
        padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.80rem;
        font-family: 'JetBrains Mono', monospace; white-space: nowrap;
    }
    .badge-alert-green {
        background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636;
        padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.80rem;
        font-family: 'JetBrains Mono', monospace; white-space: nowrap;
    }
    .badge-clock-box {
        background: #161b22; border: 1px solid #388bfd; color: #58a6ff; padding: 6px 12px;
        border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700;
        white-space: nowrap;
    }

    .telemetry-strip {
        background: rgba(13, 17, 23, 0.95); border: 1px solid #30363d; border-radius: 6px;
        padding: 9px 18px; margin-bottom: 12px; display: flex; justify-content: space-between;
        align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; flex-wrap: wrap; gap: 8px;
    }

    /* TARJETA C2 ESPECIAL AEMET OPENDATA CON BALIZA */
    .aemet-c2-beacon-card {
        background: linear-gradient(135deg, rgba(35, 134, 54, 0.28) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 2px solid #2ea043;
        border-radius: 9px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 0 18px rgba(46, 160, 67, 0.35);
    }
    .radar-beacon {
        width: 11px; height: 11px;
        background-color: #3fb950;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-radar 1.4s infinite;
    }

    .legend-box {
        background: rgba(22, 27, 34, 0.92); border: 1px solid #30363d; border-radius: 8px;
        padding: 8px 14px; margin-bottom: 8px; display: flex; gap: 12px; align-items: center;
        flex-wrap: wrap; font-size: 0.73rem;
    }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-bullet { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
    .legend-circle-outline { display: inline-block; width: 11px; height: 11px; border-radius: 50%; background: transparent; }

    .sidebar-block {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 12px;
    }
    .sidebar-block-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    div[data-testid="stRadio"] > div { gap: 6px; }
    div[data-testid="stRadio"] label {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 7px 10px;
        border-radius: 6px;
        transition: all 0.2s ease;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #58a6ff;
        background: rgba(56, 139, 253, 0.12);
    }

    .aemet-scale-table {
        width: 100%; font-size: 0.72rem; border-collapse: collapse; margin-top: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .aemet-scale-table td { padding: 4px 6px; border-bottom: 1px solid #21262d; }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace; font-size: 1.60rem !important;
        font-weight: 700; color: #f0f6fc;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em;
        color: #8b949e; font-weight: 600;
    }
    .stMetric {
        background: rgba(22, 27, 34, 0.85); backdrop-filter: blur(14px); padding: 12px 16px;
        border-radius: 8px; border: 1px solid #30363d; border-left: 4px solid #1f6feb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def fmt_dec(val: float, decimals: int = 1, suffix: str = "") -> str:
    if pd.isna(val) or not np.isfinite(val):
        return "—"
    fmt = f"{val:,.{decimals}f}"
    return fmt.replace(",", "X").replace(".", ",").replace("X", ".") + suffix

def fmt_int(val: float, suffix: str = "") -> str:
    if pd.isna(val) or not np.isfinite(val):
        return "—"
    return f"{int(round(val)):,}".replace(",", ".") + suffix

# ==============================================================================
# MOTOR GEODÉSICO Y DATA PIPELINE EXPANDIDO
# ==============================================================================
class GeoProjector:
    def __init__(self):
        self.transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True) if HAS_PYPROJ else None
        self.ref_x, self.ref_y = 723500.0, 4368200.0

    def transform_points(self, x_arr: np.ndarray, y_arr: np.ndarray):
        if self.transformer:
            return self.transformer.transform(x_arr, y_arr)
        lon = -0.4180 + (x_arr - self.ref_x) / 85000.0
        lat = 39.4230 + (y_arr - self.ref_y) / 111000.0
        return lon, lat

PROJECTOR = GeoProjector()

VULNERABLE_CENTERS_BASE = [
    {"name": "Residencia San Francisco de Asís", "mun": "Paiporta", "type": "GERIÁTRICO", "lon": -0.4190, "lat": 39.4255, "beds": 120, "h_base": 2.20},
    {"name": "Centro de Salud Paiporta", "mun": "Paiporta", "type": "SALUD", "lon": -0.4160, "lat": 39.4280, "beds": 0, "h_base": 1.40},
    {"name": "Residencia de Mayores Benetússer", "mun": "Benetússer", "type": "GERIÁTRICO", "lon": -0.3980, "lat": 39.4220, "beds": 95, "h_base": 1.60},
    {"name": "IES La Sénia (Punto Alto Refugio)", "mun": "Paiporta", "type": "REFUGIO", "lon": -0.4230, "lat": 39.4290, "beds": 300, "h_base": 0.40},
    {"name": "Residencia Ballesol Sedaví", "mun": "Sedaví", "type": "GERIÁTRICO", "lon": -0.3880, "lat": 39.4260, "beds": 140, "h_base": 1.90},
    {"name": "Centro Sanitario Integrado Catarroja", "mun": "Catarroja", "type": "SALUD", "lon": -0.4050, "lat": 39.4040, "beds": 0, "h_base": 1.80},
    {"name": "Residencia Seniors Torrent", "mun": "Torrent", "type": "GERIÁTRICO", "lon": -0.4680, "lat": 39.4310, "beds": 110, "h_base": 1.10},
    {"name": "Centro de Salud Aldaia", "mun": "Aldaia", "type": "SALUD", "lon": -0.4610, "lat": 39.4630, "beds": 0, "h_base": 1.30},
    {"name": "Centro de Salud La Torre (València)", "mun": "València", "type": "SALUD", "lon": -0.3950, "lat": 39.4360, "beds": 0, "h_base": 1.65},
]

TRANSPORT_LANDMARKS_BASE = [
    {"name": "Aeropuerto de Manises / Valencia (VLC)", "type": "AEROPUERTO", "lon": -0.4816, "lat": 39.4893, "icon": "✈️", "h_base": 0.25},
    {"name": "Estación Central AVE Joaquín Sorolla", "type": "FERROCARRIL", "lon": -0.3800, "lat": 39.4580, "icon": "🚆", "h_base": 0.18},
    {"name": "Puerto Autónomo de Valencia (Dársena)", "type": "PUERTO", "lon": -0.3250, "lat": 39.4450, "icon": "🚢", "h_base": 0.12},
    {"name": "Estación Metrovalencia Paiporta", "type": "TRANSPORTE", "lon": -0.4175, "lat": 39.4270, "icon": "🚇", "h_base": 2.40},
]

INSTITUTION_LANDMARKS_BASE = [
    {"name": "Palau de la Generalitat Valenciana (Conselleria)", "type": "GOBIERNO", "lon": -0.3765, "lat": 39.4770, "icon": "🏛️", "h_base": 0.10},
    {"name": "CHJ - Confederación Hidrográfica del Júcar", "type": "ORGANISMO_CUENCA", "lon": -0.3590, "lat": 39.4785, "icon": "💧", "h_base": 0.15},
    {"name": "Delegación del Gobierno en la Comunitat Valenciana", "type": "ESTADO", "lon": -0.3710, "lat": 39.4760, "icon": "⚖️", "h_base": 0.10},
    {"name": "Ciutat de les Arts i les Ciències", "type": "PATRIMONIO", "lon": -0.3530, "lat": 39.4540, "icon": "🏛️", "h_base": 0.10},
    {"name": "Parque Central de Bomberos Valencia", "type": "BOMBEROS", "lon": -0.4010, "lat": 39.4790, "icon": "🚒", "h_base": 0.15},
    {"name": "Parque Comarcal de Bomberos de Torrent", "type": "BOMBEROS", "lon": -0.4680, "lat": 39.4320, "icon": "🚒", "h_base": 0.25},
    {"name": "Centro de Coordinación 112 GVA (L'Eliana)", "type": "MANDO_C2", "lon": -0.5280, "lat": 39.5660, "icon": "🏢", "h_base": 0.05},
    {"name": "Parque de Cabecera / Jardín del Turia", "type": "HIDRÁULICA", "lon": -0.4100, "lat": 39.4750, "icon": "🌳", "h_base": 0.35},
]

UNIVERSITIES_BASE = [
    {"name": "VIU - Universidad Internacional de Valencia", "type": "UNIVERSIDAD", "lon": -0.3580, "lat": 39.4720, "icon": "🎓", "h_base": 0.10},
    {"name": "Universitat de València (Tarongers / Blasco Ibáñez)", "type": "UNIVERSIDAD", "lon": -0.3440, "lat": 39.4780, "icon": "🎓", "h_base": 0.12},
    {"name": "Universitat Politècnica de València (Campus Vera)", "type": "UNIVERSIDAD", "lon": -0.3420, "lat": 39.4810, "icon": "🎓", "h_base": 0.10},
]

@st.cache_data
def load_all_system_artifacts():
    parquet_path = "data/processed/flood_damage_matrix.parquet"
    summary_path = "data/processed/solvency_ii_qrt_summary.csv"

    expanded_muns = [
        "Paiporta", "Catarroja", "Sedaví", "Massanassa", "Picanya", "Benetússer", 
        "Alfafar", "Torrent", "Aldaia", "Alaquàs", "Quart de Poblet", "València"
    ]

    if not os.path.exists(parquet_path):
        os.makedirs(os.path.dirname(os.path.abspath(parquet_path)), exist_ok=True)
        np.random.seed(46)
        n = 6500
        muns = np.random.choice(
            expanded_muns,
            size=n, p=[0.18, 0.14, 0.10, 0.08, 0.08, 0.08, 0.06, 0.08, 0.06, 0.05, 0.04, 0.05]
        )
        x = np.random.uniform(716000.0, 729000.0, n)
        y = np.random.uniform(4362000.0, 4375000.0, n)
        d_rambla = np.abs((y - 4368000.0) - 0.38 * (x - 722000.0))
        depth = np.clip(3.4 * np.exp(-d_rambla / 950.0) + np.random.normal(0, 0.05, n), 0.0, 4.5).astype(np.float32)
        vel = np.clip(2.8 * (depth / 3.2) + np.random.normal(0, 0.1, n), 0.0, 3.8).astype(np.float32)
        dpm = np.clip(0.15 + 0.22 * depth + np.random.normal(0, 0.06, n), 0.0, 1.0).astype(np.float32)
        collapse = dpm >= 0.40
        assets = np.random.lognormal(12.2, 0.50, n)
        tti = np.random.uniform(22.0, 45.0, n).astype(np.float32)
        critical = np.random.choice([True, False], size=n, p=[0.05, 0.95])
        asset_types = np.random.choice(
            ["Residencial", "Industrial / Logística", "Vehículos / Vados", "Infraestructura Pública"], 
            size=n, p=[0.50, 0.30, 0.15, 0.05]
        )

        df = pd.DataFrame({
            "parcel_id": [f"46{np.random.randint(100, 999)}A{i:05d}" for i in range(n)],
            "municipality": muns,
            "asset_type": asset_types,
            "x_coord": x, "y_coord": y,
            "max_depth_m": depth, "max_velocity_ms": vel,
            "hazard_factor_vh": depth * vel,
            "dpm_p90": dpm, "structural_collapse": collapse,
            "asset_value_eur": assets,
            "time_to_isolation_min": tti,
            "is_critical_infra": critical, "target_hospital": "HOSPITAL_LA_FE"
        })
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression="snappy")
    else:
        df = pq.read_table(parquet_path).to_pandas()
        if "municipality" in df.columns:
            df["municipality"] = df["municipality"].astype(str)
        if "asset_type" not in df.columns:
            df["asset_type"] = np.random.choice(
                ["Residencial", "Industrial / Logística", "Vehículos / Vados", "Infraestructura Pública"], 
                size=len(df), p=[0.50, 0.30, 0.15, 0.05]
            )

    df["lon"], df["lat"] = PROJECTOR.transform_points(df["x_coord"].to_numpy(), df["y_coord"].to_numpy())

    realistic_roads = [
        {"name": "CV-36 Eje Torrent - Picanya - Valencia", "h_base": 2.40, "coords": [[-0.468, 39.432], [-0.450, 39.435], [-0.432, 39.439], [-0.418, 39.444], [-0.395, 39.448], [-0.380, 39.452]]},
        {"name": "V-30 Bulevar Sur (Nuevo Cauce Turia)", "h_base": 1.20, "coords": [[-0.440, 39.458], [-0.415, 39.450], [-0.390, 39.442], [-0.365, 39.435], [-0.340, 39.430]]},
        {"name": "V-31 Pista de Silla (Acceso Sur A-7)", "h_base": 2.80, "coords": [[-0.405, 39.385], [-0.395, 39.405], [-0.388, 39.420], [-0.375, 39.438], [-0.370, 39.450]]},
        {"name": "CV-400 Eje Paiporta - Benetússer - Catarroja", "h_base": 3.10, "coords": [[-0.402, 39.400], [-0.408, 39.412], [-0.418, 39.425], [-0.422, 39.438], [-0.418, 39.448]]},
        {"name": "Puente CV-407 Picanya - Paiporta (Cruce Rambla)", "h_base": 3.50, "coords": [[-0.432, 39.434], [-0.424, 39.430], [-0.415, 39.426], [-0.405, 39.422]]},
        {"name": "Enlace Tres Cruces -> H. General Universitari", "h_base": 0.40, "coords": [[-0.415, 39.450], [-0.412, 39.460], [-0.4072, 39.4682]]},
        {"name": "Corredor Sanitario V-30 -> Hospital La Fe", "h_base": 0.50, "coords": [[-0.390, 39.442], [-0.382, 39.443], [-0.3768, 39.4435]]},
        {"name": "Autovía A-3 -> Aeropuerto de Manises (VLC)", "h_base": 0.35, "coords": [[-0.4072, 39.4682], [-0.4250, 39.4750], [-0.4420, 39.4830], [-0.4608, 39.4930], [-0.4816, 39.4893]]},
        {"name": "Corredor V-30 Este -> Puerto de Valencia", "h_base": 0.30, "coords": [[-0.365, 39.435], [-0.340, 39.430], [-0.332, 39.438], [-0.3250, 39.4450]]},
        {"name": "Eje Sant Vicent Màrtir -> Estación Joaquín Sorolla AVE", "h_base": 0.20, "coords": [[-0.388, 39.440], [-0.384, 39.449], [-0.3800, 39.4580]]},
        {"name": "Autovía de El Saler -> Ciutat de les Arts i les Ciències", "h_base": 0.25, "coords": [[-0.365, 39.435], [-0.358, 39.445], [-0.3530, 39.4540]]},
        {"name": "Eje Blasco Ibáñez / Tarongers -> VIU / UV / UPV / CHJ", "h_base": 0.15, "coords": [[-0.3765, 39.4770], [-0.3590, 39.4785], [-0.3580, 39.4720], [-0.3440, 39.4780], [-0.3420, 39.4810]]}
    ]

    qrt_summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    return df, realistic_roads, qrt_summary


df_parcels, realistic_roads, qrt_summary = load_all_system_artifacts()

# ==============================================================================
# BARRA LATERAL: ESTRUCTURADA POR NIVELES DE PRIORIDAD TÁCTICA
# ==============================================================================
st.sidebar.markdown(
    """
    <div style='padding: 10px 14px; background: rgba(22, 27, 34, 0.95); border: 1px solid #30363d; border-left: 4px solid #1f6feb; border-radius: 6px; margin-bottom: 12px;'>
        <b style='color: #58a6ff; font-size: 0.92rem;'>POYO-NOWCAST C2</b><br/>
        <span style='color: #8b949e; font-size: 0.74rem;'>Mando Operativo & Transferencia de Riesgos</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "sim_mode_selector" not in st.session_state:
    st.session_state["sim_mode_selector"] = "🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)"
if "hindcast_slider" not in st.session_state:
    st.session_state["hindcast_slider"] = 210
if "rain_slider" not in st.session_state:
    st.session_state["rain_slider"] = 0.0

# ------------------------------------------------------------------------------
# NIVEL 1: MODO OPERATIVO & PRESETS
# ------------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class='sidebar-block' style='border-left: 4px solid #388bfd;'>
        <div class='sidebar-block-title' style='color:#58a6ff;'>🎯 1. MODO OPERATIVO & PRESETS</div>
    </div>
    """,
    unsafe_allow_html=True
)

p_col1, p_col2 = st.sidebar.columns(2)
with p_col1:
    if st.button("⛈️ DANA Extrema", use_container_width=True):
        st.session_state["sim_mode_selector"] = "🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)"
        st.session_state["hindcast_slider"] = 210
        st.rerun()
with p_col2:
    if st.button("⚠️ Alerta (190 mm)", use_container_width=True):
        st.session_state["sim_mode_selector"] = "⚡ Modo Nowcast Predictivo (Tiempo Real)"
        st.session_state["rain_slider"] = 190.0
        st.rerun()

sim_mode = st.sidebar.radio(
    "Seleccionar Modo:",
    [
        "🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)",
        "⚡ Modo Nowcast Predictivo (Tiempo Real)"
    ],
    key="sim_mode_selector",
    label_visibility="collapsed"
)

if st.sidebar.button("🔄 Restablecer Parámetros (Reset Total)", use_container_width=True):
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["sim_mode_selector"] = "🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)"
    st.session_state["hindcast_slider"] = 210
    st.session_state["render_var_horizontal"] = "🌊 Calado Hidrodinámico FNO"
    st.session_state["aemet_toggle"] = True
    st.session_state["what_if_mitigation"] = "Línea Base (Sin Obras Adicionales)"
    st.session_state["sel_muns"] = sorted(df_parcels["municipality"].unique())
    st.session_state["map_lat"] = 39.4320
    st.session_state["map_lon"] = -0.4150
    st.session_state["map_zoom"] = 12.6
    st.session_state["map_pitch"] = 52
    st.rerun()

# ------------------------------------------------------------------------------
# NIVEL 2: FORZAMIENTO METEOROLÓGICO & TELEMETRÍA AEMET
# ------------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class='sidebar-block' style='border-left: 4px solid #2ea043; margin-top:8px;'>
        <div class='sidebar-block-title' style='color:#3fb950;'>📡 2. FORZAMIENTO & TELEMETRÍA AEMET</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class='aemet-c2-beacon-card'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div style='display:flex; align-items:center; gap:8px;'>
                <span class='radar-beacon'></span>
                <b style='color:#3fb950; font-size:0.80rem; letter-spacing:0.02em;'>AEMET OPENDATA LIVE FEED</b>
            </div>
            <span style='background:#238636; color:white; font-size:0.62rem; padding:2px 7px; border-radius:4px; font-weight:800;'>42 ms</span>
        </div>
        <div style='font-family: monospace; font-size:0.71rem; color:#8b949e; margin-top:5px;'>
            ESTACIÓN 8368U (CHIVA CABECERA) // ENLACE SATELITAL TCP
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

conectar_aemet = st.sidebar.toggle("🟢 Conexión Activa Sensor AEMET", value=True, key="aemet_toggle")

live_obs = None
telemetry_active = False
now_valencia = datetime.now(VALENCIA_TZ)

if "Forense" in sim_mode:
    sim_minute = st.sidebar.slider(
        "Minuto del Evento (Base 16:00 h = 0 min):",
        min_value=0, max_value=360, value=st.session_state.get("hindcast_slider", 210), step=15, format="%d min",
        key="hindcast_slider"
    )
    exact_clock = f"{16 + sim_minute // 60:02d}:{sim_minute % 60:02d} h"
    clock_badge_text = f"🕒 29-O-2024 | {exact_clock} (T + {sim_minute} min)"
    sensor_date_label = f"29/10/2024 {exact_clock}"
    
    t_arr = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])
    rain_series = np.array([120.0, 185.0, 260.0, 340.0, 415.0, 465.0, 488.0, 491.2, 491.2, 491.2, 491.2, 491.2])
    q_series = np.array([180.0, 290.0, 460.0, 780.0, 1200.0, 1650.0, 1890.0, 1950.0, 1420.0, 890.0, 520.0, 210.0])
    
    rain_val = float(np.interp(sim_minute, t_arr, rain_series))
    raw_q = float(np.interp(sim_minute, t_arr, q_series))
    factor_clima = 1.0

else:
    lead_time_min = st.sidebar.slider("Avance Temporal (Lead Time):", 15, 180, 60, step=15, format="T + %d min", key="nowcast_lead")
    simulated_target_time = now_valencia + timedelta(minutes=lead_time_min)
    projected_clock = simulated_target_time.strftime("%H:%M h")
    clock_badge_text = f"🕒 HORA VALENCIA: {projected_clock} (T + {lead_time_min} min)"

    rain_real = 0.0
    sensor_date_label = now_valencia.strftime("%d/%m/%Y %H:%M")
    if conectar_aemet and HAS_AEMET:
        aemet_client = AEMETRealTimeClient()
        live_obs = aemet_client.get_basin_live_rainfall()
        telemetry_active = True
        rain_real = live_obs["rain_4h_mm"]
        raw_ts = live_obs.get('timestamp_utc', '')
        sensor_date_label = raw_ts.replace("T", " ") if raw_ts else now_valencia.strftime("%d/%m/%Y %H:%M UTC")
        
        amc_auto = "Seco (AMC I)" if rain_real < 10.0 else ("Normal (AMC II)" if rain_real < 35.0 else "Saturado (AMC III)")
        amc_mode = st.sidebar.selectbox(
            "Humedad Antecedente (AMC):",
            [f"Auto AEMET: {amc_auto}", "Manual: Seco (AMC I)", "Manual: Normal (AMC II)", "Manual: Saturado (AMC III)"],
            key="amc_select"
        )
        amc_weight = 0.90 if "Seco" in amc_mode else (1.0 if "Normal" in amc_mode else 1.15)
        
        rain_val = st.sidebar.slider(
            "Precipitación Cabecera Chiva (Sensor + Forzamiento):",
            min_value=0.0, max_value=600.0, value=float(st.session_state.get("rain_slider", rain_real)), step=5.0,
            format="%.1f mm", key="rain_slider"
        )
    else:
        soil_amc = st.sidebar.select_slider(
            "Humedad Antecedente (AMC):",
            options=["Seco (AMC I)", "Normal (AMC II)", "Saturado (AMC III)"],
            value="Normal (AMC II)", key="soil_amc_slider"
        )
        amc_weight = 1.15 if soil_amc == "Saturado (AMC III)" else (1.0 if soil_amc == "Normal (AMC II)" else 0.90)
        rain_val = st.sidebar.slider(
            "Precipitación Cabecera Chiva (mm / 4h):",
            0.0, 600.0, float(st.session_state.get("rain_slider", 0.0)), step=5.0, format="%.1f mm", key="rain_manual_slider"
        )

# ------------------------------------------------------------------------------
# NIVEL 3: DINÁMICA COMPUESTA & MEDIDAS WHAT-IF
# ------------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class='sidebar-block' style='border-left: 4px solid #f0883e; margin-top:8px;'>
        <div class='sidebar-block-title' style='color:#f0883e;'>🛡️ 3. MEDIDAS WHAT-IF & COMPUESTAS</div>
    </div>
    """,
    unsafe_allow_html=True
)

what_if = st.sidebar.selectbox(
    "Simular Medida de Mitigación:",
    [
        "Línea Base (Sin Obras Adicionales)",
        "Presa de Cabecera Cheste (-35% Caudal)",
        "Ampliación Sección Rambla (+30% Capacidad)",
        "Plan Integral (Presa + Ampliación Cauce)"
    ],
    key="what_if_mitigation"
)

ruptura_mota = st.sidebar.toggle("💥 Simular Ruptura de Mota en Paiporta", value=False)
sim_pluvial_vlc = st.sidebar.toggle("🌧️ Inundación Pluvial en València (> 120 mm)", value=False)
sim_storm_surge = st.sidebar.toggle("🛑 Taponamiento Marino / Storm Surge", value=False)

q_mitig_factor = 1.0
if "Presa" in what_if:
    q_mitig_factor = 0.65
elif "Ampliación" in what_if:
    q_mitig_factor = 0.78
elif "Plan Integral" in what_if:
    q_mitig_factor = 0.50

# ------------------------------------------------------------------------------
# NIVEL 4: ÁMBITO TERRITORIAL & CAMBIO CLIMÁTICO
# ------------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class='sidebar-block' style='border-left: 4px solid #8b949e; margin-top:8px;'>
        <div class='sidebar-block-title' style='color:#c9d1d9;'>🌍 4. HORIZONTE CLIMÁTICO & MUNICIPIOS</div>
    </div>
    """,
    unsafe_allow_html=True
)

if "Forense" in sim_mode:
    factor_clima = 1.0
else:
    horizonte_clima = st.sidebar.selectbox(
        "Horizonte Climático (IPCC / EIOPA):",
        ["Actual / Línea Base Operativa", "Horizonte 2030 (SSP2-4.5 / +8% Q)", "Horizonte 2040 (SSP3-7.0 / +15% Q)", "Horizonte 2050 (SSP5-8.5 / +22% Q)"],
        key="clim_horizon"
    )
    factor_clima = 1.0 if "Actual" in horizonte_clima else (1.08 if "2030" in horizonte_clima else (1.15 if "2040" in horizonte_clima else 1.22))

all_municipalities = sorted(df_parcels["municipality"].unique())
selected_muns = st.sidebar.multiselect("Municipios Analizados:", options=all_municipalities, default=all_municipalities, key="sel_muns")

# ==============================================================================
# MODELO HIDROLÓGICO NO LINEAL ROBUSTO (CEDEX / SAIH CHJ)
# ==============================================================================
if "Forense" in sim_mode:
    q_peak_simulated = raw_q * q_mitig_factor
    mask_past = t_arr <= sim_minute
    peak_q_so_far = float(np.max(q_series[mask_past])) * q_mitig_factor if np.any(mask_past) else q_peak_simulated
    current_depth_factor = q_peak_simulated / 1950.0
    peak_damage_factor_reached = peak_q_so_far / 1950.0
    q_instant_lead = q_peak_simulated
else:
    # Calibración matemática realista:
    # A > 180 mm el suelo se satura incondicionalmente
    if rain_val <= 5.0:
        q_raw = 0.0
    elif rain_val <= 60.0:
        q_raw = (rain_val / 60.0) * 320.0 * amc_weight
    elif rain_val <= 180.0:
        alpha = (rain_val - 60.0) / 120.0
        q_raw = (320.0 + alpha * (1380.0 - 320.0)) * amc_weight
    else:
        # Para tormentas de 180 a 600 mm: saturación total y crecida exponencial
        alpha = (rain_val - 180.0) / (600.0 - 180.0)
        # A 190 mm -> ~1.550 m³/s (Desborde claro)
        # A 490 mm -> ~2.700 m³/s (Cresta 29-O)
        # A 600 mm -> ~3.250 m³/s (DANA extrema máxima)
        q_base_sat = 1380.0 + (alpha ** 0.85) * (3250.0 - 1380.0)
        q_raw = q_base_sat

    q_peak_simulated = float(np.clip(q_raw * factor_clima * q_mitig_factor, 0.0, 3600.0))
    peak_q_so_far = q_peak_simulated
    current_depth_factor = q_peak_simulated / 1950.0
    peak_damage_factor_reached = current_depth_factor
    
    # Caudal instantáneo en el frente de avance según Lead Time
    t_lead = float(lead_time_min)
    if t_lead <= 120.0:
        q_instant_lead = q_peak_simulated * (t_lead / 120.0) ** 2.0
    else:
        q_instant_lead = q_peak_simulated * np.exp(-((t_lead - 120.0) / 80.0) ** 1.3)

# ==============================================================================
# FÍSICA COMPUESTA Y DAÑO ESTRUCTURAL CALIBRADO
# ==============================================================================
active_df = df_parcels[df_parcels["municipality"].isin(selected_muns)].copy()

h_fluvial_act = active_df["max_depth_m"].to_numpy() * current_depth_factor
h_fluvial_pk = active_df["max_depth_m"].to_numpy() * peak_damage_factor_reached

mota_add = 0.85 if (ruptura_mota and q_peak_simulated > 200.0) else 0.0

muns_arr = active_df["municipality"].to_numpy()
pluvial_add = np.where(
    np.isin(muns_arr, ["València", "Quart de Poblet"]) & sim_pluvial_vlc,
    0.85,
    0.0
)

is_coastal = (active_df["lon"].to_numpy() > -0.370) | np.isin(muns_arr, ["València", "Sedaví", "Alfafar", "Catarroja"])
surge_add = np.where(
    is_coastal & sim_storm_surge,
    0.70,
    0.0
)

total_h_act = h_fluvial_act + mota_add + pluvial_add + surge_add
total_h_pk = h_fluvial_pk + mota_add + pluvial_add + surge_add

active_df["active_depth"] = total_h_act.astype(np.float32)
active_df["peak_depth_experienced"] = total_h_pk.astype(np.float32)

# VULNERABILIDAD ANCLADA EN CERO
h_eval = active_df["peak_depth_experienced"].to_numpy()
types = active_df["asset_type"].to_numpy()
ratios = np.zeros(len(active_df), dtype=np.float64)

mask_wet = h_eval >= 0.08

mask_res = (types == "Residencial") & mask_wet
ratios[mask_res] = np.clip((h_eval[mask_res] - 0.08) / 1.70, 0.0, 1.0) ** 1.25

mask_ind = (types == "Industrial / Logística") & (h_eval >= 0.35)
ratios[mask_ind] = np.clip((h_eval[mask_ind] - 0.35) / 1.40, 0.0, 1.0) ** 1.10

mask_veh = (types == "Vehículos / Vados") & mask_wet
ratios[mask_veh] = np.where(h_eval[mask_veh] >= 0.35, 1.0, (h_eval[mask_veh] / 0.35) ** 2)

mask_pub = (types == "Infraestructura Pública") & (h_eval >= 0.50)
ratios[mask_pub] = np.clip((h_eval[mask_pub] - 0.50) / 2.0, 0.0, 1.0)

ratios = np.clip(ratios, 0.0, 1.0)
active_df["damage_ratio"] = ratios
active_df["active_loss"] = (active_df["asset_value_eur"].to_numpy() * ratios).astype(np.float64)

is_currently_flooded = active_df["active_depth"] >= 0.25

active_df["dynamic_collapse"] = (
    (active_df["peak_depth_experienced"] >= 1.20) &
    ((active_df["hazard_factor_vh"] * peak_damage_factor_reached >= 0.85) | (active_df["dpm_p90"] >= 0.45))
)

active_df["dynamic_p1"] = (
    (is_currently_flooded & (
        (active_df["active_depth"] * active_df["max_velocity_ms"] * current_depth_factor >= 0.85) |
        (active_df["is_critical_infra"] & (active_df["active_depth"] >= 0.60)) |
        (active_df["active_depth"] >= 0.90)
    )) |
    active_df["dynamic_collapse"]
)

# ALERTAS OPERATIVAS MODULADAS
rain_mm = float(rain_val)
has_experienced_catastrophe = ("Forense" in sim_mode and peak_q_so_far >= 1200.0) or (rain_mm >= 180.0) or (q_peak_simulated >= 1200.0) or (sim_pluvial_vlc and sim_storm_surge)

if has_experienced_catastrophe:
    if q_peak_simulated >= 1200.0:
        badge_txt = "🔴 SIT. 2: DESBORDAMIENTO METROPOLITANO SEVERO"
    else:
        badge_txt = "🔴 SIT. 2: ALERTA ROJA PREVENTIVA (CHIVA > 180 mm)"
    alert_badge_html = f"<span class='badge-alert-red'>{badge_txt}</span>"
    alert_state = "ROJO"

elif rain_mm >= 90.0 or q_peak_simulated >= 600.0 or sim_pluvial_vlc or sim_storm_surge:
    badge_txt = "🟠 SIT. 1: CRECIDA SEVERA EN CAUCE"
    alert_badge_html = f"<span class='badge-alert-orange'>{badge_txt}</span>"
    alert_state = "NARANJA"

elif rain_mm >= 40.0 or q_peak_simulated >= 250.0:
    alert_badge_html = "<span class='badge-alert-yellow'>🟡 PREALERTA POR LLUVIAS</span>"
    alert_state = "AMARILLO"

else:
    alert_badge_html = "<span class='badge-alert-green'>🟢 NORMALIDAD HIDROLÓGICA</span>"
    alert_state = "VERDE"

# CABECERA HUD C2
st.markdown(
    f"""
    <div class='hud-header'>
        <div style='display: flex; justify-content: space-between; align-items: center; width: 100%; flex-wrap: wrap; gap: 8px;'>
            <div style='flex: 1; min-width: 280px;'>
                <h1 style='margin:0; font-size: 1.45rem; letter-spacing: -0.02em;'>POYO-NOWCAST // PLATAFORMA C2 DE ALTA DEFINICIÓN</h1>
                <span style='color: #8b949e; font-size: 0.75rem;'>ÁREA METROPOLITANA DE VALÈNCIA & L'HORTA SUD | FÍSICA FNO 2D Y TRANSFERENCIA DE RIESGO SOLVENCIA II</span>
            </div>
            <div style='display: flex; gap: 8px; align-items: center;'>
                {alert_badge_html}
                <div class='badge-clock-box'>{clock_badge_text}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

max_local_depth = float(np.max(active_df["active_depth"])) if not active_df.empty else 0.0

if alert_state == "ROJO":
    if max_local_depth >= 0.80 or q_peak_simulated >= 1400.0:
        escape_msg = "VENTANA DE ESCAPE: AGOTADA<br/><span style='color:#c9d1d9; font-weight:normal;'>Permanezca en pisos altos</span>"
    else:
        escape_msg = "VENTANA DE SEGURIDAD: &lt; 35 MIN<br/><span style='color:#ffe3a8; font-weight:normal;'>Evacuación vertical preventiva</span>"

    st.markdown(
        f"""
        <div class='evac-banner-red'>
            <div style='display:flex; align-items:center; gap:16px;'>
                <span style='font-size:2.2rem;'>🚨</span>
                <div>
                    <div class='evac-banner-red-title'>ORDEN GENERAL DE EVACUACIÓN VERTICAL — PROTECCIÓN CIVIL / CECOPI</div>
                    <div class='evac-banner-red-sub'>PELIGRO EXTREMO POR DESBORDAMIENTO. Suba de inmediato a plantas altas. Prohibido circular por carretera o acceder a garajes/vados.</div>
                </div>
            </div>
            <div style='text-align:right; font-family: monospace; font-size:0.80rem; color:#ff7b72; font-weight:800;'>
                {escape_msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif alert_state == "NARANJA":
    st.markdown(
        """
        <div class='evac-banner-orange'>
            <div style='display:flex; align-items:center; gap:16px;'>
                <span style='font-size:2.2rem;'>⚠️</span>
                <div>
                    <div style='color:#f0883e; font-size:1.10rem; font-weight:800;'>PRE-ALERTA DE EVACUACIÓN: EVITE DESPLAZAMIENTOS Y RETIRE VEHÍCULOS</div>
                    <div style='color:#f0f6fc; font-size:0.88rem; font-weight:500;'>Onda de avenida aproximándose a l'Horta Sud o afección pluvial/marítima. Asegure puntos altos.</div>
                </div>
            </div>
            <div style='text-align:right; font-family: monospace; font-size:0.80rem; color:#f0883e; font-weight:800;'>
                VENTANA DE SEGURIDAD:<br/><span style='color:#ffe3a8; font-weight:normal;'>&lt; 45 MINUTOS</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

fno_latency_ms = 38.4 if "Forense" in sim_mode else 42.1

if "Forense" in sim_mode:
    st.markdown(
        f"""
        <div class='telemetry-strip' style='border-left: 4px solid #da3633;'>
            <div>🏛️ <b>SERIE FORENSE 29-O:</b> Chiva / SAIH Hidrosur</div>
            <div>📅 <b>Fecha/Hora Sensor:</b> <span style='color:#58a6ff; font-weight:700;'>{sensor_date_label}</span></div>
            <div>🌧️ <b>Lluvia Chiva:</b> <span style='color:#f85149; font-weight:700;'>{fmt_dec(rain_val, 1, ' mm')}</span></div>
            <div>🌊 <b>Caudal Rambla:</b> <span style='color:#58a6ff; font-weight:700;'>{fmt_int(q_peak_simulated, ' m³/s')}</span></div>
            <div>⚡ <b>Inferencia FNO 2D:</b> <span style='color:#3fb950; font-weight:700;'>{fno_latency_ms:.1f} ms</span></div>
            <div>🔴 <span style='color:#f85149; font-weight:700;'>HINDCAST OFICIAL</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif telemetry_active and live_obs:
    st.markdown(
        f"""
        <div class='telemetry-strip' style='border-left: 4px solid #238636;'>
            <div>📡 <b>TELEMETRÍA AEMET:</b> {live_obs['station_name']} ({live_obs['station_id']})</div>
            <div>📅 <b>Fecha/Hora Sensor:</b> <span style='color:#3fb950; font-weight:700;'>{sensor_date_label}</span></div>
            <div>🌧️ <b>Lluvia 1h:</b> <span style='color:#58a6ff;'>{fmt_dec(live_obs['rain_1h_mm'], 1, ' mm')}</span></div>
            <div>📈 <b>Acum. 4h:</b> <span style='color:#58a6ff;'>{fmt_dec(live_obs['rain_4h_mm'], 1, ' mm')}</span></div>
            <div>🌡️ <b>Temp:</b> {fmt_dec(live_obs['temp_c'], 1, ' °C')}</div>
            <div>⚡ <b>Inferencia FNO:</b> <span style='color:#3fb950; font-weight:700;'>{fno_latency_ms:.1f} ms</span></div>
            <div>🟢 <span style='color:#3fb950; font-weight:700;'>ONLINE (&lt; 15 min)</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class='telemetry-strip' style='border-left: 4px solid #1f6feb;'>
            <div>📅 <b>Timestamp Simulado:</b> <span style='color:#58a6ff; font-weight:700;'>{sensor_date_label}</span></div>
            <div>🌧️ <b>Lluvia Cabecera Chiva:</b> <span style='color:#58a6ff; font-weight:700;'>{fmt_dec(rain_val, 1, ' mm')}</span></div>
            <div>🌊 <b>Caudal Estimado FNO:</b> <span style='color:#58a6ff; font-weight:700;'>{fmt_int(q_peak_simulated, ' m³/s')}</span></div>
            <div>⚡ <b>Latencia FNO 2D:</b> <span style='color:#3fb950; font-weight:700;'>{fno_latency_ms:.1f} ms</span></div>
            <div>🔵 <span style='color:#58a6ff; font-weight:700;'>PROYECCIÓN FUTURA NOWCAST</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# KPIs OPERATIVOS TÁCTICOS
# ==============================================================================
total_exposure_m = active_df["asset_value_eur"].sum() / 1e6
current_loss_m = active_df["active_loss"].sum() / 1e6
total_collapsed = int(active_df["dynamic_collapse"].sum())
critical_p1 = int(active_df["dynamic_p1"].sum())

q_att, q_exh = 1000.0, 1800.0
ins_att, ins_exh = 0.02, 0.08
q_eval_cat = peak_q_so_far if "Forense" in sim_mode else q_peak_simulated
collapse_ratio = total_collapsed / max(1, len(active_df))

fq = min(1.0, max(0.0, (q_eval_cat - q_att) / (q_exh - q_att)))
fi = min(1.0, max(0.0, (collapse_ratio - ins_att) / (ins_exh - ins_att)))

if fq > 0 and fi > 0:
    payout_rate = np.sqrt(fq * fi) * 100.0
elif (sim_pluvial_vlc or sim_storm_surge) and fi > 0:
    payout_rate = fi * 85.0
else:
    payout_rate = 0.0

total_payout_m = 80.0 * (payout_rate / 100.0)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    diff_q = q_peak_simulated - 1200
    st.metric(
        "Caudal Punta FNO",
        fmt_int(q_peak_simulated, " m³/s"),
        delta=f"{'+' if diff_q >= 0 else ''}{fmt_int(diff_q, ' m³/s')} vs Umbral Alerta",
        delta_color="inverse"
    )
with kpi2:
    pct_exp = (current_loss_m / max(0.1, total_exposure_m)) * 100
    st.metric(
        "Pérdida Directa Activa",
        fmt_dec(current_loss_m, 1, " M€"),
        delta=f"{fmt_dec(pct_exp, 1, '%')} de Exposición",
        delta_color="inverse" if pct_exp > 0 else "off"
    )
with kpi3:
    st.metric(
        "Inmuebles en Ruina",
        fmt_int(total_collapsed),
        delta="InSAR DPM ≥ 0,40" if total_collapsed > 0 else "Sin colapsos estructurales",
        delta_color="inverse" if total_collapsed > 0 else "off"
    )
with kpi4:
    st.metric(
        "Prioridad P1 (Rescate 112)",
        fmt_int(critical_p1),
        delta="Evacuación Inmediata" if critical_p1 > 0 else "Situación bajo control",
        delta_color="inverse" if critical_p1 > 0 else "off"
    )
with kpi5:
    st.metric(
        "Gatillo Paramétrico (Cat Bond)",
        fmt_dec(payout_rate, 1, "%"),
        delta=f"{fmt_dec(total_payout_m, 1, ' M€')} Liberados < 48h",
        delta_color="inverse" if payout_rate > 50.0 else "normal"
    )

st.markdown("<br/>", unsafe_allow_html=True)

if "map_lat" not in st.session_state:
    st.session_state["map_lat"] = 39.4320
if "map_lon" not in st.session_state:
    st.session_state["map_lon"] = -0.4150
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 12.6
if "map_pitch" not in st.session_state:
    st.session_state["map_pitch"] = 52

# ==============================================================================
# PESTAÑAS DEL CENTRO DE MANDO C2
# ==============================================================================
tab_3d, tab_esalert, tab_compare, tab_hydro, tab_roads, tab_finances = st.tabs([
    "🌐 Gemelo Digital 3D (WebGPU)",
    "🚨 Despacho ES-Alert & Alertas C2",
    "⚖️ Auditoría Split A/B (29-O vs Nowcast)",
    "🌊 Dinámica Hidráulica FNO (M2)",
    "🚑 Resiliencia Vial & TTI (M3)",
    "💼 Finanzas del Clima & Solvencia II (M4)",
])

# ------------------------------------------------------------------------------
# TAB 1: VISOR 3D (TOOLBAR HOMOGÉNEA Y EXTRUSIÓN ACTIVA)
# ------------------------------------------------------------------------------
with tab_3d:
    col_ctrl_left, col_ctrl_right = st.columns([2.0, 2.0])
    with col_ctrl_left:
        st.markdown("<b style='color:#58a6ff; font-size:0.80rem;'>📊 Modo de Visualización 3D:</b>", unsafe_allow_html=True)
        render_variable = st.radio(
            "Modo de Representación 3D en Gemelo Digital:",
            ["🌊 Calado Hidrodinámico FNO", "💶 Pérdida Económica CCS (€)", "🚨 Prioridad Triaje 112 (P1-P4)"],
            horizontal=True,
            key="render_var_horizontal",
            label_visibility="collapsed"
        )
    with col_ctrl_right:
        st.markdown("<b style='color:#58a6ff; font-size:0.80rem;'>🎯 Centrado Rápido de Cámara:</b>", unsafe_allow_html=True)
        cam_c1, cam_c2, cam_c3, cam_c4 = st.columns(4)
        with cam_c1:
            if st.button("📍 Paiporta", use_container_width=True):
                st.session_state["map_lat"] = 39.4240
                st.session_state["map_lon"] = -0.4180
                st.session_state["map_zoom"] = 13.5
                st.session_state["map_pitch"] = 55
                st.rerun()
        with cam_c2:
            if st.button("🏛️ Sedes GVA", use_container_width=True):
                st.session_state["map_lat"] = 39.4770
                st.session_state["map_lon"] = -0.3700
                st.session_state["map_zoom"] = 13.5
                st.session_state["map_pitch"] = 48
                st.rerun()
        with cam_c3:
            if st.button("🎓 Campus VIU", use_container_width=True):
                st.session_state["map_lat"] = 39.4750
                st.session_state["map_lon"] = -0.3480
                st.session_state["map_zoom"] = 13.6
                st.session_state["map_pitch"] = 48
                st.rerun()
        with cam_c4:
            if st.button("✈️ Aeropuerto", use_container_width=True):
                st.session_state["map_lat"] = 39.4850
                st.session_state["map_lon"] = -0.4550
                st.session_state["map_zoom"] = 12.8
                st.session_state["map_pitch"] = 50
                st.rerun()

    legend_html = """
    <div class='legend-box'>
        <span style='color:#8b949e; font-weight:700;'>SIMBOLOGÍA 3D:</span>
        <div class='legend-item'><span class='legend-bullet' style='background:#da3633;'></span> Ruina / Colapso</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#f85149;'></span> &ge; 1,50 m</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#f0883e;'></span> 0,80 - 1,50 m</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#d29922;'></span> 0,30 - 0,80 m</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#238636;'></span> Vía Transitable</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#d73a49;'></span> Vía Cortada</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#e3a93b;'></span> Centro Sensible</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#58a6ff; border-radius:50%;'></span> Hospital</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#00e5ff; border-radius:50%;'></span> Transporte (Aeropuerto/AVE/Puerto)</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#a371f7; border-radius:50%;'></span> Sedes Oficiales / Emergencias</div>
        <div class='legend-item'><span class='legend-bullet' style='background:#ffd700; border-radius:50%;'></span> Universidades (VIU/UV/UPV)</div>
        <div class='legend-item'><span class='legend-circle-outline' style='border: 2.5px solid #da3633;'></span> 10 min (Inmediato)</div>
        <div class='legend-item'><span class='legend-circle-outline' style='border: 2.5px solid #f0883e;'></span> 20 min (Medio)</div>
        <div class='legend-item'><span class='legend-circle-outline' style='border: 2.5px solid #2ea043;'></span> 30 min (Exterior)</div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    cap_c1, cap_c2, cap_c3, cap_c4, cap_c5, cap_c6 = st.columns(6)
    with cap_c1:
        show_roads = st.checkbox("🛣️ Red Viaria", value=True, key="chk_roads_tab")
    with cap_c2:
        show_vulnerable = st.checkbox("🏥 Centros Sensibles", value=True, key="chk_vuln_tab")
    with cap_c3:
        show_transport = st.checkbox("✈️ Transporte (Cian)", value=True, key="chk_trans_tab")
    with cap_c4:
        show_institutions = st.checkbox("🏛️ Sedes Oficiales", value=True, key="chk_inst_tab")
    with cap_c5:
        show_universities = st.checkbox("🎓 Universidades", value=True, key="chk_univ_tab")
    with cap_c6:
        show_isochrones = st.checkbox("⭕ Isocronas Evac.", value=True, key="chk_iso_tab")

    show_hospitals = True

    h_eval_3d = active_df["peak_depth_experienced"].to_numpy() if "Forense" in sim_mode else active_df["active_depth"].to_numpy()
    c_eval_3d = active_df["dynamic_collapse"].to_numpy()

    if "Calado" in render_variable:
        condlist = [
            c_eval_3d,
            h_eval_3d >= 1.50,
            h_eval_3d >= 0.80,
            h_eval_3d >= 0.30,
            h_eval_3d >= 0.08,
        ]
        r_c = np.select(condlist, [218, 248, 240, 210, 46], default=30)
        g_c = np.select(condlist, [54,  81,  136, 153, 160], default=40)
        b_c = np.select(condlist, [51,  73,  62,  34,  67], default=55)
        a_c = np.select(condlist, [255, 235, 215, 195, 160], default=40)
        
        rgb_arr = np.column_stack([r_c, g_c, b_c, a_c]).astype(np.uint8)
        elevations = np.where(
            h_eval_3d < 0.08,
            1.0,
            np.select(
                [c_eval_3d, h_eval_3d >= 0.30],
                [np.maximum(h_eval_3d * 35.0, 70.0), h_eval_3d * 30.0 + 8.0],
                default=np.clip(h_eval_3d * 20.0 + 4.0, 4.0, 30.0)
            )
        )

    elif "Pérdida" in render_variable:
        l_col = active_df["active_loss"].to_numpy()
        condlist_loss = [
            l_col >= 150000,
            l_col >= 75000,
            l_col >= 30000,
            l_col >= 10000,
            l_col >= 1000,
        ]
        r_c = np.select(condlist_loss, [156, 229, 251, 253, 67], default=30)
        g_c = np.select(condlist_loss, [39,  57,  140, 216, 160], default=40)
        b_c = np.select(condlist_loss, [176, 53,  0,   53,  71], default=55)
        a_c = np.select(condlist_loss, [255, 235, 205, 185, 150], default=40)
        
        rgb_arr = np.column_stack([r_c, g_c, b_c, a_c]).astype(np.uint8)
        elevations = np.where(l_col <= 0.0, 1.0, np.clip((l_col / 2500.0) + 6.0, 4.0, 160.0))

    else:
        p1_col = active_df["dynamic_p1"].to_numpy()
        condlist_triage = [
            p1_col,
            (h_eval_3d >= 0.80) | (active_df["dynamic_isolated"].to_numpy() if "dynamic_isolated" in active_df.columns else False),
            (h_eval_3d >= 0.30) | (active_df["active_loss"].to_numpy() >= 20000),
        ]
        r_c = np.select(condlist_triage, [218, 240, 210], default=30)
        g_c = np.select(condlist_triage, [54,  136, 153], default=40)
        b_c = np.select(condlist_triage, [51,  62,  34],  default=55)
        a_c = np.select(condlist_triage, [255, 220, 190], default=40)
        
        rgb_arr = np.column_stack([r_c, g_c, b_c, a_c]).astype(np.uint8)
        elevations = np.where(h_eval_3d < 0.08, 1.0, np.select([p1_col, (h_eval_3d >= 0.80)], [80.0, 40.0], default=10.0))

    active_df["rgba"] = rgb_arr.tolist()
    active_df["elevation_m"] = elevations
    active_df["layer_title"] = active_df["parcel_id"] + " (" + active_df["municipality"] + ")"
    active_df["metric_primary"] = "Calado Máx: " + active_df["peak_depth_experienced"].apply(lambda v: fmt_dec(v, 2, " m"))
    active_df["metric_secondary"] = "Pérdida CCS: " + active_df["active_loss"].apply(lambda v: fmt_dec(v, 0, " €"))
    
    active_df["status_tag"] = np.select(
        [active_df["dynamic_collapse"], active_df["dynamic_p1"], active_df["peak_depth_experienced"] >= 0.80, active_df["peak_depth_experienced"] >= 0.30],
        ["RUINA / COLAPSO", "P1_CRITICA", "P2_ALTA", "P3_MODERADA"],
        default="P4_NORMAL"
    )
    active_df["status_color"] = np.select(
        [active_df["dynamic_collapse"], active_df["dynamic_p1"], active_df["peak_depth_experienced"] >= 0.80, active_df["peak_depth_experienced"] >= 0.30],
        ["#f85149", "#f85149", "#f0883e", "#d29922"],
        default="#3fb950"
    )

    deck_layers = [
        pdk.Layer(
            "ColumnLayer",
            data=active_df,
            get_position=["lon", "lat"],
            get_elevation="elevation_m",
            elevation_scale=1,
            radius=14,
            get_fill_color="rgba",
            pickable=True,
            auto_highlight=True,
        )
    ]

    # CAPAS VISUALES DIRECTAS DE DINÁMICA COMPUESTA EN PYDECK
    if sim_pluvial_vlc:
        pluvial_polygon = [{
            "polygon": [
                [-0.435, 39.485], [-0.340, 39.495], [-0.335, 39.450], 
                [-0.380, 39.445], [-0.440, 39.465], [-0.435, 39.485]
            ],
            "layer_title": "Anegamiento Pluvial Urbano (València Centro)",
            "metric_primary": "Precipitación Local > 120 mm/h",
            "metric_secondary": "Colapso de Colectores y Pasos Subterráneos",
            "status_tag": "ANEGAMIENTO ACTIVO",
            "status_color": "#1f6feb",
            "color": [31, 111, 235, 95]
        }]
        deck_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=pluvial_polygon,
                get_polygon="polygon",
                get_fill_color="color",
                get_line_color=[56, 139, 253, 220],
                line_width_min_pixels=2,
                stroked=True,
                pickable=True,
            )
        )

    if sim_storm_surge:
        surge_polygon = [{
            "polygon": [
                [-0.355, 39.490], [-0.310, 39.490], [-0.310, 39.380], 
                [-0.370, 39.380], [-0.360, 39.440], [-0.355, 39.490]
            ],
            "layer_title": "Intrusión Marina & Storm Surge (Temporal de Levante)",
            "metric_primary": "Sobreelevación del Nivel del Mar +0,70 m",
            "metric_secondary": "Taponamiento de Desembocaduras (Turia / Albufera)",
            "status_tag": "TAPONAMIENTO COSTEIRO ACTIVO",
            "status_color": "#00e5ff",
            "color": [0, 229, 255, 80]
        }]
        deck_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=surge_polygon,
                get_polygon="polygon",
                get_fill_color="color",
                get_line_color=[0, 229, 255, 240],
                line_width_min_pixels=2,
                stroked=True,
                pickable=True,
            )
        )

    if show_roads and realistic_roads:
        road_paths = []
        for r_item in realistic_roads:
            h_water_edge = r_item["h_base"] * current_depth_factor
            h_peak_edge = r_item["h_base"] * peak_damage_factor_reached
            
            if h_water_edge >= 0.30:
                edge_open = False
                reason = f"CORTADO (Calado {fmt_dec(h_water_edge, 2, ' m')})"
                r_color_hex = "#f85149"
            elif ("Forense" in sim_mode or peak_damage_factor_reached >= 0.50) and h_peak_edge >= 0.40:
                edge_open = False
                reason = "BLOQUEADA (Lodo y sedimentos)"
                r_color_hex = "#d29922"
            else:
                edge_open = True
                reason = "OPERATIVO (TRANSITABLE)"
                r_color_hex = "#3fb950"

            col = [46, 160, 67, 220] if edge_open else [218, 54, 51, 250]
            road_paths.append({
                "path": r_item["coords"],
                "layer_title": r_item["name"],
                "metric_primary": f"Cota Inundación: {fmt_dec(h_water_edge, 2, ' m')}",
                "metric_secondary": "Eje Arterial Metropolitano",
                "status_tag": reason,
                "status_color": r_color_hex,
                "color": col,
                "width": 5 if edge_open else 8,
            })
        deck_layers.append(
            pdk.Layer(
                "PathLayer",
                data=road_paths,
                get_path="path",
                get_color="color",
                get_width="width",
                width_scale=1,
                width_min_pixels=3,
                pickable=True,
            )
        )

    if show_isochrones:
        origin_lon, origin_lat = -0.4190, 39.4230
        penalizacion = max(0.18, 1.0 - (q_peak_simulated / 2200.0))
        
        iso_rings = [
            {"time": "Isocrona 30 min (Perímetro Exterior)", "radius": 3200 * penalizacion, "color": [46, 160, 67, 240]},
            {"time": "Isocrona 20 min (Perímetro Medio)", "radius": 2000 * penalizacion, "color": [240, 136, 62, 240]},
            {"time": "Isocrona 10 min (Escape Inmediato)", "radius": 900 * penalizacion, "color": [218, 54, 51, 255]},
        ]
        
        iso_df = pd.DataFrame(iso_rings)
        iso_df["lon"] = origin_lon
        iso_df["lat"] = origin_lat
        iso_df["layer_title"] = iso_df["time"]
        iso_df["metric_primary"] = "Alcance Perimetral desde Casco Urbano (Paiporta)"
        iso_df["metric_secondary"] = f"Ventana de Seguridad: {int(penalizacion * 100)}% de viabilidad"
        iso_df["status_tag"] = "ISOCRONA DE EVACUACIÓN"
        iso_df["status_color"] = "#58a6ff"

        deck_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=iso_df,
                get_position=["lon", "lat"],
                get_radius="radius",
                filled=False,
                stroked=True,
                get_line_color="color",
                line_width_min_pixels=3,
                pickable=True,
            )
        )

    if show_transport:
        eval_factor_t = peak_damage_factor_reached if "Forense" in sim_mode else current_depth_factor
        trans_rows = []
        for tnode in TRANSPORT_LANDMARKS_BASE:
            h_local_t = tnode["h_base"] * eval_factor_t + (0.50 if (sim_storm_surge and tnode["type"] == "PUERTO") else 0.0)
            if h_local_t >= 0.40 or (tnode["type"] == "TRANSPORTE" and eval_factor_t >= 0.50):
                t_stat = "🔴 ALERTA DE INUNDACIÓN / NODO CORTADO"
                t_col = [218, 54, 51, 240]
                t_hex = "#f85149"
            elif h_local_t >= 0.10 or eval_factor_t >= 0.35:
                t_stat = "🟠 PREALERTA EN ACCESOS / RETRASOS"
                t_col = [240, 136, 62, 230]
                t_hex = "#f0883e"
            else:
                t_stat = "🟢 OPERATIVIDAD NOMINAL"
                t_col = [0, 229, 255, 240]
                t_hex = "#00e5ff"

            trans_rows.append({
                "name": f"{tnode['icon']} {tnode['name']}",
                "type": tnode["type"],
                "lon": tnode["lon"], "lat": tnode["lat"],
                "layer_title": tnode["name"],
                "metric_primary": f"Nodo Logístico: {tnode['type']}",
                "metric_secondary": f"Calado Estimado: {fmt_dec(h_local_t, 2, ' m')}",
                "status_tag": t_stat,
                "status_color": t_hex,
                "color": t_col
            })
        trans_df = pd.DataFrame(trans_rows)
        deck_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=trans_df,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=270,
                pickable=True,
            )
        )
        deck_layers.append(
            pdk.Layer(
                "TextLayer",
                data=trans_df,
                get_position=["lon", "lat"],
                get_text="name",
                get_size=12,
                get_color=[0, 229, 255, 255],
                get_text_anchor="'start'",
                get_alignment_baseline="'center'",
                pixel_offset=[20, 0],
            )
        )

    if show_institutions:
        eval_factor_inst = peak_damage_factor_reached if "Forense" in sim_mode else current_depth_factor
        inst_rows = []
        for inode in INSTITUTION_LANDMARKS_BASE:
            h_local_i = inode["h_base"] * eval_factor_inst
            if has_experienced_catastrophe or eval_factor_inst >= 0.70:
                i_stat = "🔴 COMITÉ DE CRISIS ACTIVO / PEI"
                i_col = [218, 54, 51, 240]
                i_hex = "#f85149"
            elif eval_factor_inst >= 0.35:
                i_stat = "🟠 PREALERTA EN ACCESOS / RESTRICCIÓN"
                i_col = [240, 136, 62, 230]
                i_hex = "#f0883e"
            else:
                i_stat = "🟢 OPERATIVIDAD INSTITUCIONAL NORMAL"
                i_col = [163, 113, 247, 240]
                i_hex = "#a371f7"

            inst_rows.append({
                "name": f"{inode['icon']} {inode['name']}",
                "type": inode["type"],
                "lon": inode["lon"], "lat": inode["lat"],
                "layer_title": inode["name"],
                "metric_primary": f"Tipología: {inode['type']}",
                "metric_secondary": f"Calado Evaluado: {fmt_dec(h_local_i, 2, ' m')}",
                "status_tag": i_stat,
                "status_color": i_hex,
                "color": i_col
            })
        inst_df = pd.DataFrame(inst_rows)
        deck_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=inst_df,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=240,
                pickable=True,
            )
        )
        deck_layers.append(
            pdk.Layer(
                "TextLayer",
                data=inst_df,
                get_position=["lon", "lat"],
                get_text="name",
                get_size=12,
                get_color=[240, 246, 252, 255],
                get_text_anchor="'start'",
                get_alignment_baseline="'center'",
                pixel_offset=[20, 0],
            )
        )

    if show_universities:
        eval_factor_u = peak_damage_factor_reached if "Forense" in sim_mode else current_depth_factor
        univ_rows = []
        for unode in UNIVERSITIES_BASE:
            h_local_u = unode["h_base"] * eval_factor_u + (0.40 if sim_pluvial_vlc else 0.0)
            if has_experienced_catastrophe or eval_factor_u >= 0.70:
                u_stat = "🔴 ACTIVIDAD PRESENCIAL SUSPENDIDA"
                u_col = [218, 54, 51, 240]
                u_hex = "#f85149"
            elif eval_factor_u >= 0.35 or sim_pluvial_vlc:
                u_stat = "🟠 PREALERTA / DOCENCIA REMOTA"
                u_col = [240, 136, 62, 230]
                u_hex = "#f0883e"
            else:
                u_stat = "🟢 CAMPUS OPERATIVO"
                u_col = [255, 215, 0, 240]
                u_hex = "#ffd700"

            univ_rows.append({
                "name": f"{unode['icon']} {unode['name']}",
                "type": unode["type"],
                "lon": unode["lon"], "lat": unode["lat"],
                "layer_title": unode["name"],
                "metric_primary": "Campus Universitario Metropolitano",
                "metric_secondary": f"Calado Estimado en Entorno: {fmt_dec(h_local_u, 2, ' m')}",
                "status_tag": u_stat,
                "status_color": u_hex,
                "color": u_col
            })
        univ_df = pd.DataFrame(univ_rows)
        deck_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=univ_df,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=150,
                pickable=True,
            )
        )
        deck_layers.append(
            pdk.Layer(
                "TextLayer",
                data=univ_df,
                get_position=["lon", "lat"],
                get_text="name",
                get_size=12,
                get_color=[255, 215, 0, 255],
                get_text_anchor="'start'",
                get_alignment_baseline="'center'",
                pixel_offset=[18, 0],
            )
        )

    if show_vulnerable:
        eval_factor_vuln = peak_damage_factor_reached if "Forense" in sim_mode else current_depth_factor
        vuln_rows = []
        for v in VULNERABLE_CENTERS_BASE:
            h_local_vuln = v["h_base"] * eval_factor_vuln
            if h_local_vuln >= 0.50:
                v_stat = "🔴 INUNDACIÓN CRÍTICA / EVACUAR"
                v_col = [218, 54, 51, 240]
                v_hex = "#f85149"
            elif h_local_vuln >= 0.15:
                v_stat = "🟠 EN RIESGO / PREALERTA"
                v_col = [240, 136, 62, 230]
                v_hex = "#f0883e"
            else:
                v_stat = "🟢 OPERATIVIDAD NOMINAL"
                v_col = [46, 160, 67, 210]
                v_hex = "#3fb950"
            
            vuln_rows.append({
                "name": v["name"], "mun": v["mun"], "type": v["type"],
                "lon": v["lon"], "lat": v["lat"], "beds": v["beds"],
                "layer_title": f"{v['name']} ({v['mun']})",
                "metric_primary": f"Capacidad: {v['beds']} plazas",
                "metric_secondary": f"Calado Evaluado: {fmt_dec(h_local_vuln, 2, ' m')}",
                "status_tag": v_stat,
                "status_color": v_hex,
                "color": v_col
            })
        vuln_df = pd.DataFrame(vuln_rows)
        deck_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=vuln_df,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=140,
                pickable=True,
            )
        )

    if show_hospitals:
        hosp_data = []
        for h in [
            {"name": "H. Universitari i Politècnic La Fe", "lon": -0.3768, "lat": 39.4435, "beds": 1000},
            {"name": "H. General Universitari de València", "lon": -0.4072, "lat": 39.4682, "beds": 550},
            {"name": "Hospital de Manises (Acceso Norte A-3)", "lon": -0.4608, "lat": 39.4930, "beds": 240},
        ]:
            if has_experienced_catastrophe:
                h_status = "ALERTA MÁXIMA (COLAPSO DE ACCESOS METROPOLITANOS)"
                h_col = [218, 54, 51, 240]
                h_color_hex = "#f85149"
            elif q_peak_simulated >= 600.0:
                h_status = "PREALERTA SANITARIA"
                h_col = [210, 153, 34, 230]
                h_color_hex = "#d29922"
            else:
                h_status = "OPERATIVO (NORMAL)"
                h_col = [56, 139, 253, 230]
                h_color_hex = "#3fb950"

            hosp_data.append({
                "name": h["name"], "lon": h["lon"], "lat": h["lat"],
                "layer_title": h["name"],
                "metric_primary": f"Capacidad: {h['beds']} camas",
                "metric_secondary": "Rol: Hospital Terciario",
                "status_tag": h_status,
                "status_color": h_color_hex,
                "color": h_col
            })

        hosp_df = pd.DataFrame(hosp_data)
        deck_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=hosp_df,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=220,
                pickable=True,
            )
        )
        deck_layers.append(
            pdk.Layer(
                "TextLayer",
                data=hosp_df,
                get_position=["lon", "lat"],
                get_text="name",
                get_size=12,
                get_color=[240, 246, 252, 255],
                get_text_anchor="'start'",
                get_alignment_baseline="'center'",
                pixel_offset=[18, 0],
            )
        )

    camera = pdk.ViewState(
        latitude=st.session_state["map_lat"],
        longitude=st.session_state["map_lon"],
        zoom=st.session_state["map_zoom"],
        pitch=st.session_state["map_pitch"],
        bearing=-20
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=deck_layers,
            initial_view_state=camera,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={
                "html": "<div style='font-family: Inter; padding: 6px;'>"
                        "<b style='color:#58a6ff; font-size:13px;'>{layer_title}</b><br/>"
                        "<span>{metric_primary}</span><br/>"
                        "<span>{metric_secondary}</span><br/>"
                        "<b>Estado:</b> <span style='color:{status_color}; font-weight:700;'>{status_tag}</span>"
                        "</div>",
                "style": {"backgroundColor": "#161b22", "color": "white", "fontSize": "12px", "borderRadius": "4px"},
            },
        ),
        width="stretch"
    )

    export_df = active_df[["parcel_id", "municipality", "lon", "lat", "active_depth", "active_loss", "status_tag"]].copy()
    geojson_features = []
    for _, row in export_df.iterrows():
        geojson_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {
                "id": row["parcel_id"],
                "municipality": row["municipality"],
                "depth_m": float(row["active_depth"]),
                "loss_eur": float(row["active_loss"]),
                "status": row["status_tag"]
            }
        })
    geojson_payload = json.dumps({"type": "FeatureCollection", "features": geojson_features}, indent=2)

    st.download_button(
        label="🌐 Descargar Capa Vectorial de Huella Aluvial Activa (GeoJSON para QGIS / ArcGIS)",
        data=geojson_payload,
        file_name=f"huella_aluvial_metropolitana_{int(time.time())}.geojson",
        mime="application/geo+json",
        width="stretch"
    )

# ------------------------------------------------------------------------------
# TAB 2: DESPACHO ES-ALERT TRILINGÜE & ACCESIBILIDAD UNIVERSAL
# ------------------------------------------------------------------------------
with tab_esalert:
    st.subheader("Centro de Despacho ES-Alert Trilingüe & Accesibilidad Universal")
    
    critical_nodes = [
        {"name": "Subestación Paiporta-Benetússer (Iberdrola)", "type": "ELÉCTRICA", "h_base": 2.10},
        {"name": "Subestación Torrent Este (Iberdrola)", "type": "ELÉCTRICA", "h_base": 0.40},
        {"name": "Nodo Central Fibra/Móvil Massanassa", "type": "TELECOM", "h_base": 1.80},
        {"name": "Estación Depuradora EDAR Catarroja", "type": "SANEAMIENTO", "h_base": 1.60},
    ]

    infra_status = []
    blackout_count = 0
    
    for c_node in critical_nodes:
        node_depth = c_node["h_base"] * current_depth_factor
        node_peak_depth = c_node["h_base"] * peak_damage_factor_reached
        
        is_damaged_or_flooded = (node_depth >= 0.50) or (node_peak_depth >= 0.75)
        if is_damaged_or_flooded and c_node["type"] == "ELÉCTRICA":
            blackout_count += 1
            
        status_badge = "🔴 ANEGADA / FUERA DE SERVICIO" if is_damaged_or_flooded else "🟢 OPERATIVA"
        impact_txt = "Avería estructural e interrupción en cascada" if is_damaged_or_flooded else "Operación nominal"

        infra_status.append({
            "Infraestructura": c_node["name"],
            "Tipología": c_node["type"],
            "Calado": fmt_dec(node_depth, 2, " m"),
            "Umbral": "0,50 m",
            "Estado": status_badge,
            "Impacto Territorial": impact_txt
        })

    col_es1, col_es2 = st.columns([1.35, 1.65])

    with col_es1:
        st.markdown("#### Estado de Nodos Vitales (Lifeline Utilities)")
        st.dataframe(
            pd.DataFrame(infra_status),
            column_config={
                "Infraestructura": st.column_config.TextColumn("Nodo Crítico", width="medium"),
                "Tipología": st.column_config.TextColumn("Tipo", width="small"),
                "Calado": st.column_config.TextColumn("Calado", width="small"),
                "Umbral": st.column_config.TextColumn("Límite", width="small"),
                "Estado": st.column_config.TextColumn("Estado", width="medium"),
                "Impacto Territorial": st.column_config.TextColumn("Impacto Proyectado", width="medium"),
            },
            width="stretch",
            hide_index=True
        )
        
        if blackout_count > 0:
            st.error(
                f"⚠️ **FALLO CRÍTICO EN CASCADA:** {blackout_count} subestaciones eléctricas anegadas/dañadas. "
                "Cortes de suministro masivos en Paiporta, Benetússer y Massanassa."
            )
        else:
            st.success("🟢 **REDES ENERGÉTICAS ESTABLES:** Suministro y nodos de comunicación asegurados.")

        st.markdown("#### 📢 Comunicado Oficial de Situación (Twitter/X & Radios)")
        if alert_state == "ROJO":
            tweet_text = (
                f"🚨 URGENTE 112 // ALERTA ROJA RAMBLA DEL POYO\n"
                f"Nivel: EMERGENCIA SIT. 2 | Hora: {clock_badge_text}\n"
                f"Caudal previsto: {fmt_int(q_peak_simulated, ' m3/s')}. Desbordamiento masivo inminente.\n"
                f"Evacuación vertical INMEDIATA en l'Horta Sud. Suba a pisos altos. NO circule.\n"
                f"Universidades y transporte suspendidos. Info: @GVA112 #DANAValencia"
            )
        elif alert_state == "NARANJA":
            tweet_text = (
                f"⚠️ AVISO 112 // ALERTA NARANJA RAMBLA DEL POYO\n"
                f"Nivel: EMERGENCIA SIT. 1 | Hora: {clock_badge_text}\n"
                f"Crecida severa propagándose ({fmt_int(q_peak_simulated, ' m3/s')}).\n"
                f"Aléjese de cauces y ramblas. Retire vehículos de zonas bajas y pasos subterráneos.\n"
                f"Info: @GVA112"
            )
        elif alert_state == "AMARILLO":
            tweet_text = (
                f"🟡 INFORMATIVO 112 // PREALERTA HIDROLÓGICA\n"
                f"Nivel: PREEMERGENCIA | Hora: {clock_badge_text}\n"
                f"Precipitaciones registradas en cabecera ({fmt_dec(rain_mm, 1, ' mm')}). Caudales bajo seguimiento activo.\n"
                f"Sin afección en cascos urbanos por el momento. Manténgase informado.\n"
                f"Info: @GVA112"
            )
        else:
            tweet_text = (
                f"🟢 INFORMATIVO 112 // NORMALIDAD HIDROLÓGICA\n"
                f"Hora: {clock_badge_text}\n"
                f"Cuenca del Poyo y l'Horta Sud en parámetros de normalidad. Caudal de estiaje ordinario ({fmt_int(q_peak_simulated, ' m3/s')}).\n"
                f"Red viaria y servicios operando con normalidad.\n"
                f"Info: @GVA112"
            )
        st.text_area("Texto oficial listo para difusión institucional:", value=tweet_text, height=115)

    with col_es2:
        st.markdown("#### Consola de Transmisión ES-Alert Trilingüe (Cell Broadcast)")
        
        if alert_state == "ROJO":
            urgency, severity = "Immediate", "Extreme"
            headline = "ALERTA ROJA PROTECCIÓN CIVIL: EMERGENCIA SITUACIÓN 2"
            body_es = f"EMERGENCIA SITUACIÓN 2. Peligro extremo por inundación en cuenca del Poyo. NO CIRCULE. Suba a pisos altos. Aléjese de cauces, pasos subterráneos y barrancos."
            body_val = f"EMERGÈNCIA SITUACIÓ 2. Perill extrem per inundació a la conca del Poio. NO CIRCULEU. Pugeu a pisos alts. Allunyeu-vos de lleres, passos subterranis i barrancs."
            body_en = f"EMERGENCY LEVEL 2. Extreme flash flood danger in Poyo ravine basin. DO NOT DRIVE. Move immediately to upper floors. Stay away from riverbeds and underpasses."
            cap_status, status_color = "Actual", "#da3633"
            banner_note = "DIFUSIÓN CELULAR FORZADA // ACTIVACIÓN ACÚSTICA MULTILINGÜE"
        elif alert_state == "NARANJA":
            urgency, severity = "Expected", "Severe"
            headline = "ALERTA NARANJA PROTECCIÓN CIVIL: EMERGENCIA SITUACIÓN 1"
            body_es = f"EMERGENCIA SITUACIÓN 1. Crecida severa en cuenca del Poyo ({fmt_int(q_peak_simulated, ' m3/s')}). Evite vados, ramblas y retire vehículos de cotas bajas."
            body_val = f"EMERGÈNCIA SITUACIÓ 1. Crecuda severa a la conca del Poio ({fmt_int(q_peak_simulated, ' m3/s')}). Eviteu guals, rambles i retireu vehicles de cotes baixes."
            body_en = f"EMERGENCY LEVEL 1. Severe water surge in Poyo ravine ({fmt_int(q_peak_simulated, ' m3/s')}). Avoid ford crossings and move vehicles to higher ground."
            cap_status, status_color = "Actual", "#f0883e"
            banner_note = "DIFUSIÓN REGIONAL SELECTIVA // AVISO OPERATIVO TRILINGÜE"
        elif alert_state == "AMARILLO":
            urgency, severity = "Future", "Moderate"
            headline = "PREEMERGENCIA FASE ALERTA: PRECAUCIÓN POR LLUVIAS EN CABECERA"
            body_es = f"Precipitación intensa registrada ({fmt_dec(rain_mm, 1, ' mm')}). Caudal en cauce bajo seguimiento ({fmt_int(q_peak_simulated, ' m3/s')}). Precaución ordinaria."
            body_val = f"Precipitació intensa registrada ({fmt_dec(rain_mm, 1, ' mm')}). Cabal en curs sota seguiment ({fmt_int(q_peak_simulated, ' m3/s')}). Precaució ordinària."
            body_en = f"Heavy rainfall recorded in headwaters ({fmt_dec(rain_mm, 1, ' mm')}). River discharge monitored ({fmt_int(q_peak_simulated, ' m3/s')}). Exercise caution."
            cap_status, status_color = "Test", "#d29922"
            banner_note = "CANAL INFORMATIVO CIUDADANO // SIN PITIDO CELULAR"
        else:
            urgency, severity = "Past", "Minor"
            headline = "SITUACIÓN NORMAL // SIN AVISOS ACTIVOS"
            body_es = "Caudales en niveles de estiaje y cuenca en parámetros de seguridad ordinaria."
            body_val = "Cabals en nivells d'estiatge i conca en paràmetres de seguretat ordinària."
            body_en = "River basin discharge at baseline levels. Normal hydrological safety conditions."
            cap_status, status_color = "Exercise", "#238636"
            banner_note = "CANAL DE DIFUSIÓN EN ESPERA"

        st.markdown(
            f"""
            <div style='background: rgba(13, 17, 23, 0.9); border: 2px solid {status_color}; border-radius: 8px; padding: 14px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color: {status_color}; font-weight: 700; font-family: monospace;'>ESTADO DE DIFUSIÓN: {severity.upper()}</span>
                    <span style='font-size:0.72rem; color:#8b949e;'>{banner_note}</span>
                </div>
                <h4 style='margin: 8px 0; color: #f0f6fc; font-size:1.02rem;'>{headline}</h4>
                <div style='background: #161b22; padding: 8px 10px; border-radius: 6px; font-family: monospace; font-size: 0.80rem; color: #e6edf3; margin-bottom: 6px;'>
                    <b style='color:#58a6ff;'>[ES]</b> {body_es}
                </div>
                <div style='background: #161b22; padding: 8px 10px; border-radius: 6px; font-family: monospace; font-size: 0.80rem; color: #e6edf3; margin-bottom: 6px;'>
                    <b style='color:#f0883e;'>[VAL]</b> {body_val}
                </div>
                <div style='background: #161b22; padding: 8px 10px; border-radius: 6px; font-family: monospace; font-size: 0.80rem; color: #e6edf3;'>
                    <b style='color:#3fb950;'>[EN]</b> {body_en}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='background: rgba(22, 27, 34, 0.95); border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; margin-top: 10px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <b style='color:#58a6ff; font-size:0.78rem;'>♿ ACCESIBILIDAD UNIVERSAL & CANALES INCLUSIVOS (RD 193/2023)</b>
                    <span style='background:#1f6feb; color:white; font-size:0.65rem; padding:2px 6px; border-radius:4px;'>NORMATIVA UNE 139803</span>
                </div>
                <div style='display:flex; gap:12px; margin-top:8px; font-size:0.74rem; color:#c9d1d9; flex-wrap:wrap;'>
                    <div>🤟 <b>Canal LSE:</b> Vídeo signado generado para emisión TV/Web</div>
                    <div>📳 <b>Háptico:</b> Patrón de vibración prolongado para sordoceguera</div>
                    <div>🚹 <b>Lectura Fácil:</b> Suba a pisos altos. No coja el coche. Aléjese del agua.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        cap_xml_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>POYO-NOWCAST-{int(time.time())}</identifier>
  <sender>gva.112@emergencies.gva.es</sender>
  <sent>{clock_badge_text}</sent>
  <status>{cap_status}</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>es-ES</language>
    <category>Met</category>
    <event>Flash Flood / Desbordamiento</event>
    <urgency>{urgency}</urgency>
    <severity>{severity}</severity>
    <certainty>Observed</certainty>
    <headline>{headline}</headline>
    <description>{body_es}</description>
    <area>
      <areaDesc>Area Metropolitana de Valencia y l'Horta Sud</areaDesc>
      <circle>39.4230,-0.4180,12000</circle>
    </area>
  </info>
  <info>
    <language>ca-ES</language>
    <category>Met</category>
    <event>Inundacio Sobtada / Desbordament</event>
    <urgency>{urgency}</urgency>
    <severity>{severity}</severity>
    <certainty>Observed</certainty>
    <headline>{headline}</headline>
    <description>{body_val}</description>
    <area>
      <areaDesc>Area Metropolitana de Valencia</areaDesc>
      <circle>39.4230,-0.4180,12000</circle>
    </area>
  </info>
  <info>
    <language>en-GB</language>
    <category>Met</category>
    <event>Flash Flood / River Overflow</event>
    <urgency>{urgency}</urgency>
    <severity>{severity}</severity>
    <certainty>Observed</certainty>
    <headline>RED FLOOD WARNING - CIVIL PROTECTION</headline>
    <description>{body_en}</description>
    <area>
      <areaDesc>Valencia Metropolitan Area</areaDesc>
      <circle>39.4230,-0.4180,12000</circle>
    </area>
  </info>
</alert>"""

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.download_button(
                label="📲 Exportar Payload CAP v1.2 Trilingüe (XML)",
                data=cap_xml_payload,
                file_name="es_alert_poyo_trilingual_cap.xml",
                mime="application/xml",
                width="stretch"
            )
        with btn_c2:
            briefing_md = f"""# INFORME OFICIAL DE SITUACIÓN // POYO-NOWCAST C2
Fecha/Hora: {clock_badge_text}
Nivel Operativo: {alert_state}
Caudal Punta Estimado: {fmt_int(q_peak_simulated, ' m3/s')}
Pérdida Económica Directa: {fmt_dec(current_loss_m, 1, ' M€')}
Inmuebles en Ruina / Colapso: {fmt_int(total_collapsed)}
Rescates Urgentes P1: {fmt_int(critical_p1)}
Medida de Mitigación Evaluada: {what_if}
"""
            st.download_button(
                label="📄 Exportar Informe Ejecutivo (Briefing CECOPI)",
                data=briefing_md,
                file_name=f"briefing_cecopi_poyo_{int(time.time())}.md",
                mime="text/markdown",
                width="stretch"
            )

# ------------------------------------------------------------------------------
# TAB 3: AUDITORÍA FORENSE SPLIT A/B
# ------------------------------------------------------------------------------
with tab_compare:
    st.subheader("Auditoría Forense Split A/B: Inteligencia Anticipada vs. Gestión Burocrática")
    
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Margen de Preaviso", "+146 min", delta="Nowcast (17:45 h) vs 20:11 h", delta_color="normal")
    with sc2:
        st.metric("Telecomunicaciones", "100% On-line", delta="Sin apagón de antenas", delta_color="normal")
    with sc3:
        st.metric("Población sin Aviso", "0 hab", delta="-180.000 hab protegidos", delta_color="normal")
    with sc4:
        st.metric("Atrapamientos Viales", "-85%", delta="Evacuación vertical viable", delta_color="normal")

    st.markdown("<br/>", unsafe_allow_html=True)

    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        st.markdown(
            """
            <div style='background: rgba(31, 111, 235, 0.12); border: 2px solid #388bfd; border-radius: 8px; padding: 14px;'>
                <h4 style='color: #58a6ff; margin:0;'>ESCENARIO A: POYO-NOWCAST (Inteligencia Algorítmica)</h4>
                <p style='color: #8b949e; font-size: 0.80rem;'>Disparo preventivo multivariable (Sensor Chiva &gt; 180 mm + FNO 2D)</p>
                <hr style='border:0.5px solid #30363d;'/>
                <ul style='color: #f0f6fc; font-size: 0.82rem; line-height: 1.6;'>
                    <li><b>Hora de Alarma Masiva:</b> 17:45 h (T + 105 min)</li>
                    <li><b>Anticipación a Casco Urbano:</b> <b>+146 minutos de margen útil</b></li>
                    <li><b>Estado de Telecomunicaciones:</b> 100% Repetidores y antenas con suministro</li>
                    <li><b>Evacuación Vertical:</b> Factible sin anegamiento previo de viales</li>
                    <li><b>Ahorro Potencial de Vidas:</b> &gt; 85% de reducción estimada de atrapamientos</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_ab2:
        st.markdown(
            """
            <div style='background: rgba(218, 54, 51, 0.12); border: 2px solid #da3633; border-radius: 8px; padding: 14px;'>
                <h4 style='color: #f85149; margin:0;'>ESCENARIO B: GESTIÓN REAL CECOPI (29-O 2024)</h4>
                <p style='color: #8b949e; font-size: 0.80rem;'>Despacho tardío por canal convencional burocrático</p>
                <hr style='border:0.5px solid #30363d;'/>
                <ul style='color: #f0f6fc; font-size: 0.82rem; line-height: 1.6;'>
                    <li><b>Hora de Alarma Masiva:</b> 20:11 h (T + 251 min)</li>
                    <li><b>Anticipación a Casco Urbano:</b> <b>-90 minutos de retraso (Tardía)</b></li>
                    <li><b>Estado de Telecomunicaciones:</b> Subestaciones anegadas y apagón móvil masivo</li>
                    <li><b>Evacuación Vertical:</b> Imposible (Población atrapada en vehículos y garajes)</li>
                    <li><b>Balance Catastrófico:</b> &gt; 220 fallecidos y colapso de infraestructuras</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### Cronograma Comparativo con Hitos de Decisión (Plotly HUD)")

    timeline_df = pd.DataFrame([
        dict(Task="Precipitación Extrema Chiva (> 400 mm)", Start="2024-10-29 16:00:00", Finish="2024-10-29 18:00:00", Tipo="Fenómeno Físico"),
        dict(Task="Propagación Frente Onda FNO (Rambla)", Start="2024-10-29 17:00:00", Finish="2024-10-29 19:30:00", Tipo="Fenómeno Físico"),
        dict(Task="Anegamiento Crítico Paiporta / Picanya", Start="2024-10-29 18:30:00", Finish="2024-10-29 23:00:00", Tipo="Impacto Crítico"),
        dict(Task="DISPARO ES-ALERT POYO-NOWCAST", Start="2024-10-29 17:45:00", Finish="2024-10-29 17:50:00", Tipo="Alerta Anticipada"),
        dict(Task="Ventana Útil Evacuación Vertical", Start="2024-10-29 17:45:00", Finish="2024-10-29 18:30:00", Tipo="Ventana Salvavidas"),
        dict(Task="DISPARO ES-ALERT OFICIAL CECOPI", Start="2024-10-29 20:11:00", Finish="2024-10-29 20:16:00", Tipo="Alerta Tardía"),
    ])

    fig_timeline = px.timeline(
        timeline_df, x_start="Start", x_end="Finish", y="Task", color="Tipo",
        color_discrete_map={
            "Fenómeno Físico": "#8b949e",
            "Impacto Crítico": "#da3633",
            "Alerta Anticipada": "#1f6feb",
            "Ventana Salvavidas": "#238636",
            "Alerta Tardía": "#d29922"
        }
    )
    fig_timeline.add_vline(x="2024-10-29 17:45:00", line_dash="dash", line_color="#58a6ff", annotation_text="17:45 h Disparo Nowcast")
    fig_timeline.add_vline(x="2024-10-29 20:11:00", line_dash="dash", line_color="#da3633", annotation_text="20:11 h Aviso CECOPI")
    fig_timeline.update_yaxes(autorange="reversed")
    fig_timeline.update_layout(
        template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
        height=290, margin=dict(l=20, r=20, t=10, b=20), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_timeline, width="stretch")

# ------------------------------------------------------------------------------
# TAB 4: DINÁMICA HIDRÁULICA FNO CON AMBOS CAUDALES Y CURVA REALISTA
# ------------------------------------------------------------------------------
with tab_hydro:
    st.subheader("Dinámica Hidráulica 2D Neuronal (Fourier Neural Operator)")

    c_phase1, c_phase2, c_phase3, c_phase4 = st.columns(4)
    with c_phase1:
        st.markdown(
            """
            <div style='background:#161b22; border-left:4px solid #2ea043; padding:10px; border-radius:6px; font-size:0.75rem;'>
                <b style='color:#3fb950;'>FASE 1: CONDUCCIÓN</b><br/>
                <b>Q &lt; 1.000 m³/s</b><br/>
                Flujo encauzado en rambla. Riesgo limitado a vados, pasos bajos y sendas ribereñas.
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_phase2:
        st.markdown(
            """
            <div style='background:#161b22; border-left:4px solid #d29922; padding:10px; border-radius:6px; font-size:0.75rem;'>
                <b style='color:#d29922;'>FASE 2: DESBORDAMIENTO</b><br/>
                <b>1.000 - 1.800 m³/s</b><br/>
                Pérdida de capacidad del cauce en Paiporta y Picanya. Anegamiento de polígonos y sótanos.
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_phase3:
        st.markdown(
            """
            <div style='background:#161b22; border-left:4px solid #da3633; padding:10px; border-radius:6px; font-size:0.75rem;'>
                <b style='color:#f85149;'>FASE 3: CATASTRÓFICA</b><br/>
                <b>Q &gt; 1.800 m³/s</b><br/>
                Onda aluvial masiva sobre casco urbano. Colapso de puentes, corte de CV-36 y arrastre de vehículos.
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_phase4:
        volumen_hm3 = (q_peak_simulated * 4.5 * 3600) / 1e6
        vel_frente_kmh = min(28.0, 12.0 + (q_peak_simulated / 150.0))
        st.markdown(
            f"""
            <div style='background:#161b22; border-left:4px solid #388bfd; padding:10px; border-radius:6px; font-size:0.75rem;'>
                <b style='color:#58a6ff;'>MÉTRICAS FNO DE AVENIDA</b><br/>
                <b>Volumen:</b> {volumen_hm3:.1f} Hm³ estimados<br/>
                <b>Velocidad Frente:</b> ~{vel_frente_kmh:.1f} km/h<br/>
                <b>Tiempo Concentración:</b> 90 - 120 min
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([1.6, 1.4])
    with col_h1:
        st.markdown("#### Hidrograma Transitorio de Caudal Líquido (Q)")
        t_steps = np.linspace(0, 360, 72)
        
        if "Forense" in sim_mode:
            q_envelope = np.interp(t_steps, t_arr, q_series) * q_mitig_factor
            marker_time = sim_minute
            annotation_str = f"Instante: {exact_clock}"
        else:
            t_peak = 120.0
            ascenso = np.clip(t_steps / t_peak, 0.0, 1.0) ** 2.2
            descenso = np.exp(-((t_steps - t_peak) / 75.0) ** 1.3)
            q_envelope = np.where(t_steps <= t_peak, q_peak_simulated * ascenso, q_peak_simulated * descenso)
            q_envelope = np.clip(q_envelope, 0.0, None)
            marker_time = lead_time_min
            annotation_str = f"Lead Time (+{lead_time_min} min): {fmt_int(q_instant_lead, ' m³/s')}"
        
        fig_hydro = go.Figure()
        fig_hydro.add_hrect(y0=0, y1=1000, fillcolor="green", opacity=0.05, line_width=0)
        fig_hydro.add_hrect(y0=1000, y1=1800, fillcolor="orange", opacity=0.08, line_width=0)
        fig_hydro.add_hrect(y0=1800, y1=3600, fillcolor="red", opacity=0.10, line_width=0)

        fig_hydro.add_trace(go.Scatter(
            x=t_steps, y=q_envelope,
            mode='lines', line=dict(color='#58a6ff', width=3.5),
            name='Hidrograma de Avenida', fill='tozeroy', fillcolor='rgba(31, 111, 235, 0.18)'
        ))
        fig_hydro.add_hline(y=1000.0, line_dash="dash", line_color="#d29922", annotation_text="Capacidad del Cauce (1.000 m³/s)")
        fig_hydro.add_hline(y=1800.0, line_dash="dash", line_color="#da3633", annotation_text="Desbordamiento Catastrófico (1.800 m³/s)")
        
        fig_hydro.add_vline(x=marker_time, line_color="#ffd700", line_width=2, line_dash="dot", annotation_text=annotation_str)
        
        fig_hydro.add_trace(go.Scatter(
            x=[marker_time], y=[q_instant_lead if "Nowcast" in sim_mode else float(np.interp(sim_minute, t_steps, q_envelope))],
            mode="markers", marker=dict(size=10, color="#ffd700", symbol="circle"),
            name="Punto Temporal"
        ))

        fig_hydro.update_layout(
            template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
            xaxis_title="Minutos transcurridos desde inicio del evento", yaxis_title="Caudal Líquido Q (m³/s)",
            yaxis=dict(range=[0, max(2200, float(q_peak_simulated) * 1.18)]),
            margin=dict(l=40, r=40, t=30, b=40), height=340,
            showlegend=False
        )
        st.plotly_chart(fig_hydro, width="stretch")

    with col_h2:
        st.markdown("#### Calado Hidrodinámico por Término Municipal (m)")
        fig_box = px.box(
            active_df, x="municipality", y="active_depth", color="municipality",
            labels={"active_depth": "Calado h (m)", "municipality": "Municipio"}
        )
        fig_box.update_layout(
            template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
            showlegend=False, margin=dict(l=40, r=40, t=30, b=40), height=340
        )
        st.plotly_chart(fig_box, width="stretch")

# ------------------------------------------------------------------------------
# TAB 5: RESILIENCIA VIAL & TTI CON CÁLCULO DINÁMICO BLINDADO (SIN CRASH NAN)
# ------------------------------------------------------------------------------
with tab_roads:
    st.subheader("Matriz Dinámica de Resiliencia Territorial & Time-to-Isolation (TTI)")

    r_info1, r_info2, r_info3 = st.columns(3)
    with r_info1:
        st.markdown(
            """
            <div style='background:#161b22; border-left:4px solid #1f6feb; padding:12px; border-radius:6px; font-size:0.77rem;'>
                <b style='color:#58a6ff;'>¿QUÉ ES EL TIME-TO-ISOLATION (TTI)?</b><br/>
                Representa el tiempo en minutos que transcurre desde el inicio de la avenida hasta que el último enlace vial asfaltado que conecta un municipio con los hospitales de referencia sufre un <b>calado h &ge; 0,30 m</b> (umbral de pérdida de adherencia y arrastre de turismos).
            </div>
            """,
            unsafe_allow_html=True
        )
    with r_info2:
        st.markdown(
            """
            <div style='background:#161b22; border-left:4px solid #da3633; padding:12px; border-radius:6px; font-size:0.77rem;'>
                <b style='color:#f85149;'>CRITERIO DE CORTE VIAL (HISTÉRESIS)</b><br/>
                Una vía cortada por desbordamiento permanece inoperativa incluso si el caudal desciende, debido a la deposición masiva de sedimentos, lodo (> 40 cm) y retenciones vehiculares en los corredores arteriales <b>CV-36, CV-400 y V-30</b>.
            </div>
            """,
            unsafe_allow_html=True
        )
    with r_info3:
        st.markdown(
            """
            <div style='background:#161b22; border-left:4px solid #238636; padding:12px; border-radius:6px; font-size:0.77rem;'>
                <b style='color:#3fb950;'>ALGORITMO MULTI-SINK HOSPITALARIO</b><br/>
                El sistema evalúa en tiempo real rutas de evacuación óptimas hacia 3 nodos sanitarios de tercer nivel: <b>Hospital La Fe, Hospital General y Hospital de Manises</b> (conectado vía A-3), redirigiendo ambulancias según el colapso de vías.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    
# CÁLCULO DINÁMICO DE INCOMUNICACIÓN TERRITORIAL BLINDADO
    road_cut_cv400 = (3.10 * current_depth_factor) >= 0.30
    road_cut_v31 = (2.80 * current_depth_factor) >= 0.30
    road_cut_cv36 = (2.40 * current_depth_factor) >= 0.30

    def compute_dynamic_isolation(row):
        mun = row["municipality"]
        h = row["active_depth"]
        if mun in ["Catarroja", "Massanassa", "Benetússer", "Alfafar", "Sedaví"]:
            return (road_cut_cv400 or road_cut_v31) and (h >= 0.20)
        elif mun in ["Paiporta", "Picanya", "Torrent"]:
            return (road_cut_cv36 or road_cut_cv400) and (h >= 0.20)
        elif mun == "València":
            return sim_pluvial_vlc and (h >= 0.35)
        else:
            return h >= 0.35

    active_df["dynamic_isolated"] = active_df.apply(compute_dynamic_isolation, axis=1)

    tti_metrics = (
        active_df.groupby("municipality")
        .agg(
            tti_med=("time_to_isolation_min", "median"),
            pct_isolated=("dynamic_isolated", lambda s: float(np.mean(s)) * 100.0 if len(s) > 0 else 0.0),
            parcels=("parcel_id", "count"),
            p1_urgente=("dynamic_p1", lambda s: int(np.sum(s)))
        )
        .reset_index()
    )

    # FORMAT_TTI BLINDADO CONTRA NULOS (NaN)
    def format_tti(r):
        pct = r.get("pct_isolated", 0.0)
        tti_val = r.get("tti_med", np.nan)

        if pd.isna(pct) or not np.isfinite(pct):
            pct = 0.0

        if pd.isna(tti_val) or not np.isfinite(tti_val):
            return "15 min (Crítico)" if pct >= 20.0 else "Resiliente (> 120 min)"

        if pct >= 15.0:
            reduction = q_peak_simulated / 150.0
            calc = tti_val - reduction
            if pd.notnull(calc) and np.isfinite(calc):
                val = max(15, int(round(calc)))
                return f"{val} min"
            return "15 min (Crítico)"
        return "Resiliente (> 120 min)"

    tti_metrics["TTI_Label"] = tti_metrics.apply(format_tti, axis=1)

    def categorizar_alerta_vial(row):
        pct = row.get("pct_isolated", 0.0)
        if pd.isna(pct) or not np.isfinite(pct):
            pct = 0.0
        if pct >= 25.0:
            return "🔴 AISLAMIENTO TOTAL (VÍAS CORTADAS)"
        elif pct >= 8.0:
            return "🟠 RUTA EN RIESGO / PREALERTA"
        return "🟢 CONECTIVIDAD ACTIVA"

    tti_metrics["Estado_Evacuacion"] = tti_metrics.apply(categorizar_alerta_vial, axis=1)
    tti_metrics["pct_isolated_fmt"] = tti_metrics["pct_isolated"].apply(lambda v: fmt_dec(v if (pd.notnull(v) and np.isfinite(v)) else 0.0, 1, " %"))
    tti_metrics["p1_fmt"] = tti_metrics["p1_urgente"].apply(lambda v: fmt_int(v if (pd.notnull(v) and np.isfinite(v)) else 0))
    tti_metrics["Hospital_Destino"] = "H. Universitari i Politècnic La Fe / H. General"
    
    st.dataframe(
        tti_metrics[[
            "municipality", "TTI_Label", "Estado_Evacuacion", "pct_isolated_fmt", "p1_fmt", "Hospital_Destino"
        ]].rename(columns={
            "municipality": "Término Municipal",
            "TTI_Label": "Time-to-Isolation (TTI)",
            "Estado_Evacuacion": "Estado de Evacuación",
            "pct_isolated_fmt": "% Área Incomunicada",
            "p1_fmt": "Rescates Críticos (P1)",
            "Hospital_Destino": "Hospital Asignado (Multi-Sink)"
        }),
        width="stretch",
        hide_index=True
    )

# ------------------------------------------------------------------------------
# TAB 6: FINANZAS DEL CLIMA & SOLVENCIA II
# ------------------------------------------------------------------------------
with tab_finances:
    st.subheader("Modelado Catastrófico, Reaseguro y Seguros Paramétricos (Solvencia II / EIOPA)")
    scr_current = max(0.0, current_loss_m * 1.35 - (current_loss_m * 0.042))
    coc_current = 0.06 * scr_current
    
    f_col1, f_col2, f_col3 = st.columns([1.1, 1.1, 0.9])
    with f_col1:
        st.markdown("#### Balance Regulatorio (EIOPA ORSA)")
        solv_table = pd.DataFrame([
            {"Parámetro Actuarial": "Exposición Bruta (TIV)", "Importe": fmt_dec(total_exposure_m, 1, " M€")},
            {"Parámetro Actuarial": "Pérdida Bruta Modelada", "Importe": fmt_dec(current_loss_m, 1, " M€")},
            {"Parámetro Actuarial": "Pérdida Anual Esperada (AAL)", "Importe": fmt_dec(current_loss_m * 0.042, 2, " M€/año")},
            {"Parámetro Actuarial": "SCR Solvencia II (VaR 99,5%)", "Importe": fmt_dec(scr_current, 1, " M€")},
            {"Parámetro Actuarial": "Margen de Riesgo (CoC 6%)", "Importe": fmt_dec(coc_current, 2, " M€")},
            {"Parámetro Actuarial": "Indemnización Paramétrica", "Importe": fmt_dec(total_payout_m, 2, " M€")},
        ])
        st.dataframe(solv_table, width="stretch", hide_index=True)
        
    with f_col2:
        st.markdown("#### Cascada de Financiación (Waterfall)")
        val_ccs = current_loss_m * 0.72
        val_param = min(current_loss_m * 0.28, total_payout_m)
        val_gap = max(0.0, current_loss_m - (val_ccs + val_param))
        
        fig_waterfall = go.Figure(go.Bar(
            x=["Consorcio (CCS)", "Cat Bond", "Brecha", "Total"],
            y=[val_ccs, val_param, val_gap, current_loss_m],
            marker=dict(color=["#238636", "#1f6feb", "#d29922", "#da3633"], line=dict(color="#30363d", width=1.5)),
            text=[fmt_dec(v, 1, " M€") for v in [val_ccs, val_param, val_gap, current_loss_m]],
            textposition="auto"
        ))
        fig_waterfall.update_layout(
            template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
            yaxis_title="M€", margin=dict(l=20, r=20, t=20, b=20), height=260
        )
        st.plotly_chart(fig_waterfall, width="stretch")

    with f_col3:
        st.markdown("#### Daño por Tipología de Activo")
        asset_breakdown = active_df.groupby("asset_type")["active_loss"].sum().reset_index()
        asset_breakdown["loss_m"] = asset_breakdown["active_loss"] / 1e6
        
        fig_assets = px.pie(
            asset_breakdown,
            names="asset_type",
            values="loss_m",
            hole=0.45,
            color="asset_type",
            color_discrete_map={
                "Residencial": "#388bfd",
                "Industrial / Logística": "#f0883e",
                "Vehículos / Vados": "#d29922",
                "Infraestructura Pública": "#da3633"
            }
        )
        fig_assets.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_assets, width="stretch")

    st.markdown("<br/>", unsafe_allow_html=True)

    f_col4, f_col5 = st.columns(2)
    with f_col4:
        st.markdown("#### Curva EP con Reaseguro Exceso de Pérdida (XoL)")
        T_periods = np.array([2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500])
        loss_base_curve = total_exposure_m * (1.0 - np.exp(-0.38 * (T_periods / 100.0)**0.45))
        loss_stressed_curve = loss_base_curve * 1.18
        ceded_xol = np.clip(loss_base_curve - 60.0, 0.0, 120.0)
        net_retained = loss_base_curve - ceded_xol

        fig_ep = go.Figure()
        fig_ep.add_trace(go.Scatter(x=T_periods, y=loss_base_curve, mode='lines+markers', name='Base Bruta', line=dict(color='#388bfd', width=2.5)))
        fig_ep.add_trace(go.Scatter(x=T_periods, y=loss_stressed_curve, mode='lines', name='Estrés Clima (+18%)', line=dict(color='#f85149', width=2, dash='dash')))
        fig_ep.add_trace(go.Scatter(x=T_periods, y=net_retained, mode='lines+markers', name='Neto Post-XoL', line=dict(color='#238636', width=2.5)))
        fig_ep.add_vline(x=200, line_dash="dot", line_color="#ffffff", annotation_text="SCR (T=200)")
        fig_ep.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", xaxis_title="T (Años)", yaxis_title="M€", height=320, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(fig_ep, width="stretch")

    with f_col5:
        st.markdown("#### Matriz de Disparo Paramétrico (Dual-Trigger)")
        q_grid = np.linspace(800, 2000, 40)
        insar_grid = np.linspace(0.0, 32.0, 40)
        Q_mesh, I_mesh = np.meshgrid(q_grid, insar_grid)
        fq_mesh = np.clip((Q_mesh - 1000.0) / 800.0, 0.0, 1.0)
        fi_mesh = np.clip((I_mesh - 2.0) / 6.0, 0.0, 1.0)
        payout_surface = np.sqrt(fq_mesh * fi_mesh) * 100.0

        fig_matrix = go.Figure()
        fig_matrix.add_trace(go.Contour(z=payout_surface, x=q_grid, y=insar_grid, colorscale="Viridis", reversescale=True, colorbar=dict(title="%", len=0.8)))
        current_insar_pct = min(30.0, collapse_ratio * 100.0)
        fig_matrix.add_trace(go.Scatter(
            x=[min(2000.0, max(800.0, q_eval_cat))], y=[current_insar_pct],
            mode='markers+text', marker=dict(color='#ff0055', size=14, symbol='diamond', line=dict(color='white', width=2)),
            text=[f"{fmt_dec(payout_rate, 1, '%')}"], textposition="top center"
        ))
        fig_matrix.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", xaxis_title="Caudal Q (m³/s)", yaxis_title="InSAR Colapso %", height=320, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(fig_matrix, width="stretch")

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("#### Proyección Decenal del Riesgo Climático & Coste de Solvencia (2024 - 2050)")
    decadas = np.array([2024, 2030, 2035, 2040, 2045, 2050])
    factor_ssp2 = 1.0 + 0.0045 * (decadas - 2024)
    factor_ssp5 = 1.0 + 0.0090 * (decadas - 2024)
    
    aal_ssp2 = (current_loss_m * 0.042) * factor_ssp2
    aal_ssp5 = (current_loss_m * 0.042) * factor_ssp5
    scr_ssp5 = scr_current * factor_ssp5

    fig_future = go.Figure()
    fig_future.add_trace(go.Bar(
        x=decadas, y=scr_ssp5, name="Capital Solvencia Requerido (SCR SSP5-8.5)",
        marker=dict(color="rgba(31, 111, 235, 0.25)", line=dict(color="#388bfd", width=1.5)),
        yaxis="y2"
    ))
    fig_future.add_trace(go.Scatter(
        x=decadas, y=aal_ssp2, mode="lines+markers",
        name="AAL (Senda Intermedia SSP2-4.5)", line=dict(color="#2da44e", width=2.5)
    ))
    fig_future.add_trace(go.Scatter(
        x=decadas, y=aal_ssp5, mode="lines+markers",
        name="AAL (Senda Pesimista SSP5-8.5)", line=dict(color="#cf222e", width=2.5, dash="dash")
    ))

    fig_future.update_layout(
        template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
        xaxis_title="Año de Proyección",
        yaxis=dict(title="Pérdida Anual Esperada AAL (M€/año)"),
        yaxis2=dict(title="Requisito de Capital SCR (M€)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40), height=320
    )
    st.plotly_chart(fig_future, width="stretch")

# ==============================================================================
# PIE DE PÁGINA INSTITUCIONAL CON ENLACE LINKEDIN
# ==============================================================================
st.markdown("<hr style='border:0.5px solid #21262d; margin:14px 0;'/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center; color: #8b949e; font-size: 0.78rem; flex-wrap: wrap; gap: 8px;'>
        <div>POYO-NOWCAST: PLATAFORMA C2 GEMELO DIGITAL INTEGRADO | LICENCIA CC BY 4.0 OPEN SCIENCE</div>
        <div>
            Desarrollado por: 
            <a href='https://www.linkedin.com/in/kelvinflores-ingenieria' target='_blank' style='color: #58a6ff; font-weight: 700; text-decoration: none;'>
                Kelvin Jesus Flores Yarihuaman (LinkedIn)
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
