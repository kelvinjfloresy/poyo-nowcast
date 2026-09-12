"""
POYO-NOWCAST: Módulo 5 - Gemelo Digital Hidrodinámico, Territorial y Actuarial 3D
Tecnología: Streamlit + PyDeck (Deck.gl WebGPU) + Plotly C2 HUD + PyProj Geodésico.
Integración: AEMET OpenData API + FNO 2D + Resiliencia Red Vial + Solvencia II.
Autor: Kelvin Jesus Flores Yarihuaman (https://www.linkedin.com/in/kelvinflores-ingenieria)
Licencia: Open Science (CC BY 4.0)
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
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
except ImportError:
    HAS_AEMET = False

# ==============================================================================
# CONFIGURACIÓN DEL ENTORNO Y ESTILOS HUD C2
# ==============================================================================
st.set_page_config(
    page_title="POYO-NOWCAST | Gemelo Digital Horta Sud",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main { 
        background: radial-gradient(circle at 10% 10%, #0d1117 0%, #04070b 100%);
    }

    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #30363d;
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
        0% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(218, 54, 51, 0); }
        100% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0); }
    }

    .evac-banner-red {
        background: linear-gradient(90deg, rgba(218, 54, 51, 0.35) 0%, rgba(218, 54, 51, 0.12) 100%);
        border: 1px solid #da3633;
        border-left: 6px solid #da3633;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        animation: pulse-evac 1.8s infinite;
    }
    .evac-banner-orange {
        background: linear-gradient(90deg, rgba(219, 109, 40, 0.28) 0%, rgba(219, 109, 40, 0.08) 100%);
        border: 1px solid #bd561d;
        border-left: 6px solid #f0883e;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .badge-alert-red {
        background: #da3633; color: #ffffff; padding: 5px 12px; border-radius: 4px;
        font-weight: 700; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;
        white-space: nowrap;
    }
    .badge-alert-orange {
        background: rgba(219, 109, 40, 0.25); color: #f0883e; border: 1px solid #bd561d;
        padding: 5px 12px; border-radius: 4px; font-weight: 700; font-size: 0.74rem;
        font-family: 'JetBrains Mono', monospace; white-space: nowrap;
    }
    .badge-alert-yellow {
        background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #bb8009;
        padding: 5px 12px; border-radius: 4px; font-weight: 700; font-size: 0.74rem;
        font-family: 'JetBrains Mono', monospace; white-space: nowrap;
    }
    .badge-alert-green {
        background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636;
        padding: 5px 12px; border-radius: 4px; font-weight: 700; font-size: 0.74rem;
        font-family: 'JetBrains Mono', monospace; white-space: nowrap;
    }
    .badge-clock-box {
        background: #161b22; border: 1px solid #388bfd; color: #58a6ff; padding: 5px 10px;
        border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 700;
        white-space: nowrap;
    }

    .telemetry-strip {
        background: rgba(13, 17, 23, 0.95); border: 1px solid #30363d; border-radius: 6px;
        padding: 8px 16px; margin-bottom: 12px; display: flex; justify-content: space-between;
        align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.80rem;
    }

    .legend-box {
        background: rgba(22, 27, 34, 0.92); border: 1px solid #30363d; border-radius: 8px;
        padding: 8px 14px; margin-bottom: 10px; display: flex; gap: 14px; align-items: center;
        flex-wrap: wrap; font-size: 0.76rem;
    }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-bullet { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }

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

# ==============================================================================
# FORMATEADORES NUMÉRICOS EUROPEOS
# ==============================================================================
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
# MOTOR GEODÉSICO Y DATA PIPELINE
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
    {"name": "IES La Sénia (Punto Alto Evacuación)", "mun": "Paiporta", "type": "REFUGIO", "lon": -0.4230, "lat": 39.4290, "beds": 300, "h_base": 0.40},
    {"name": "Residencia Ballesol Sedaví", "mun": "Sedaví", "type": "GERIÁTRICO", "lon": -0.3880, "lat": 39.4260, "beds": 140, "h_base": 1.90},
    {"name": "Centro Sanitario Integrado Catarroja", "mun": "Catarroja", "type": "SALUD", "lon": -0.4050, "lat": 39.4040, "beds": 0, "h_base": 1.80},
]

@st.cache_data
def load_all_system_artifacts():
    parquet_path = "data/processed/flood_damage_matrix.parquet"
    summary_path = "data/processed/solvency_ii_qrt_summary.csv"

    if not os.path.exists(parquet_path):
        os.makedirs(os.path.dirname(os.path.abspath(parquet_path)), exist_ok=True)
        np.random.seed(46)
        n = 5000
        muns = np.random.choice(
            ["Paiporta", "Catarroja", "Sedaví", "Massanassa", "Picanya", "Benetússer", "Alfafar"],
            size=n, p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]
        )
        x = np.random.uniform(720000.0, 726500.0, n)
        y = np.random.uniform(4365000.0, 4371000.0, n)
        d_rambla = np.abs((y - 4368000.0) - 0.4 * (x - 722000.0))
        depth = np.clip(3.2 * np.exp(-d_rambla / 800.0) + np.random.normal(0, 0.05, n), 0.0, 4.2).astype(np.float32)
        vel = np.clip(2.6 * (depth / 3.0) + np.random.normal(0, 0.1, n), 0.0, 3.5).astype(np.float32)
        dpm = np.clip(0.15 + 0.22 * depth + np.random.normal(0, 0.06, n), 0.0, 1.0).astype(np.float32)
        collapse = dpm >= 0.40
        assets = np.random.lognormal(12.1, 0.45, n)
        ratio = np.clip(1.0 / (1.0 + np.exp(-1.8 * (depth - 1.2))), 0.0, 1.0)
        losses = assets * ratio
        isolated = np.isin(muns, ["Paiporta", "Picanya", "Sedaví"]) & (depth > 1.2)
        tti = np.where(isolated, np.random.uniform(20.0, 45.0, n), np.nan).astype(np.float32)
        critical = np.random.choice([True, False], size=n, p=[0.04, 0.96])

        asset_types = np.random.choice(["Residencial", "Industrial / Logística", "Vehículos / Vados", "Infraestructura Pública"], size=n, p=[0.50, 0.30, 0.15, 0.05])

        df = pd.DataFrame({
            "parcel_id": [f"46{np.random.randint(100, 999)}A{i:05d}" for i in range(n)],
            "municipality": muns,
            "asset_type": asset_types,
            "x_coord": x, "y_coord": y,
            "max_depth_m": depth, "max_velocity_ms": vel,
            "hazard_factor_vh": depth * vel,
            "dpm_p90": dpm, "structural_collapse": collapse,
            "asset_value_eur": assets, "damage_ratio": ratio,
            "economic_loss_eur": losses,
            "time_to_isolation_min": tti, "is_isolated": isolated,
            "is_critical_infra": critical, "target_hospital": "HOSPITAL_LA_FE"
        })
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression="snappy")
    else:
        df = pq.read_table(parquet_path).to_pandas()
        if "municipality" in df.columns:
            df["municipality"] = df["municipality"].astype(str)
        if "asset_type" not in df.columns:
            df["asset_type"] = np.random.choice(["Residencial", "Industrial / Logística", "Vehículos / Vados", "Infraestructura Pública"], size=len(df), p=[0.50, 0.30, 0.15, 0.05])

    df["lon"], df["lat"] = PROJECTOR.transform_points(df["x_coord"].to_numpy(), df["y_coord"].to_numpy())

    realistic_roads = [
        {
            "name": "CV-36 Eje Torrent - Picanya - Valencia (Autovía)",
            "h_base": 2.40,
            "coords": [[-0.450, 39.435], [-0.432, 39.439], [-0.418, 39.444], [-0.395, 39.448], [-0.380, 39.452]]
        },
        {
            "name": "V-30 Bulevar Sur / Nuevo Cauce Turia",
            "h_base": 1.20,
            "coords": [[-0.440, 39.458], [-0.415, 39.450], [-0.390, 39.442], [-0.365, 39.435], [-0.340, 39.430]]
        },
        {
            "name": "V-31 Pista de Silla (Acceso Sur A-7)",
            "h_base": 2.80,
            "coords": [[-0.405, 39.385], [-0.395, 39.405], [-0.388, 39.420], [-0.375, 39.438], [-0.370, 39.450]]
        },
        {
            "name": "CV-400 Eje Paiporta - Benetússer - Catarroja",
            "h_base": 3.10,
            "coords": [[-0.402, 39.400], [-0.408, 39.412], [-0.418, 39.425], [-0.422, 39.438], [-0.418, 39.448]]
        },
        {
            "name": "Puente CV-407 Picanya - Paiporta (Cruce Rambla)",
            "h_base": 3.50,
            "coords": [[-0.432, 39.434], [-0.424, 39.430], [-0.415, 39.426], [-0.405, 39.422]]
        },
        {
            "name": "Enlace Bulevar Tres Cruces -> Hospital General",
            "h_base": 0.40,
            "coords": [[-0.415, 39.450], [-0.412, 39.460], [-0.4072, 39.4682]]
        },
        {
            "name": "Corredor Sanitario V-30 Sur -> Hospital La Fe",
            "h_base": 0.50,
            "coords": [[-0.390, 39.442], [-0.382, 39.443], [-0.3768, 39.4435]]
        }
    ]

    qrt_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    return df, realistic_roads, qrt_df


df_parcels, realistic_roads, qrt_summary = load_all_system_artifacts()

# ==============================================================================
# BARRA LATERAL: PANEL DE CONTROL Y RESET ROBUSTO
# ==============================================================================
st.sidebar.markdown(
    """
    <div style='padding: 10px 14px; background: rgba(22, 27, 34, 0.95); border: 1px solid #30363d; border-left: 4px solid #1f6feb; border-radius: 6px; margin-bottom: 12px;'>
        <b style='color: #58a6ff; font-size: 0.92rem;'>POYO-NOWCAST HUD</b><br/>
        <span style='color: #8b949e; font-size: 0.74rem;'>Mando Operativo & Transferencia de Riesgos</span>
    </div>
    """,
    unsafe_allow_html=True,
)

def reset_all_controls_and_gpu():
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["sim_mode_selector"] = "🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)"
    st.session_state["hindcast_slider"] = 210
    st.session_state["render_var_horizontal"] = "🌊 Calado Hidrodinámico FNO"
    st.session_state["aemet_toggle"] = True
    st.session_state["what_if_mitigation"] = "Línea Base (Sin Obras Adicionales)"
    st.session_state["chk_roads"] = True
    st.session_state["chk_hosp"] = True
    st.session_state["chk_vuln"] = True
    st.session_state["sel_muns"] = sorted(df_parcels["municipality"].unique())

if st.sidebar.button("🔄 Restablecer Parámetros (Reset Total)", use_container_width=True):
    reset_all_controls_and_gpu()
    st.rerun()

st.sidebar.markdown("<b style='color:#c9d1d9; font-size:0.80rem;'>⚡ Presets de Escenario Rápido</b>", unsafe_allow_html=True)
p_col1, p_col2 = st.sidebar.columns(2)
with p_col1:
    if st.button("⛈️ DANA Extrema", use_container_width=True):
        st.session_state["sim_mode_selector"] = "🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)"
        st.session_state["hindcast_slider"] = 210
        st.rerun()
with p_col2:
    if st.button("⚠️ Alerta (190 mm)", use_container_width=True):
        st.session_state["sim_mode_selector"] = "Modo Nowcast Predictivo (Tiempo Real)"
        st.session_state["rain_slider"] = 190.0
        st.rerun()

sim_mode = st.sidebar.radio(
    "Modo de Operación:",
    ["🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)", "Modo Nowcast Predictivo (Tiempo Real)"],
    key="sim_mode_selector"
)

st.sidebar.markdown("<hr style='border:0.5px solid #21262d; margin:8px 0;'/>", unsafe_allow_html=True)

st.sidebar.markdown("<b style='color:#c9d1d9; font-size:0.80rem;'>🛡️ Obras de Defensa / Mitigación (What-If)</b>", unsafe_allow_html=True)
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

ruptura_mota = st.sidebar.toggle("💥 Simular Ruptura de Mota en Paiporta", value=False, help="Provoca una brecha en la mota lateral de la rambla incrementando el anegamiento en casco urbano.")

q_mitig_factor = 1.0
if "Presa" in what_if:
    q_mitig_factor = 0.65
elif "Ampliación" in what_if:
    q_mitig_factor = 0.78
elif "Plan Integral" in what_if:
    q_mitig_factor = 0.50

live_obs = None
telemetry_active = False
now_valencia = datetime.now(VALENCIA_TZ)

if "Forense" in sim_mode:
    st.sidebar.markdown("<b style='color:#c9d1d9; font-size:0.80rem;'>⏱️ Progresión Temporal de Avenida</b>", unsafe_allow_html=True)
    sim_minute = st.sidebar.slider(
        "Minuto del Evento (Base 16:00 h = 0 min):",
        min_value=0, max_value=360, value=210, step=15, format="%d min",
        key="hindcast_slider"
    )
    exact_clock = f"{16 + sim_minute // 60:02d}:{sim_minute % 60:02d} h"
    clock_badge_text = f"🕒 29-O-2024 | {exact_clock} (T + {sim_minute} min)"
    
    t_arr = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])
    rain_series = np.array([120.0, 185.0, 260.0, 340.0, 415.0, 465.0, 488.0, 491.2, 491.2, 491.2, 491.2, 491.2])
    q_series = np.array([180.0, 290.0, 460.0, 780.0, 1200.0, 1650.0, 1890.0, 1950.0, 1420.0, 890.0, 520.0, 210.0])
    
    rain_val = float(np.interp(sim_minute, t_arr, rain_series))
    raw_q = float(np.interp(sim_minute, t_arr, q_series))
    q_peak_simulated = raw_q * q_mitig_factor
    current_depth_factor = (q_peak_simulated / 1950.0)
    
    mask_past = t_arr <= sim_minute
    peak_q_so_far = float(np.max(q_series[mask_past])) * q_mitig_factor if np.any(mask_past) else q_peak_simulated
    peak_damage_factor_reached = (peak_q_so_far / 1950.0)
    factor_clima = 1.0

else:
    st.sidebar.markdown("<b style='color:#c9d1d9; font-size:0.80rem;'>⚡ Nowcast & Conexión AEMET</b>", unsafe_allow_html=True)
    conectar_aemet = st.sidebar.toggle("📡 Telemetría AEMET en Vivo", value=True, key="aemet_toggle")
    
    horizonte_clima = st.sidebar.selectbox(
        "Horizonte Climático (IPCC / EIOPA):",
        ["Actual / Línea Base Operativa", "Horizonte 2030 (SSP2-4.5 / +8% Q)", "Horizonte 2040 (SSP3-7.0 / +15% Q)", "Horizonte 2050 (SSP5-8.5 / +22% Q)"],
        key="clim_horizon"
    )
    factor_clima = 1.0 if "Actual" in horizonte_clima else (1.08 if "2030" in horizonte_clima else (1.15 if "2040" in horizonte_clima else 1.22))
    
    lead_time_min = st.sidebar.slider("Avance Temporal (Lead Time):", 15, 180, 60, step=15, format="T + %d min", key="nowcast_lead")
    projected_clock = (now_valencia + timedelta(minutes=lead_time_min)).strftime("%H:%M h")
    clock_badge_text = f"🕒 HORA VALENCIA: {projected_clock} (T + {lead_time_min} min)"

    rain_real = 0.0
    if conectar_aemet and HAS_AEMET:
        aemet_client = AEMETRealTimeClient()
        live_obs = aemet_client.get_basin_live_rainfall()
        telemetry_active = True
        rain_real = live_obs["rain_4h_mm"]
        
        amc_auto = "Seco (AMC I)" if rain_real < 10.0 else ("Normal (AMC II)" if rain_real < 35.0 else "Saturado (AMC III)")
        amc_mode = st.sidebar.selectbox(
            "Humedad Antecedente (AMC):",
            [f"Auto AEMET: {amc_auto}", "Manual: Seco (AMC I)", "Manual: Normal (AMC II)", "Manual: Saturado (AMC III)"],
            key="amc_select"
        )
        amc_weight = 0.75 if "Seco" in amc_mode else (1.0 if "Normal" in amc_mode else 1.25)
        
        rain_val = st.sidebar.slider(
            "Precipitación Cabecera Chiva (Sensor + Forzamiento):",
            min_value=0.0, max_value=600.0, value=float(rain_real), step=5.0,
            format="%.1f mm", key="rain_slider"
        )
    else:
        soil_amc = st.sidebar.select_slider(
            "Humedad Antecedente (AMC):",
            options=["Seco (AMC I)", "Normal (AMC II)", "Saturado (AMC III)"],
            value="Seco (AMC I)", key="soil_amc_slider"
        )
        amc_weight = 1.25 if soil_amc == "Saturado (AMC III)" else (1.0 if soil_amc == "Normal (AMC II)" else 0.75)
        rain_val = st.sidebar.slider(
            "Precipitación Cabecera Chiva (mm / 4h):",
            0.0, 600.0, 0.0, step=5.0, format="%.1f mm", key="rain_manual_slider"
        )

    st.sidebar.markdown(
        """
        <table class='aemet-scale-table'>
            <tr style='color:#8b949e; font-weight:bold;'><td colspan='2'>ESCALA DE INTENSIDAD AEMET</td></tr>
            <tr><td><span style='color:#3fb950;'>●</span> &lt; 40 mm</td><td>Lluvia Ordinaria</td></tr>
            <tr><td><span style='color:#d29922;'>●</span> 40 - 90 mm</td><td>Intensa (Prealerta)</td></tr>
            <tr><td><span style='color:#f0883e;'>●</span> 90 - 180 mm</td><td>Torrencial (Naranja)</td></tr>
            <tr><td><span style='color:#f85149;'>●</span> &gt; 180 mm</td><td>DANA Extrema (Roja)</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    propagation_factor = min(1.0, lead_time_min / 90.0)
    current_depth_factor = (rain_val / 450.0) * amc_weight * factor_clima * propagation_factor * q_mitig_factor
    q_peak_simulated = min(3200.0, 1950.0 * (rain_val / 490.0) * amc_weight * factor_clima * propagation_factor * q_mitig_factor)
    peak_damage_factor_reached = current_depth_factor

st.sidebar.markdown("<hr style='border:0.5px solid #21262d; margin:8px 0;'/>", unsafe_allow_html=True)
st.sidebar.markdown("<b style='color:#c9d1d9; font-size:0.80rem;'>📍 Filtro Territorial y Capas Críticas</b>", unsafe_allow_html=True)
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    show_roads = st.checkbox("Red Viaria Arterial", value=True, key="chk_roads")
    show_vulnerable = st.checkbox("Centros Sensibles", value=True, key="chk_vuln")
with col_s2:
    show_hospitals = st.checkbox("Hospitales", value=True, key="chk_hosp")

all_municipalities = sorted(df_parcels["municipality"].unique())
selected_muns = st.sidebar.multiselect("Términos Municipales:", options=all_municipalities, default=all_municipalities, key="sel_muns")

active_df = df_parcels[df_parcels["municipality"].isin(selected_muns)].copy()

extra_mota = 0.85 if (ruptura_mota and q_peak_simulated > 300.0) else 0.0
active_df["active_depth"] = ((active_df["max_depth_m"] + extra_mota) * current_depth_factor).astype(np.float32)

active_df["peak_depth_experienced"] = ((active_df["max_depth_m"] + extra_mota) * peak_damage_factor_reached).astype(np.float32)
active_df["active_loss"] = (active_df["economic_loss_eur"] * min(1.6, peak_damage_factor_reached**1.35)).astype(np.float64)

is_currently_flooded = active_df["active_depth"] >= 0.25

active_df["dynamic_collapse"] = (
    (active_df["peak_depth_experienced"] >= 1.50) &
    ((active_df["hazard_factor_vh"] * peak_damage_factor_reached >= 1.2) | (active_df["dpm_p90"] >= 0.55))
)

active_df["dynamic_p1"] = (
    (is_currently_flooded & (
        (active_df["active_depth"] * active_df["max_velocity_ms"] * current_depth_factor >= 1.6) |
        (active_df["is_critical_infra"] & (active_df["active_depth"] >= 1.0))
    )) |
    active_df["dynamic_collapse"]
)

rain_mm = float(rain_val)
has_experienced_catastrophe = ("Forense" in sim_mode and peak_q_so_far >= 1200.0) or (rain_mm >= 180.0) or (q_peak_simulated >= 1200.0)

if has_experienced_catastrophe:
    if q_peak_simulated < 500.0 and "Forense" in sim_mode:
        badge_txt = "🔴 SIT. 2: CATASTRÓFICA (FASE RESCATE / RUINA)"
    elif rain_mm >= 180.0 and q_peak_simulated < 1200.0:
        badge_txt = "🔴 SIT. 2: ALERTA ROJA PREVENTIVA (CHIVA > 180 mm)"
    elif q_peak_simulated >= 1200.0 and rain_mm < 180.0:
        badge_txt = "🔴 SIT. 2: DESBORDAMIENTO RAMBLA DEL POYO"
    else:
        badge_txt = "🔴 SIT. 2: EMERGENCIA CATASTRÓFICA MÁXIMA"
    alert_badge_html = f"<span class='badge-alert-red'>{badge_txt}</span>"
    alert_state = "ROJO"

elif rain_mm >= 90.0 or q_peak_simulated >= 600.0:
    if rain_mm >= 90.0 and q_peak_simulated < 600.0:
        badge_txt = "🟠 SIT. 1: LLUVIA TORRENCIAL CABECERA"
    else:
        badge_txt = "🟠 SIT. 1: CRECIDA SEVERA EN CAUCE"
    alert_badge_html = f"<span class='badge-alert-orange'>{badge_txt}</span>"
    alert_state = "NARANJA"

elif rain_mm >= 40.0 or q_peak_simulated >= 250.0:
    alert_badge_html = "<span class='badge-alert-yellow'>🟡 PREALERTA POR LLUVIAS</span>"
    alert_state = "AMARILLO"

else:
    alert_badge_html = "<span class='badge-alert-green'>🟢 NORMALIDAD HIDROLÓGICA</span>"
    alert_state = "VERDE"

st.markdown(
    f"""
    <div class='hud-header'>
        <div style='display: flex; justify-content: space-between; align-items: center; width: 100%; flex-wrap: wrap; gap: 8px;'>
            <div style='flex: 1; min-width: 280px;'>
                <h1 style='margin:0; font-size: 1.45rem; letter-spacing: -0.02em;'>POYO-NOWCAST // GEMELO DIGITAL DE ALTA DEFINICIÓN</h1>
                <span style='color: #8b949e; font-size: 0.75rem;'>RAMBLA DEL POYO & HORTA SUD | FÍSICA NEURONAL FNO 2D Y TRANSFERENCIA DE RIESGO SOLVENCIA II</span>
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

if alert_state == "ROJO":
    st.markdown(
        """
        <div class='evac-banner-red'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <span style='font-size:1.5rem;'>🚨</span>
                <div>
                    <b style='color:#ff7b72; font-size:0.90rem;'>ORDEN GENERAL DE EVACUACIÓN VERTICAL — PROTECCIÓN CIVIL / CECOPI</b><br/>
                    <span style='color:#f0f6fc; font-size:0.78rem;'>PELIGRO EXTREMO POR DESBORDAMIENTO. Suba de inmediato a plantas altas. Prohibido circular por carretera o acceder a garajes/vados.</span>
                </div>
            </div>
            <div style='text-align:right; font-family: monospace; font-size:0.75rem; color:#ff7b72; font-weight:700;'>
                VENTANA DE ESCAPE: AGOTADA<br/><span style='color:#c9d1d9; font-weight:normal;'>Permanezca en pisos altos</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif alert_state == "NARANJA":
    st.markdown(
        """
        <div class='evac-banner-orange'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <span style='font-size:1.5rem;'>⚠️</span>
                <div>
                    <b style='color:#f0883e; font-size:0.90rem;'>PRE-ALERTA DE EVACUACIÓN: EVITE DESPLAZAMIENTOS Y RETIRE VEHÍCULOS</b><br/>
                    <span style='color:#f0f6fc; font-size:0.78rem;'>Onda de avenida aproximándose a l'Horta Sud. Asegure puntos altos y aléjese de puentes y ramblas.</span>
                </div>
            </div>
            <div style='text-align:right; font-family: monospace; font-size:0.75rem; color:#f0883e; font-weight:700;'>
                VENTANA DE SEGURIDAD:<br/><span style='color:#ffe3a8; font-weight:normal;'>&lt; 45 MINUTOS</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "Forense" in sim_mode:
    st.markdown(
        f"""
        <div class='telemetry-strip' style='border-left: 4px solid #da3633;'>
            <div>🏛️ <b>REGISTROS FORENSES 29-O:</b> AEMET / SAIH Hidrosur</div>
            <div>🌧️ <b>Lluvia Chiva:</b> <span style='color:#f85149; font-weight:700;'>{fmt_dec(rain_val, 1, ' mm')}</span></div>
            <div>🌊 <b>Caudal Rambla:</b> <span style='color:#58a6ff; font-weight:700;'>{fmt_int(q_peak_simulated, ' m³/s')}</span></div>
            <div>⏱️ <b>Minuto Hidrograma:</b> T + {sim_minute} min</div>
            <div>🔴 <span style='color:#f85149; font-weight:700;'>SERIE HISTÓRICA REPRODUCIDA</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif telemetry_active and live_obs:
    st.markdown(
        f"""
        <div class='telemetry-strip' style='border-left: 4px solid #238636;'>
            <div>📡 <b>TELEMETRÍA AEMET:</b> {live_obs['station_name']} ({live_obs['station_id']})</div>
            <div>🌧️ <b>Lluvia 1h:</b> <span style='color:#58a6ff;'>{fmt_dec(live_obs['rain_1h_mm'], 1, ' mm')}</span></div>
            <div>📈 <b>Acum. 4h:</b> <span style='color:#58a6ff;'>{fmt_dec(live_obs['rain_4h_mm'], 1, ' mm')}</span></div>
            <div>🌡️ <b>Temp:</b> {fmt_dec(live_obs['temp_c'], 1, ' °C')}</div>
            <div>⏱️ <b>Corte API:</b> {live_obs.get('timestamp_utc', 'Reciente')}</div>
            <div>🟢 <span style='color:#3fb950; font-weight:700;'>SENSOR EN LÍNEA (Latencia: &lt; 15 min)</span></div>
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
ins_att, ins_exh = 0.03, 0.12
q_eval_cat = peak_q_so_far if "Forense" in sim_mode else q_peak_simulated
collapse_ratio = total_collapsed / max(1, len(active_df))
fq = min(1.0, max(0.0, (q_eval_cat - q_att) / (q_exh - q_att)))
fi = min(1.0, max(0.0, (collapse_ratio - ins_att) / (ins_exh - ins_att)))
payout_rate = np.sqrt(fq * fi) * 100.0
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

# ==============================================================================
# PESTAÑAS (ORDEN OPERATIVO)
# ==============================================================================
tab_3d, tab_esalert, tab_hydro, tab_roads, tab_finances = st.tabs([
    "🌐 Gemelo Digital 3D (WebGPU)",
    "🚨 Despacho ES-Alert & Alertas C2",
    "🌊 Dinámica Hidráulica FNO (M2)",
    "🚑 Resiliencia Vial & TTI (M3)",
    "💼 Finanzas del Clima & Solvencia II (M4)",
])

# ------------------------------------------------------------------------------
# TAB 1: VISOR 3D DECK.GL (CENTROS SENSIBLES CON EVALUACIÓN DINÁMICA DE RIESGO)
# ------------------------------------------------------------------------------
with tab_3d:
    render_variable = st.radio(
        "Modo de Representación 3D en Gemelo Digital:",
        ["🌊 Calado Hidrodinámico FNO", "💶 Pérdida Económica CCS (€)", "🚨 Prioridad Triaje 112 (P1-P4)"],
        horizontal=True,
        key="render_var_horizontal"
    )

    if "Calado" in render_variable:
        legend_html = "<div class='legend-box'><span style='color:#8b949e; font-weight:700;'>CALADO:</span><div class='legend-item'><span class='legend-bullet' style='background:#da3633;'></span> Ruina InSAR DPM</div><div class='legend-item'><span class='legend-bullet' style='background:#f85149;'></span> &ge; 1,50 m (Catastrófico)</div><div class='legend-item'><span class='legend-bullet' style='background:#f0883e;'></span> 0,80 - 1,50 m (Severo)</div><div class='legend-item'><span class='legend-bullet' style='background:#d29922;'></span> 0,30 - 0,80 m (Flotabilidad)</div><div class='legend-item'><span class='legend-bullet' style='background:#2ea043;'></span> 0,05 - 0,30 m (Leve)</div><div class='legend-item'><span class='legend-bullet' style='background:#388bfd;'></span> &lt; 0,05 m (Seco)</div><div class='legend-item'><span class='legend-bullet' style='background:#238636;'></span> Vía Operativa</div><div class='legend-item'><span class='legend-bullet' style='background:#d73a49;'></span> Vía Cortada</div><div class='legend-item'><span class='legend-bullet' style='background:#e3a93b;'></span> Centro Sensible</div><div class='legend-item'><span class='legend-bullet' style='background:#58a6ff; border-radius:50%;'></span> Hospital</div></div>"
    elif "Pérdida" in render_variable:
        legend_html = "<div class='legend-box'><span style='color:#8b949e; font-weight:700;'>DAÑO ECONÓMICO:</span><div class='legend-item'><span class='legend-bullet' style='background:#9c27b0;'></span> &ge; 150.000 € (Ruina Económica)</div><div class='legend-item'><span class='legend-bullet' style='background:#e53935;'></span> 75.000 - 150.000 € (Daño Grave)</div><div class='legend-item'><span class='legend-bullet' style='background:#fb8c00;'></span> 30.000 - 75.000 € (Daño Medio)</div><div class='legend-item'><span class='legend-bullet' style='background:#fdd835;'></span> 10.000 - 30.000 € (Daño Bajo)</div><div class='legend-item'><span class='legend-bullet' style='background:#43a047;'></span> &lt; 10.000 € (Residual)</div><div class='legend-item'><span class='legend-bullet' style='background:#30363d;'></span> Sin Pérdidas</div><div class='legend-item'><span class='legend-bullet' style='background:#238636;'></span> Vía Operativa</div><div class='legend-item'><span class='legend-bullet' style='background:#d73a49;'></span> Vía Cortada</div></div>"
    else:
        legend_html = "<div class='legend-box'><span style='color:#8b949e; font-weight:700;'>TRIAJE 112:</span><div class='legend-item'><span class='legend-bullet' style='background:#da3633;'></span> P1 Crítica (Rescate Inmediato / Ruina)</div><div class='legend-item'><span class='legend-bullet' style='background:#f0883e;'></span> P2 Alta (Evacuación Necesaria)</div><div class='legend-item'><span class='legend-bullet' style='background:#d29922;'></span> P3 Moderada (Lámina Menor)</div><div class='legend-item'><span class='legend-bullet' style='background:#238636;'></span> P4 Normal (Sin Afección)</div><div class='legend-item'><span class='legend-bullet' style='background:#d73a49;'></span> Vía Inutilizada</div><div class='legend-item'><span class='legend-bullet' style='background:#58a6ff; border-radius:50%;'></span> Hospital</div></div>"

    st.markdown(legend_html, unsafe_allow_html=True)

    if "Calado" in render_variable:
        h_eval = active_df["peak_depth_experienced"].to_numpy() if "Forense" in sim_mode else active_df["active_depth"].to_numpy()
        c_eval = active_df["dynamic_collapse"].to_numpy()

        condlist = [
            c_eval,
            h_eval >= 1.50,
            h_eval >= 0.80,
            h_eval >= 0.30,
            h_eval >= 0.05,
        ]
        r_c = np.select(condlist, [218, 248, 240, 210, 46], default=40)
        g_c = np.select(condlist, [54,  81,  136, 153, 160], default=65)
        b_c = np.select(condlist, [51,  73,  62,  34,  67], default=95)
        a_c = np.select(condlist, [240, 220, 200, 180, 150], default=80)
        
        rgb_arr = np.column_stack([r_c, g_c, b_c, a_c]).astype(np.uint8)
        elevations = np.select(
            [c_eval, h_eval >= 0.30],
            [np.maximum(h_eval * 22.0, 40.0), h_eval * 20.0 + 4.0],
            default=np.clip(h_eval * 14.0 + 2.0, 2.0, 18.0)
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
        r_c = np.select(condlist_loss, [156, 229, 251, 253, 67], default=40)
        g_c = np.select(condlist_loss, [39,  57,  140, 216, 160], default=50)
        b_c = np.select(condlist_loss, [176, 53,  0,   53,  71], default=65)
        a_c = np.select(condlist_loss, [240, 220, 190, 170, 140], default=80)
        
        rgb_arr = np.column_stack([r_c, g_c, b_c, a_c]).astype(np.uint8)
        elevations = np.clip((l_col / 4200.0) + 4.0, 2.0, 120.0)

    else:
        p1_col = active_df["dynamic_p1"].to_numpy()
        h_tri = active_df["peak_depth_experienced"].to_numpy() if "Forense" in sim_mode else active_df["active_depth"].to_numpy()
        is_iso = active_df["is_isolated"].to_numpy()

        condlist_triage = [
            p1_col,
            (h_tri >= 0.80) | (is_iso & (h_tri >= 0.30)),
            (h_tri >= 0.30) | (active_df["active_loss"] >= 20000),
        ]
        r_c = np.select(condlist_triage, [218, 240, 210], default=35)
        g_c = np.select(condlist_triage, [54,  136, 153], default=134)
        b_c = np.select(condlist_triage, [51,  62,  34],  default=54)
        a_c = np.select(condlist_triage, [240, 210, 180], default=120)
        
        rgb_arr = np.column_stack([r_c, g_c, b_c, a_c]).astype(np.uint8)
        elevations = np.select([p1_col, (h_tri >= 0.80)], [50.0, 25.0], default=6.0)

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
            radius=12,
            get_fill_color="rgba",
            pickable=True,
            auto_highlight=True,
        )
    ]

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
                "width": 4 if edge_open else 8,
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
                "metric_primary": f"Capacidad / Censo: {v['beds']} personas",
                "metric_secondary": f"Calado Evaluado: {fmt_dec(h_local_vuln, 2, ' m')} ({v['type']})",
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
            {"name": "Hospital de Manises", "lon": -0.4608, "lat": 39.4930, "beds": 240},
        ]:
            if has_experienced_catastrophe:
                h_status = "ALERTA MÁXIMA (ACCESOS BLOQUEADOS / SATURACIÓN CATÁSTROFE)"
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
                get_radius=200,
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

    c_lat = float(active_df["lat"].mean()) if not active_df.empty else 39.4280
    c_lon = float(active_df["lon"].mean()) if not active_df.empty else -0.4150
    zoom_level = 13.4 if len(selected_muns) <= 2 else 12.6

    camera = pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=zoom_level, pitch=48, bearing=-18)

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

# ------------------------------------------------------------------------------
# TAB 2: DESPACHO ES-ALERT, RRSS & COMUNICADOS MODULADOS
# ------------------------------------------------------------------------------
with tab_esalert:
    st.subheader("Centro de Despacho ES-Alert & Resiliencia de Servicios Vitales")
    
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

    col_es1, col_es2 = st.columns([1.5, 1.5])

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
                f"Evacuación vertical INMEDIATA en Paiporta, Picanya, Sedaví y Catarroja. Suba a pisos altos. NO circule.\n"
                f"Info oficial: @GVA112 #DANAValencia"
            )
        elif alert_state == "NARANJA":
            tweet_text = (
                f"⚠️ AVISO 112 // ALERTA NARANJA RAMBLA DEL POYO\n"
                f"Nivel: EMERGENCIA SIT. 1 | Hora: {clock_badge_text}\n"
                f"Crecida severa propagándose ({fmt_int(q_peak_simulated, ' m3/s')}).\n"
                f"Aléjese de cauces y ramblas. Retire vehículos de zonas bajas y pasos subterráneos en l'Horta Sud.\n"
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
        st.markdown("#### Consola de Transmisión ES-Alert (Cell Broadcast)")
        
        if alert_state == "ROJO":
            urgency, severity = "Immediate", "Extreme"
            headline = "ALERTA ROJA PROTECCIÓN CIVIL: EMERGENCIA SITUACIÓN 2"
            body_es = f"EMERGENCIA SITUACIÓN 2. Peligro extremo por inundación en cuenca del Poyo. NO CIRCULE. Suba a pisos altos. Aléjese de cauces, pasos subterráneos y barrancos."
            body_val = f"EMERGÈNCIA SITUACIÓ 2. Perill extrem per inundació a la conca del Poio. NO CIRCULEU. Pugeu a pisos alts. Allunyeu-vos de lleres, passos subterranis i barrancs."
            cap_status, status_color = "Actual", "#da3633"
            banner_note = "DIFUSIÓN CELULAR FORZADA (ACTIVACIÓN ACÚSTICA EN SMARTPHONES)"
        elif alert_state == "NARANJA":
            urgency, severity = "Expected", "Severe"
            headline = "ALERTA NARANJA PROTECCIÓN CIVIL: EMERGENCIA SITUACIÓN 1"
            body_es = f"EMERGENCIA SITUACIÓN 1. Crecida severa en cuenca del Poyo ({fmt_int(q_peak_simulated, ' m3/s')}). Evite vados, ramblas y retire vehículos de cotas bajas."
            body_val = f"EMERGÈNCIA SITUACIÓ 1. Crecuda severa a la conca del Poio ({fmt_int(q_peak_simulated, ' m3/s')}). Eviteu guals, rambles i retireu vehicles de cotes baixes."
            cap_status, status_color = "Actual", "#f0883e"
            banner_note = "DIFUSIÓN REGIONAL SELECTIVA (AVISO OPERATIVO A POBLACIÓN EXPUESTA)"
        elif alert_state == "AMARILLO":
            urgency, severity = "Future", "Moderate"
            headline = "PREEMERGENCIA FASE ALERTA: PRECAUCIÓN POR LLUVIAS EN CABECERA"
            body_es = f"Precipitación intensa registrada ({fmt_dec(rain_mm, 1, ' mm')}). Caudal en cauce bajo seguimiento ({fmt_int(q_peak_simulated, ' m3/s')}). Precaución ordinaria."
            body_val = f"Precipitació intensa registrada ({fmt_dec(rain_mm, 1, ' mm')}). Cabal en curs sota seguiment ({fmt_int(q_peak_simulated, ' m3/s')}). Precaució ordinària."
            cap_status, status_color = "Test", "#d29922"
            banner_note = "CANAL INFORMATIVO CIUDADANO (SIN PITIDO DE ALARMA CELULAR)"
        else:
            urgency, severity = "Past", "Minor"
            headline = "SITUACIÓN NORMAL // SIN AVISOS ACTIVOS"
            body_es = "Caudales en niveles de estiaje y cuenca en parámetros de seguridad ordinaria."
            body_val = "Cabals en nivells d'estiatge i conca en paràmetres de seguretat ordinària."
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
                <div style='background: #161b22; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.82rem; color: #e6edf3; margin-bottom: 8px;'>
                    <b>[ES]</b> {body_es}
                </div>
                <div style='background: #161b22; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.82rem; color: #e6edf3;'>
                    <b>[VAL]</b> {body_val}
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
    <category>Met</category>
    <event>Flash Flood / Desbordamiento</event>
    <urgency>{urgency}</urgency>
    <severity>{severity}</severity>
    <certainty>Observed</certainty>
    <headline>{headline}</headline>
    <description>{body_es}</description>
    <area>
      <areaDesc>Horta Sud: Paiporta, Catarroja, Sedavi, Picanya, Massanassa</areaDesc>
      <circle>39.4230,-0.4180,8000</circle>
    </area>
  </info>
</alert>"""

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.download_button(
                label="📲 Exportar Payload CAP v1.2 (XML)",
                data=cap_xml_payload,
                file_name="es_alert_poyo_nowcast_payload.xml",
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
# TAB 3: DINÁMICA HIDRÁULICA FNO (MÓDULO 2)
# ------------------------------------------------------------------------------
with tab_hydro:
    col_h1, col_h2 = st.columns([1.6, 1.4])
    with col_h1:
        st.subheader("Hidrograma Transitorio de la Avenida (Rambla del Poyo)")
        t_steps = np.linspace(0, 360, 60)
        
        if "Forense" in sim_mode:
            q_envelope = np.interp(t_steps, t_arr, q_series) * q_mitig_factor
        else:
            q_envelope = q_peak_simulated * np.exp(-((t_steps - 210) ** 2) / (2 * 45 ** 2))
            q_envelope[:12] = np.linspace(min(50, q_peak_simulated * 0.1), min(250, q_peak_simulated * 0.3), 12)
        
        fig_hydro = go.Figure()
        fig_hydro.add_trace(go.Scatter(
            x=t_steps, y=q_envelope,
            mode='lines', line=dict(color='#58a6ff', width=3),
            name='Caudal FNO 2D', fill='tozeroy', fillcolor='rgba(31, 111, 235, 0.15)'
        ))
        fig_hydro.add_hline(y=1000.0, line_dash="dash", line_color="#d29922", annotation_text="Capacidad Cauce (1.000 m³/s)")
        fig_hydro.add_hline(y=1800.0, line_dash="dash", line_color="#da3633", annotation_text="Desbordamiento Catastrófico (1.800 m³/s)")
        
        if "Forense" in sim_mode:
            fig_hydro.add_vline(x=sim_minute, line_color="#ffffff", line_width=2, annotation_text=f"Instante: {exact_clock}")
        
        fig_hydro.update_layout(
            template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
            xaxis_title="Minutos transcurridos", yaxis_title="Caudal Líquido Q (m³/s)",
            margin=dict(l=40, r=40, t=30, b=40), height=340
        )
        st.plotly_chart(fig_hydro, width="stretch")

    with col_h2:
        st.subheader("Distribución de Calados por Término Municipal")
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
# TAB 4: RESILIENCIA VIAL & TTI (MÓDULO 3 CON HISTÉRESIS DE INCOMUNICACIÓN)
# ------------------------------------------------------------------------------
with tab_roads:
    st.subheader("Matriz Dinámica de Resiliencia Territorial & Time-to-Isolation (TTI)")
    
    eval_factor = peak_damage_factor_reached if "Forense" in sim_mode else current_depth_factor

    tti_metrics = (
        active_df.groupby("municipality")
        .agg(
            tti_med=("time_to_isolation_min", "median"),
            pct_isolated=("is_isolated", lambda s: float(np.mean(s)) * 100.0 if eval_factor > 0.35 else 0.0),
            parcels=("parcel_id", "count"),
            p1_urgente=("dynamic_p1", lambda s: int(np.sum(s)))
        )
        .reset_index()
    )
    
    tti_metrics["TTI_Label"] = tti_metrics.apply(
        lambda r: f"{int(round(r['tti_med']))} min" if pd.notnull(r['tti_med']) and np.isfinite(r['tti_med']) and r['pct_isolated'] > 0 else "Resiliente (> 120 min)",
        axis=1
    )
    
    def categorizar_alerta_vial(row):
        pct = row["pct_isolated"]
        tti = row["tti_med"]
        if pct >= 30.0 or (pct > 0 and pd.notnull(tti) and tti <= 25.0):
            return "🔴 AISLAMIENTO TOTAL (VÍAS CORTADAS)"
        elif pct >= 10.0 or (pct > 0 and pd.notnull(tti) and tti <= 45.0):
            return "🟠 RUTA EN RIESGO / ALERTA"
        return "🟢 CONECTIVIDAD ACTIVA"

    tti_metrics["Estado_Evacuacion"] = tti_metrics.apply(categorizar_alerta_vial, axis=1)
    tti_metrics["pct_isolated_fmt"] = tti_metrics["pct_isolated"].apply(lambda v: fmt_dec(v, 1, " %"))
    tti_metrics["p1_fmt"] = tti_metrics["p1_urgente"].apply(fmt_int)
    tti_metrics["Hospital_Destino"] = "H. Universitari i Politècnic La Fe"
    
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
# TAB 5: FINANZAS DEL CLIMA, CASCADA WATERFALL, ACTIVOS & SOLVENCIA II
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
        st.markdown("#### Cascada de Absorción (Waterfall)")
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
        fi_mesh = np.clip((I_mesh - 3.0) / 12.0, 0.0, 1.0)
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
        <div>POYO-NOWCAST: MÓDULO 5 GEMELO DIGITAL INTEGRADO | LICENCIA CC BY 4.0 OPEN SCIENCE</div>
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
