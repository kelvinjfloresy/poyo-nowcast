"""
POYO-NOWCAST: Módulo 5 - Gemelo Digital Hidrodinámico, Territorial y Actuarial 3D
Tecnología: Streamlit + PyDeck (Deck.gl WebGPU) + Plotly C2 HUD + PyProj Geodésico.
Integración: AEMET OpenData API + FNO 2D + Resiliencia Red Vial + Solvencia II.
Autor: Kelvin Jesus Flores Yarihuaman
Licencia: Open Science (CC BY 4.0)
"""

import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
# CONFIGURACIÓN DEL ENTORNO Y ESTILOS HUD C2 (CENTRO DE CONTROL)
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
    
    /* Cabecera HUD Táctica */
    .hud-header {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
    }

    /* Tarjeta AEMET Destacada en Sidebar */
    .aemet-card {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.22) 0%, rgba(35, 134, 54, 0.22) 100%);
        border: 1px solid #388bfd;
        border-left: 5px solid #2ea043;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
        box-shadow: 0 0 14px rgba(56, 139, 253, 0.2);
    }

    /* Banner de Telemetría AEMET en Directo */
    .telemetry-strip {
        background: rgba(13, 17, 23, 0.92);
        border: 1px solid #30363d;
        border-left: 4px solid #238636;
        border-radius: 6px;
        padding: 10px 18px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* Leyenda Táctica del Visor 3D */
    .legend-box {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 12px;
        display: flex;
        gap: 20px;
        align-items: center;
        flex-wrap: wrap;
        font-size: 0.80rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .legend-bullet {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        display: inline-block;
    }
    
    /* Métricas C2 */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.70rem !important;
        font-weight: 700;
        color: #f0f6fc;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        font-weight: 600;
    }
    .stMetric {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(14px);
        padding: 14px 18px;
        border-radius: 8px;
        border: 1px solid #30363d;
        border-left: 4px solid #1f6feb;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    
    /* Badges de Alerta Dinámicos */
    .badge-alert-green {
        background: rgba(35, 134, 54, 0.2);
        color: #3fb950;
        border: 1px solid #238636;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-alert-yellow {
        background: rgba(210, 153, 34, 0.2);
        color: #d29922;
        border: 1px solid #bb8009;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-alert-orange {
        background: rgba(219, 109, 40, 0.25);
        color: #f0883e;
        border: 1px solid #bd561d;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-alert-red {
        background: #da3633;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 0 12px rgba(218, 54, 51, 0.6);
    }
    
    .deckgl-container {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #30363d;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
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
    formatted = fmt.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{formatted}{suffix}"

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

    def transform_geojson_coords(self, coords_list):
        if self.transformer:
            xs = [c[0] for c in coords_list]
            ys = [c[1] for c in coords_list]
            lons, lats = self.transformer.transform(xs, ys)
            return [[float(lo), float(la)] for lo, la in zip(lons, lats)]
        return [
            [-0.4180 + (c[0] - self.ref_x) / 85000.0, 39.4230 + (c[1] - self.ref_y) / 111000.0]
            for c in coords_list
        ]

PROJECTOR = GeoProjector()

@st.cache_data
def load_all_system_artifacts():
    parquet_path = "data/processed/flood_damage_matrix.parquet"
    geojson_path = "data/processed/horta_sud_road_network_status.geojson"
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
        
        triage = np.full(n, "P4_BAJA", dtype=object)
        triage[(depth >= 0.30) | (ratio >= 0.20)] = "P3_MODERADA"
        triage[(depth >= 0.80) | (depth * vel >= 0.50) | isolated] = "P2_ALTA"
        triage[collapse | (depth * vel >= 1.5) | (critical & isolated)] = "P1_CRITICA"

        grade = np.where(dpm >= 0.6, "COLLAPSE", np.where(dpm >= 0.4, "SEVERE", np.where(dpm >= 0.2, "MODERATE", "NONE")))

        df = pd.DataFrame({
            "parcel_id": [f"46{np.random.randint(100, 999)}A{i:05d}" for i in range(n)],
            "municipality": muns,
            "x_coord": x, "y_coord": y,
            "max_depth_m": depth, "max_velocity_ms": vel,
            "hazard_factor_vh": depth * vel,
            "dpm_p90": dpm, "structural_collapse": collapse,
            "damage_grade": grade, "asset_value_eur": assets,
            "damage_ratio": ratio, "economic_loss_eur": losses,
            "time_to_isolation_min": tti, "is_isolated": isolated,
            "is_critical_infra": critical, "triage_priority": triage,
            "target_hospital": "HOSPITAL_LA_FE"
        })
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression="snappy")
    else:
        df = pq.read_table(parquet_path).to_pandas()
        if "municipality" in df.columns:
            df["municipality"] = df["municipality"].astype(str)

    df["lon"], df["lat"] = PROJECTOR.transform_points(df["x_coord"].to_numpy(), df["y_coord"].to_numpy())

    network_data = None
    if os.path.exists(geojson_path):
        with open(geojson_path, "r", encoding="utf-8") as f:
            network_data = json.load(f)
        for feat in network_data.get("features", []):
            geom = feat["geometry"]
            if geom["type"] == "LineString":
                geom["coordinates"] = PROJECTOR.transform_geojson_coords(geom["coordinates"])
            elif geom["type"] == "Point":
                geom["coordinates"] = PROJECTOR.transform_geojson_coords([geom["coordinates"]])[0]

    qrt_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    return df, network_data, qrt_df


df_parcels, network_geojson, qrt_summary = load_all_system_artifacts()

# ==============================================================================
# BARRA LATERAL: PANEL DE CONTROL Y CONFIGURACIÓN AEMET
# ==============================================================================
st.sidebar.markdown(
    """
    <div style='padding: 12px; background: rgba(31, 111, 235, 0.12); border-left: 4px solid #1f6feb; border-radius: 6px; margin-bottom: 16px;'>
        <b style='color: #58a6ff; font-size: 0.95rem;'>POYO-NOWCAST HUD</b><br/>
        <span style='color: #8b949e; font-size: 0.78rem;'>Mando Operativo & Transferencia de Riesgos</span>
    </div>
    """,
    unsafe_allow_html=True,
)

sim_mode = st.sidebar.radio(
    "Modo de Operación:",
    ["Modo Hindcast (Forense 29-O 2024)", "Modo Nowcast Predictivo (Tiempo Real)"],
)

st.sidebar.markdown("---")

live_obs = None
telemetry_active = False

if sim_mode == "Modo Hindcast (Forense 29-O 2024)":
    st.sidebar.subheader("⏱️ Progresión Temporal de Avenida")
    sim_minute = st.sidebar.slider(
        "Minuto del Evento (Base 16:00 h = 0 min):",
        min_value=0, max_value=360, value=210, step=15, format="%d min"
    )
    hour_label = f"{16 + sim_minute // 60:02d}:{sim_minute % 60:02d} h"
    progress = min(1.0, sim_minute / 240.0)
    current_depth_factor = progress
    q_peak_simulated = 1950.0 * progress
    factor_clima = 1.0
else:
    st.sidebar.subheader("⚡ Nowcast & Conexión AEMET")
    
    # Destacado especial del Conector AEMET
    st.sidebar.markdown("<div class='aemet-card'>", unsafe_allow_html=True)
    conectar_aemet = st.sidebar.toggle("📡 Telemetría AEMET en Vivo", value=True)
    st.sidebar.markdown(
        "<span style='font-size:0.75rem; color:#c9d1d9;'>Conexión automática a red SAIH/AEMET de la cuenca (Chiva, Turís, Manises).</span></div>",
        unsafe_allow_html=True
    )
    
    horizonte_clima = st.sidebar.selectbox(
        "Horizonte Climático (IPCC / EIOPA):",
        ["Actual (2024-2026)", "Horizonte 2030 (SSP2-4.5 / +8% Q)", "Horizonte 2050 (SSP5-8.5 / +22% Q)"],
    )
    factor_clima = 1.0 if "Actual" in horizonte_clima else (1.08 if "2030" in horizonte_clima else 1.22)
    
    lead_time_min = st.sidebar.slider("Avance Temporal (Lead Time):", 15, 180, 60, step=15, format="T + %d min")

    # Ingesta meteorológica
    rain_real = 0.0
    if conectar_aemet and HAS_AEMET:
        aemet_client = AEMETRealTimeClient()
        live_obs = aemet_client.get_basin_live_rainfall()
        telemetry_active = True
        rain_real = live_obs["rain_4h_mm"]
        
        # Detección de AMC basada en lluvia
        amc_auto = "Seco (AMC I)" if rain_real < 10.0 else ("Normal (AMC II)" if rain_real < 35.0 else "Saturado (AMC III)")
        amc_mode = st.sidebar.selectbox("Humedad Antecedente (AMC):", [f"Auto AEMET: {amc_auto}", "Manual: Seco (AMC I)", "Manual: Normal (AMC II)", "Manual: Saturado (AMC III)"])
        
        if "Seco" in amc_mode:
            amc_weight = 0.75
        elif "Normal" in amc_mode:
            amc_weight = 1.0
        else:
            amc_weight = 1.25
        
        rain_mm = st.sidebar.slider(
            "Lluvia Cabecera (Sensor en vivo + Adicional):",
            min_value=float(rain_real), max_value=600.0, value=float(rain_real), step=10.0,
            help="Si hoy no llueve (0 mm), sube este control para simular una avenida convectiva sobre el sensor real."
        )
    else:
        rain_mm = st.sidebar.slider("Precipitación Cabecera Chiva (mm / 4h):", 0, 650, 0, step=10)
        soil_amc = st.sidebar.select_slider("Humedad Antecedente (AMC):", options=["Seco (AMC I)", "Normal (AMC II)", "Saturado (AMC III)"], value="Normal (AMC II)")
        amc_weight = 1.25 if soil_amc == "Saturado (AMC III)" else (1.0 if soil_amc == "Normal (AMC II)" else 0.75)

    propagation_factor = min(1.0, lead_time_min / 90.0)
    hour_label = f"T + {lead_time_min} min (Proyección FNO)"

    current_depth_factor = (rain_mm / 450.0) * amc_weight * factor_clima * propagation_factor
    q_peak_simulated = min(3200.0, 1950.0 * (rain_mm / 490.0) * amc_weight * factor_clima * propagation_factor)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Modos de Visualización 3D")
render_variable = st.sidebar.selectbox(
    "Variable Activa de Extrusión:",
    ["Calado Hidrodinámico FNO (m)", "Pérdida Económica CCS (€)", "Prioridad Triaje 112 (P1-P4)"],
)

show_roads = st.sidebar.checkbox("Mostrar Red Viaria y Cortes", value=True)
show_hospitals = st.sidebar.checkbox("Mostrar Centros Hospitalarios", value=True)

all_municipalities = sorted(df_parcels["municipality"].unique())
selected_muns = st.sidebar.multiselect("Términos Municipales:", options=all_municipalities, default=all_municipalities)

# Filtrado y escalado dinámico
active_df = df_parcels[df_parcels["municipality"].isin(selected_muns)].copy()
active_df["active_depth"] = (active_df["max_depth_m"] * current_depth_factor).astype(np.float32)
active_df["active_loss"] = (active_df["economic_loss_eur"] * min(1.6, current_depth_factor**1.35)).astype(np.float64)

# Triaje dinámico: si no hay agua, no hay rescates
is_flooded = active_df["active_depth"] >= 0.25
active_df["dynamic_collapse"] = active_df["structural_collapse"] & (active_df["active_depth"] >= 0.40)
active_df["dynamic_p1"] = is_flooded & (
    active_df["dynamic_collapse"] | 
    (active_df["active_depth"] * active_df["max_velocity_ms"] >= 1.5) | 
    (active_df["is_critical_infra"] & (active_df["active_depth"] >= 0.80))
)

# ==============================================================================
# LÓGICA DE ALERTA ANTICIPADA MULTIVARIABLE (METEOROLOGÍA + HIDROLOGÍA)
# ==============================================================================
# Mapeo seguro del valor de precipitación de cabecera
rain_mm = locals().get('rain_val', locals().get('rainfall_mm', locals().get('precip_total', locals().get('add_rain', 0.0))))
# Umbrales AEMET Meteoalerta (Chiva) y Plan Especial de Inundaciones GVA:
# 1. Alerta Roja: Lluvia >= 180 mm en 4h O Caudal Rambla >= 1.200 m³/s
# 2. Alerta Naranja: Lluvia >= 90 mm en 4h O Caudal Rambla >= 600 m³/s
# 3. Alerta Amarilla: Lluvia >= 40 mm en 4h O Caudal Rambla >= 250 m³/s

if rain_mm >= 180.0 or q_peak_simulated >= 1200.0:
    if rain_mm >= 180.0 and q_peak_simulated < 1200.0:
        badge_txt = "NIVEL ROJO // ALERTA PREVENTIVA METEOROLÓGICA (CABECERA > 180 mm)"
    elif q_peak_simulated >= 1200.0 and rain_mm < 180.0:
        badge_txt = "NIVEL ROJO // EMERGENCIA HIDRÁULICA (DESBORDAMIENTO RAMBLA)"
    else:
        badge_txt = "NIVEL ROJO // EMERGENCIA EXTREMA CO-DEPENDIENTE"
    alert_badge_html = f"<span class='badge-alert-red'>{badge_txt}</span>"

elif rain_mm >= 90.0 or q_peak_simulated >= 600.0:
    if rain_mm >= 90.0 and q_peak_simulated < 600.0:
        badge_txt = "NIVEL NARANJA // PRECIPITACIÓN SEVERA EN CABECERA"
    else:
        badge_txt = "NIVEL NARANJA // CRECIDA SIGNIFICATIVA EN CAUCE"
    alert_badge_html = f"<span class='badge-alert-orange'>{badge_txt}</span>"

elif rain_mm >= 40.0 or q_peak_simulated >= 250.0:
    alert_badge_html = "<span class='badge-alert-yellow'>NIVEL AMARILLO // PREEMERGENCIA POR LLUVIAS</span>"

else:
    alert_badge_html = "<span class='badge-alert-green'>NIVEL VERDE // SITUACIÓN NORMAL</span>"

# ==============================================================================
# CABECERA PRINCIPAL Y TELEMETRÍA EN PANTALLA
# ==============================================================================
st.markdown(
    f"""
    <div class='hud-header'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h1 style='margin:0; font-size: 1.85rem; letter-spacing: -0.02em;'>POYO-NOWCAST // GEMELO DIGITAL DE ALTA DEFINICIÓN</h1>
                <span style='color: #8b949e; font-size: 0.82rem;'>RAMBLA DEL POYO & HORTA SUD | FÍSICA NEURONAL FNO 2D Y TRANSFERENCIA DE RIESGO SOLVENCIA II</span>
            </div>
            <div style='text-align: right;'>
                {alert_badge_html}<br/>
                <span style='font-family: "JetBrains Mono"; color: #58a6ff; font-weight:700; font-size: 0.85rem; margin-top: 4px; display: inline-block;'>HORA REF: {hour_label}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if telemetry_active and live_obs:
    st.markdown(
        f"""
        <div class='telemetry-strip'>
            <div>📡 <b>TELEMETRÍA AEMET EN TIEMPO REAL:</b> Estación {live_obs['station_name']} ({live_obs['station_id']})</div>
            <div>🌧️ <b>Lluvia (1 hora):</b> <span style='color:#58a6ff;'>{fmt_dec(live_obs['rain_1h_mm'], 1, ' mm')}</span></div>
            <div>📈 <b>Acumulado (4 h):</b> <span style='color:#58a6ff;'>{fmt_dec(live_obs['rain_4h_mm'], 1, ' mm')}</span></div>
            <div>🌡️ <b>Temperatura:</b> {fmt_dec(live_obs['temp_c'], 1, ' °C')}</div>
            <div>⏱️ <b>UTC:</b> {live_obs['timestamp_utc']}</div>
            <div>🟢 <span style='color:#3fb950; font-weight:700;'>SENSOR EN LÍNEA</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# KPIs OPERATIVOS TÁCTICOS (DELTAS CON SEMÁFORO DE SEGURIDAD CORREGIDO)
# ==============================================================================
total_exposure_m = active_df["asset_value_eur"].sum() / 1e6
current_loss_m = active_df["active_loss"].sum() / 1e6
total_collapsed = int(active_df["dynamic_collapse"].sum())
critical_p1 = int(active_df["dynamic_p1"].sum())

q_att, q_exh = 1000.0, 1800.0
ins_att, ins_exh = 0.03, 0.12
collapse_ratio = total_collapsed / max(1, len(active_df))
fq = min(1.0, max(0.0, (q_peak_simulated - q_att) / (q_exh - q_att)))
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
        delta_color="inverse"  # Si diff_q > 0 (peligro) se pinta ROJO; si es negativo (seguro) se pinta VERDE
    )
with kpi2:
    pct_exp = (current_loss_m / max(0.1, total_exposure_m)) * 100
    st.metric(
        "Pérdida Directa Activa",
        fmt_dec(current_loss_m, 1, " M€"),
        delta=f"{fmt_dec(pct_exp, 1, '%')} de Exposición",
        delta_color="inverse" if pct_exp > 0 else "off"  # Pérdida > 0 se marca en rojo
    )
with kpi3:
    st.metric("Inmuebles en Ruina", fmt_int(total_collapsed), delta="InSAR DPM ≥ 0,40", delta_color="inverse")
with kpi4:
    st.metric("Prioridad P1 (Rescate 112)", fmt_int(critical_p1), delta="Evacuación Inmediata", delta_color="inverse")
with kpi5:
    st.metric(
        "Gatillo Paramétrico (Cat Bond)",
        fmt_dec(payout_rate, 1, "%"),
        delta=f"{fmt_dec(total_payout_m, 1, ' M€')} Liberados < 48h",
        delta_color="inverse" if payout_rate > 50.0 else "normal"
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ==============================================================================
# CUERPO PRINCIPAL MULTIPANEL
# ==============================================================================
tab_3d, tab_hydro, tab_roads, tab_finances = st.tabs([
    "🌐 Gemelo Digital 3D (WebGPU)",
    "🌊 Dinámica Hidráulica FNO (M2)",
    "🚑 Resiliencia Vial & TTI (M3)",
    "💼 Finanzas del Clima & Solvencia II (M4)",
])

# ------------------------------------------------------------------------------
# TAB 1: VISOR 3D DECK.GL (CON LEYENDA TÁCTICA Y ESTADOS DINÁMICOS)
# ------------------------------------------------------------------------------
with tab_3d:
    # Leyenda explicativa interactiva
    st.markdown(
        """
        <div class='legend-box'>
            <span style='color:#8b949e; font-weight:700;'>LEYENDA OPERATIVA C2:</span>
            <div class='legend-item'><span class='legend-bullet' style='background:#388bfd;'></span> Edificación Urbana (Catastro LOD1)</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#da3633;'></span> Ruina / Colapso InSAR DPM</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#238636;'></span> Vía Operativa / Evacuación</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#d73a49;'></span> Vía Cortada (Calado o Intradós)</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#58a6ff; border-radius:50%;'></span> Hospital Terciario</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h_vals = active_df["active_depth"].to_numpy()
    c_vals = active_df["dynamic_collapse"].to_numpy()
    l_vals = active_df["active_loss"].to_numpy()
    p1_vals = active_df["dynamic_p1"].to_numpy()

    if render_variable == "Calado Hidrodinámico FNO (m)":
        conds = [c_vals, h_vals >= 1.5, h_vals >= 0.5, h_vals >= 0.1]
        palette = [
            [218, 54, 51, 230],
            [248, 81, 73, 210],
            [210, 153, 34, 190],
            [56, 139, 253, 170],
        ]
        fallback_c = [31, 111, 235, 120]
        extrusions = h_vals * 20.0 + 4.0
    elif render_variable == "Pérdida Económica CCS (€)":
        conds = [l_vals >= 150000, l_vals >= 75000, l_vals >= 25000, l_vals > 0]
        palette = [
            [189, 0, 38, 230],
            [240, 59, 32, 200],
            [254, 178, 76, 180],
            [255, 237, 160, 150],
        ]
        fallback_c = [50, 50, 50, 80]
        extrusions = (l_vals / 4500.0) + 4.0
    else:
        conds = [p1_vals, h_vals >= 0.8, h_vals >= 0.3]
        palette = [
            [218, 54, 51, 240],   # P1 Rojo
            [210, 153, 34, 200],  # P2 Naranja
            [227, 179, 65, 170],  # P3 Amarillo
        ]
        fallback_c = [46, 160, 67, 120] # P4 Normal
        extrusions = np.where(p1_vals, 50.0, np.where(h_vals >= 0.3, 15.0, 4.0))

    rgb_arr = np.full((len(active_df), 4), fallback_c, dtype=np.uint8)
    for c, col in zip(conds, palette):
        rgb_arr[c] = col

    active_df["rgba"] = rgb_arr.tolist()
    active_df["elevation_m"] = np.clip(extrusions, 2.0, 120.0)
    active_df["layer_title"] = active_df["parcel_id"] + " (" + active_df["municipality"] + ")"
    active_df["metric_primary"] = "Calado: " + active_df["active_depth"].apply(lambda v: fmt_dec(v, 2, " m"))
    active_df["metric_secondary"] = "Pérdida CCS: " + active_df["active_loss"].apply(lambda v: fmt_dec(v, 0, " €"))
    active_df["status_tag"] = np.where(p1_vals, "P1_CRITICA", np.where(h_vals >= 0.8, "P2_ALTA", np.where(h_vals >= 0.3, "P3_MODERADA", "P4_NORMAL")))

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

    # Red viaria: transitabilidad dinámica en función de si hay agua
    if show_roads and network_geojson:
        road_paths = []
        for feat in network_geojson.get("features", []):
            if feat["geometry"]["type"] == "LineString":
                p = feat["properties"]
                # Si el río está seco, todas las vías están transitables (excepto ruina InSAR si la hubiera)
                if current_depth_factor < 0.08:
                    edge_open = True
                    reason = "OPERATIVO (SECO)"
                else:
                    edge_open = p.get("is_active", True)
                    reason = p.get("failure_reason", "CORTADO")

                col = [46, 160, 67, 210] if edge_open else [218, 54, 51, 250]
                road_paths.append({
                    "path": feat["geometry"]["coordinates"],
                    "layer_title": p.get("name", "Vía de Comunicación"),
                    "metric_primary": f"Longitud: {fmt_int(p.get('length_m', 0.0), ' m')}",
                    "metric_secondary": f"Calado en Eje: {fmt_dec(p.get('h_water_m', 0.0) * current_depth_factor, 2, ' m')}",
                    "status_tag": "OPERATIVO" if edge_open else reason,
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

    # Hospitales: estado y color sincronizados dinámicamente con el caudal
    if show_hospitals:
        hosp_data = []
        for h in [
            {"name": "H. Universitari i Politècnic La Fe", "lon": -0.3768, "lat": 39.4435, "beds": 1000},
            {"name": "H. General Universitari de València", "lon": -0.4072, "lat": 39.4682, "beds": 550},
            {"name": "Hospital de Manises", "lon": -0.4608, "lat": 39.4930, "beds": 240},
        ]:
            if q_peak_simulated < 300.0:
                h_status = "OPERATIVO (NORMAL)"
                h_col = [56, 139, 253, 230]
            elif q_peak_simulated < 1200.0:
                h_status = "PREALERTA SANITARIA"
                h_col = [210, 153, 34, 230]
            else:
                h_status = "ALERTA MÁXIMA / SATURACIÓN"
                h_col = [218, 54, 51, 240]

            hosp_data.append({
                "name": h["name"],
                "lon": h["lon"],
                "lat": h["lat"],
                "layer_title": h["name"],
                "metric_primary": f"Capacidad: {h['beds']} camas",
                "metric_secondary": "Rol: Hospital Terciario",
                "status_tag": h_status,
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

    camera = pdk.ViewState(
        latitude=c_lat,
        longitude=c_lon,
        zoom=zoom_level,
        pitch=48,
        bearing=-18,
    )

    st.markdown("<div class='deckgl-container'>", unsafe_allow_html=True)
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
                        "<b>Estado:</b> <span style='color:#f85149; font-weight:700;'>{status_tag}</span>"
                        "</div>",
                "style": {"backgroundColor": "#161b22", "color": "white", "fontSize": "12px", "borderRadius": "4px"},
            },
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: DINÁMICA HIDRÁULICA FNO (MÓDULO 2)
# ------------------------------------------------------------------------------
with tab_hydro:
    col_h1, col_h2 = st.columns([1.6, 1.4])
    
    with col_h1:
        st.subheader("Hidrograma Transitorio de la Avenida (Rambla del Poyo)")
        t_steps = np.linspace(0, 360, 60)
        q_envelope = q_peak_simulated * np.exp(-((t_steps - 210) ** 2) / (2 * 45 ** 2))
        q_envelope[:12] = np.linspace(min(50, q_peak_simulated * 0.1), min(250, q_peak_simulated * 0.3), 12)
        
        fig_hydro = go.Figure()
        fig_hydro.add_trace(go.Scatter(
            x=t_steps, y=q_envelope,
            mode='lines',
            line=dict(color='#58a6ff', width=3),
            name='Caudal FNO 2D',
            fill='tozeroy',
            fillcolor='rgba(31, 111, 235, 0.15)'
        ))
        fig_hydro.add_hline(y=1000.0, line_dash="dash", line_color="#d29922", annotation_text="Capacidad Cauce (1.000 m³/s)")
        fig_hydro.add_hline(y=1800.0, line_dash="dash", line_color="#da3633", annotation_text="Desbordamiento Catastrófico (1.800 m³/s)")
        
        if sim_mode == "Modo Hindcast (Forense 29-O 2024)":
            fig_hydro.add_vline(x=sim_minute, line_color="#ffffff", line_width=2, annotation_text=f"T = {hour_label}")
            
        fig_hydro.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            xaxis_title="Minutos transcurridos (16:00 h = 0 min)",
            yaxis_title="Caudal Líquido Q (m³/s)",
            margin=dict(l=40, r=40, t=30, b=40),
            height=340
        )
        st.plotly_chart(fig_hydro, use_container_width=True)

    with col_h2:
        st.subheader("Distribución de Calados por Término Municipal")
        fig_box = px.box(
            active_df,
            x="municipality", y="active_depth",
            color="municipality",
            labels={"active_depth": "Calado h (m)", "municipality": "Municipio"},
        )
        fig_box.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=40),
            height=340
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: RESILIENCIA VIAL & TTI (MÓDULO 3)
# ------------------------------------------------------------------------------
with tab_roads:
    st.subheader("Matriz Dinámica de Resiliencia Territorial & Time-to-Isolation (TTI)")
    
    tti_metrics = (
        active_df.groupby("municipality")
        .agg(
            tti_med=("time_to_isolation_min", "median"),
            pct_isolated=("is_isolated", lambda s: float(np.mean(s)) * 100.0 if current_depth_factor > 0.08 else 0.0),
            parcels=("parcel_id", "count"),
            p1_urgente=("dynamic_p1", lambda s: int(np.sum(s)))
        )
        .reset_index()
    )
    
    tti_metrics["TTI_Label"] = tti_metrics.apply(
        lambda r: f"{int(round(r['tti_med']))} min" if pd.notnull(r['tti_med']) and np.isfinite(r['tti_med']) and r['pct_isolated'] > 0 else "Resiliente (> 120 min)",
        axis=1
    )
    
    def categorizar_alerta(row):
        pct = row["pct_isolated"]
        tti = row["tti_med"]
        if pct >= 30.0 or (pct > 0 and pd.notnull(tti) and tti <= 25.0):
            return "🔴 AISLAMIENTO TOTAL"
        elif pct >= 10.0 or (pct > 0 and pd.notnull(tti) and tti <= 45.0):
            return "🟠 RUTA EN RIESGO / ALERTA"
        return "🟢 CONECTIVIDAD ACTIVA"

    tti_metrics["Estado_Evacuacion"] = tti_metrics.apply(categorizar_alerta, axis=1)
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
        use_container_width=True,
        hide_index=True
    )
    
    if network_geojson:
        st.download_button(
            label="📥 Descargar Capa de Red Viaria Activa (GeoJSON para CECOPI / 112)",
            data=json.dumps(network_geojson, indent=2),
            file_name="poyo_road_network_emergency.geojson",
            mime="application/geo+json"
        )

# ------------------------------------------------------------------------------
# TAB 4: FINANZAS DEL CLIMA, SOLVENCIA II Y PROYECCIÓN 2024-2050 (M4)
# ------------------------------------------------------------------------------
with tab_finances:
    st.subheader("Modelado Catastrófico, Reaseguro y Seguros Paramétricos (Solvencia II / EIOPA)")
    
    scr_current = max(0.0, current_loss_m * 1.35 - (current_loss_m * 0.042))
    coc_current = 0.06 * scr_current
    
    # 1. Fila Superior: Balance QRT + Cascada
    f_col1, f_col2 = st.columns([1.2, 1.8])
    
    with f_col1:
        st.markdown("#### Balance Regulatorio (EIOPA ORSA)")
        solv_table = pd.DataFrame([
            {"Parámetro Actuarial": "Exposición Bruta (TIV)", "Importe": fmt_dec(total_exposure_m, 1, " M€")},
            {"Parámetro Actuarial": "Pérdida Bruta Modelada (Ground-Up)", "Importe": fmt_dec(current_loss_m, 1, " M€")},
            {"Parámetro Actuarial": "Pérdida Anual Esperada (AAL)", "Importe": fmt_dec(current_loss_m * 0.042, 2, " M€/año")},
            {"Parámetro Actuarial": "SCR Solvencia II (VaR 99,5%)", "Importe": fmt_dec(scr_current, 1, " M€")},
            {"Parámetro Actuarial": "Margen de Riesgo (CoC 6%)", "Importe": fmt_dec(coc_current, 2, " M€")},
            {"Parámetro Actuarial": "Indemnización Paramétrica Activa", "Importe": fmt_dec(total_payout_m, 2, " M€")},
        ])
        st.dataframe(solv_table, use_container_width=True, hide_index=True)
        
    with f_col2:
        st.markdown("#### Cascada de Financiación de la Catástrofe")
        val_ccs = current_loss_m * 0.72
        val_param = min(current_loss_m * 0.28, total_payout_m)
        val_gap = max(0.0, current_loss_m - (val_ccs + val_param))
        
        fig_waterfall = go.Figure(go.Bar(
            x=["Consorcio (CCS)", "Cat Bond Paramétrico", "Brecha No Cubierta", "Pérdida Total"],
            y=[val_ccs, val_param, val_gap, current_loss_m],
            marker=dict(
                color=["#238636", "#1f6feb", "#d29922", "#da3633"],
                line=dict(color="#30363d", width=1.5)
            ),
            text=[fmt_dec(v, 1, " M€") for v in [val_ccs, val_param, val_gap, current_loss_m]],
            textposition="auto"
        ))
        fig_waterfall.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            yaxis_title="Capital Movilizado (M€)",
            margin=dict(l=40, r=40, t=30, b=40),
            height=280
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. Fila Media: Curva EP y Matriz 2D Dual-Trigger
    f_col3, f_col4 = st.columns(2)

    with f_col3:
        st.markdown("#### Curva EP con Reaseguro Exceso de Pérdida (XoL)")
        T_periods = np.array([2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500])
        loss_base_curve = total_exposure_m * (1.0 - np.exp(-0.38 * (T_periods / 100.0)**0.45))
        loss_stressed_curve = loss_base_curve * 1.18
        
        attach_xol, limit_xol = 60.0, 120.0
        ceded_xol = np.clip(loss_base_curve - attach_xol, 0.0, limit_xol)
        net_retained = loss_base_curve - ceded_xol

        fig_ep = go.Figure()
        fig_ep.add_trace(go.Scatter(
            x=T_periods, y=loss_base_curve,
            mode='lines+markers', name='Base Bruta (Ground-Up)',
            line=dict(color='#388bfd', width=2.5)
        ))
        fig_ep.add_trace(go.Scatter(
            x=T_periods, y=loss_stressed_curve,
            mode='lines', name='Estrés Climático (ORSA +18%)',
            line=dict(color='#f85149', width=2, dash='dash')
        ))
        fig_ep.add_trace(go.Scatter(
            x=T_periods, y=net_retained,
            mode='lines+markers', name='Retención Neta (Post-XoL)',
            line=dict(color='#238636', width=2.5)
        ))
        fig_ep.add_vline(x=200, line_dash="dot", line_color="#ffffff", annotation_text="SCR (T=200 años)")

        fig_ep.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            xaxis_title="Periodo de Retorno T (Años)",
            yaxis_title="Pérdida Modelada (M€)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            height=340
        )
        st.plotly_chart(fig_ep, use_container_width=True)

    with f_col4:
        st.markdown("#### Matriz de Disparo Paramétrico (Dual-Trigger Surface)")
        
        q_grid = np.linspace(800, 2000, 40)
        insar_grid = np.linspace(0.0, 32.0, 40)
        Q_mesh, I_mesh = np.meshgrid(q_grid, insar_grid)

        fq_mesh = np.clip((Q_mesh - 1000.0) / 800.0, 0.0, 1.0)
        fi_mesh = np.clip((I_mesh - 3.0) / 12.0, 0.0, 1.0)
        payout_surface = np.sqrt(fq_mesh * fi_mesh) * 100.0

        fig_matrix = go.Figure()
        fig_matrix.add_trace(go.Contour(
            z=payout_surface, x=q_grid, y=insar_grid,
            colorscale="Viridis", reversescale=True,
            colorbar=dict(title="Payout %", len=0.8),
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white"))
        ))
        
        current_insar_pct = min(30.0, collapse_ratio * 100.0)
        fig_matrix.add_trace(go.Scatter(
            x=[min(2000.0, max(800.0, q_peak_simulated))], y=[current_insar_pct],
            mode='markers+text',
            marker=dict(color='#ff0055', size=14, symbol='diamond', line=dict(color='white', width=2)),
            text=[f"ACTIVO: {fmt_dec(payout_rate, 1, '%')}"],
            textposition="top center",
            name='Instante Activo'
        ))

        fig_matrix.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            xaxis_title="Caudal Punta FNO (m³/s)",
            yaxis_title="Colapso Físico InSAR (%)",
            yaxis=dict(range=[0, 32]),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=340
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # 3. Proyección Decenal del Riesgo Climático
    st.markdown("#### Proyección Decenal del Riesgo Climático & Coste de Solvencia (2024 - 2050)")
    
    decadas = np.array([2024, 2030, 2035, 2040, 2045, 2050])
    factor_ssp2 = 1.0 + 0.0045 * (decadas - 2024)
    factor_ssp5 = 1.0 + 0.0090 * (decadas - 2024)
    
    aal_ssp2 = (current_loss_m * 0.042) * factor_ssp2
    aal_ssp5 = (current_loss_m * 0.042) * factor_ssp5
    scr_ssp5 = scr_current * factor_ssp5

    fig_future = go.Figure()
    fig_future.add_trace(go.Bar(
        x=decadas, y=scr_ssp5, name="Capital de Solvencia Requerido (SCR SSP5-8.5)",
        marker=dict(color="rgba(31, 111, 235, 0.25)", line=dict(color="#388bfd", width=1.5)),
        yaxis="y2"
    ))
    fig_future.add_trace(go.Scatter(
        x=decadas, y=aal_ssp2, mode="lines+markers",
        name="AAL (Senda Intermedia SSP2-4.5)",
        line=dict(color="#2da44e", width=2.5)
    ))
    fig_future.add_trace(go.Scatter(
        x=decadas, y=aal_ssp5, mode="lines+markers",
        name="AAL (Senda Pesimista SSP5-8.5)",
        line=dict(color="#cf222e", width=2.5, dash="dash")
    ))

    fig_future.update_layout(
        template="plotly_dark",
        plot_bgcolor="#161b22",
        paper_bgcolor="#0d1117",
        xaxis_title="Año de Proyección",
        yaxis=dict(title="Pérdida Anual Esperada AAL (M€/año)"),
        yaxis2=dict(title="Requisito de Capital SCR (M€)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=320
    )
    st.plotly_chart(fig_future, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; color: #8b949e; font-size: 0.78rem;'>
        <div>POYO-NOWCAST: MÓDULO 5 GEMELO DIGITAL INTEGRADO | LICENCIA CC BY 4.0 OPEN SCIENCE</div>
        <div>AUTOR: KELVIN JESUS FLORES YARIHUAMAN</div>
    </div>
    """,
    unsafe_allow_html=True,
)
