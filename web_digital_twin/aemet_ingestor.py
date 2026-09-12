"""
POYO-NOWCAST: Ingestor Meteorológico Oficial AEMET OpenData
Conexión en tiempo real con estaciones automáticas de la cuenca del Poyo.
Gestión desacoplada de credenciales (Streamlit Secrets / Env / Fallback).
Autor: Kelvin Jesus Flores Yarihuaman
Licencia: Open Science (CC BY 4.0)
"""

from typing import Dict, Any, Optional
import os
import requests


class AEMETRealTimeClient:
    BASE_URL = "https://opendata.aemet.es/opendata/api"

    # Estaciones automáticas estratégicas en la cuenca del Poyo
    ESTACIONES_RED = [
        {"id": "8368U", "nombre": "Chiva (Cabecera Rambla)"},
        {"id": "8373X", "nombre": "Turís (Cuenca Media)"},
        {"id": "8414A", "nombre": "Valencia Aeropuerto (Manises)"},
        {"id": "8416Y", "nombre": "Valencia Viveros (Litoral)"},
    ]

    # Clave de respaldo (dejar vacía en GitHub; se lee de st.secrets en local)
FALLBACK_KEY = ""

    )

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
            return provided_key

        # 1. Intento de lectura desde .streamlit/secrets.toml
        try:
            import streamlit as st
            if "AEMET_API_KEY" in st.secrets:
                return st.secrets["AEMET_API_KEY"]
        except Exception:
            pass

        # 2. Intento de lectura desde variable de entorno del sistema
        env_key = os.getenv("AEMET_API_KEY")
        if env_key:
            return env_key

        # 3. Token por defecto
        return self.FALLBACK_KEY

    def _execute_two_step_query(self, endpoint: str) -> Any:
        """Ejecuta el protocolo de redirección de dos pasos de AEMET OpenData."""
        url = f"{self.BASE_URL}{endpoint}"

        # Paso 1: Petición de URL de descarga
        resp = requests.get(url, headers=self.headers, timeout=10)
        
        if resp.status_code == 401:
            raise PermissionError("La API Key de AEMET ha expirado o es inválida (HTTP 401).")
        if resp.status_code != 200:
            raise ConnectionError(f"Error AEMET Paso 1 (Status {resp.status_code}): {resp.text}")

        meta = resp.json()
        estado = meta.get("estado")
        if estado != 200:
            desc = meta.get("descripcion", "Error desconocido")
            if estado == 401 or "autorizado" in desc.lower():
                raise PermissionError(f"API Key no autorizada o caducada: {desc}")
            raise ValueError(f"Respuesta no exitosa de AEMET (Estado {estado}): {desc}")

        data_url = meta.get("datos")
        if not data_url:
            raise ValueError("No se obtuvo la URL temporal de datos de AEMET.")

        # Paso 2: Descarga del payload real
        data_resp = requests.get(data_url, timeout=15)
        if data_resp.status_code != 200:
            raise ConnectionError(f"Error AEMET Paso 2 (Status {data_resp.status_code})")

        return data_resp.json()

    def get_basin_live_rainfall(self) -> Dict[str, Any]:
        """
        Consulta las estaciones de la cuenca en orden de prioridad geográfica.
        Extrae la precipitación horaria y el acumulado móvil estimado de 4 horas.
        """
        last_error = "SIN_DATOS"

        for est in self.ESTACIONES_RED:
            endpoint = f"/observacion/convencional/datos/estacion/{est['id']}"
            try:
                records = self._execute_two_step_query(endpoint)
                if not records:
                    continue

                # Acumulación de las últimas 4 horas reportadas
                recent_precips = []
                for r in records[-4:]:
                    p = r.get("prec")
                    if p is not None:
                        try:
                            recent_precips.append(float(p))
                        except (ValueError, TypeError):
                            pass

                latest_r = records[-1]
                rain_1h = float(latest_r.get("prec") or 0.0)
                temp = float(latest_r.get("ta") or 18.0)
                timestamp_utc = latest_r.get("fint", "Reciente")

                accum_4h = sum(recent_precips) if recent_precips else (rain_1h * 3.5)

                return {
                    "station_id": est["id"],
                    "station_name": est["nombre"],
                    "timestamp_utc": timestamp_utc,
                    "rain_1h_mm": rain_1h,
                    "rain_4h_mm": max(rain_1h, accum_4h),
                    "temp_c": temp,
                    "status": "OPERATIVO",
                }

            except PermissionError as pe:
                # Si la clave caducó, detener el bucle inmediatamente para reportar la incidencia
                return {
                    "station_id": est["id"],
                    "station_name": est["nombre"],
                    "timestamp_utc": "Clave caducada",
                    "rain_1h_mm": 0.0,
                    "rain_4h_mm": 0.0,
                    "temp_c": 20.0,
                    "status": f"TOKEN_CADUCADO: {str(pe)}",
                }
            except Exception as e:
                last_error = str(e)
                continue

        # Degradación controlada si los servidores no responden o no hay datos
        return {
            "station_id": "8368U",
            "station_name": "Chiva (Cabecera Rambla)",
            "timestamp_utc": "Sin conexión",
            "rain_1h_mm": 0.0,
            "rain_4h_mm": 0.0,
            "temp_c": 20.0,
            "status": f"MODO_SIMULACION ({last_error})",
        }


if __name__ == "__main__":
    client = AEMETRealTimeClient()
    obs = client.get_basin_live_rainfall()
    print("--- DIAGNÓSTICO DEL INGESTOR AEMET ---")
    print(f"Estación Activa : {obs['station_name']} ({obs['station_id']})")
    print(f"Timestamp UTC   : {obs['timestamp_utc']}")
    print(f"Lluvia (1 hora) : {obs['rain_1h_mm']} mm")
    print(f"Acumulado (4 h) : {obs['rain_4h_mm']:.1f} mm")
    print(f"Estado Sensor   : {obs['status']}")
