import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import joblib
import requests
from io import BytesIO
import cloudpickle

st.set_page_config(layout="wide")
st.title("🕊️ Proyección de Abundancia de Aves")

# === CONFIGURACIÓN ===
años = list(range(2025, 2035))
umbral = 1  # Abundancia mínima para mostrar en el mapa
variables_bio = {
    "Grallaria milleri": ['lat', 'lon','año','BIO1','BIO2','BIO4', 'BIO8', 'BIO9', 'BIO10', 'BIO15'],
    "Oxypogon stuebelii": ['lat', 'lon','año','BIO1', 'BIO2', 'BIO4', 'BIO12', 'BIO14', 'BIO15']
}


# === ENLACES ===
modelos_urls = {
    "Grallaria milleri": "https://raw.githubusercontent.com/brianpiragauta/UAM_MGDPS_Aves/main/Modelos/modelo_Grallaria_Milleri_Regressor_RandomForest.pkl",
    "Oxypogon stuebelii": "https://raw.githubusercontent.com/brianpiragauta/UAM_MGDPS_Aves/main/Modelos/modelo_Oxypogon_stuebelii_Regressor_RandomForest.plk"
}

base_csv_url = "https://raw.githubusercontent.com/brianpiragauta/UAM_MGDPS_Aves/main/DatosClimaticos/DatosRandomForest/Proyectadas/BIO_variables_{}.csv"
ruta_geojson = "https://raw.githubusercontent.com/brianpiragauta/UAM_MGDPS_Aves/main/DatosClimaticos/eje_cafetero_v2.geojson"

# === INTERFAZ ===
especie = st.selectbox("Selecciona la especie", list(modelos_urls.keys()))
anio = st.selectbox("Selecciona el año de predicción", años)

if st.button("Generar mapa"):
    with st.spinner("Cargando modelo y datos..."):
        # Descargar modelo
        modelo_url = modelos_urls[especie]
        response = requests.get(modelo_url)
        modelo = cloudpickle.load(BytesIO(response.content))


        # Descargar CSV
        url_csv = base_csv_url.format(anio)
        df = pd.read_csv(url_csv)
        df['año'] = anio

        # Predecir abundancia
        X = df[variables_bio[especie]]
        df["abundancia_predicha"] = modelo.predict(X)

        # Filtrar por umbral
        df_altas = df[df["abundancia_predicha"] >= umbral]

        # Crear mapa base
        mapa = folium.Map(location=[5.1, -75.5], zoom_start=8, tiles='CartoDB positron')

        colores_departamentos = {
            "CALDAS": "#ff9999",
            "RISARALDA": "#99ccff",
            "QUINDIO": "#99ff99",
            "TOLIMA": "#FFFACD"
        }

        geojson_layer = folium.GeoJson(
            ruta_geojson,
            name="Departamentos",
            style_function=lambda feature: {
                "fillColor": colores_departamentos.get(feature["properties"]["NOMBRE_DPT"].upper(), "#cccccc"),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.1
            },
            tooltip=folium.GeoJsonTooltip(fields=["NOMBRE_DPT"], aliases=["Departamento:"])
        )
        geojson_layer.add_to(mapa)

        for _, row in df_altas.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=max(3, row["abundancia_predicha"] * 3),
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.4,
                popup=folium.Popup(f"<b>Abundancia estimada:</b> {row['abundancia_predicha']:.2f}", max_width=150),
                tooltip=f"{row['abundancia_predicha']:.2f}"
            ).add_to(mapa)

        explicacion = f"""
        <b> Mapa de abundancia proyectada - <i>{especie}</i> ({anio})</b><br><br>
        Puntos con abundancia predicha mayor o igual a {umbral}.
        """

        folium.Marker(
            location=[6.1, -75.5],
            icon=folium.Icon(color="purple", icon="info-sign"),
            popup=folium.Popup(folium.Html(explicacion, script=True), max_width=450)
        ).add_to(mapa)

        leyenda_html = f"""
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            width: 260px;
            background-color: white;
            border:2px solid gray;
            z-index:9999;
            font-size:14px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <b> Mapa de abundancia proyectada - <i>{especie}</i> ({anio})</b><br>
        </div>
        """
        mapa.get_root().html.add_child(folium.Element(leyenda_html))
        folium_static(mapa)

    st.success("¡Mapa generado exitosamente!")
