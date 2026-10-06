# ============================================================
# EARTHQUAKE INSIGHT
# Earthquake Intelligence & Analytics Dashboard
# ============================================================

import os
from datetime import date, time, datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sqlalchemy import create_engine, text


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Earthquake Insight",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_magnitude.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "model_features.pkl"
)


# ============================================================
# 3. DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_database_engine():

    try:

        database_url = os.getenv("DATABASE_URL")

        if database_url:

            engine = create_engine(
                database_url,
                pool_pre_ping=True
            )

        else:

            engine = create_engine(
                "mysql+pymysql://root:root@127.0.0.1:3306/earthquake_db",
                pool_pre_ping=True
            )

        with engine.connect() as connection:

            connection.execute(
                text("SELECT 1")
            )

        return engine

    except Exception as e:

        st.error(
            "❌ Koneksi ke database gagal."
        )

        st.code(
            str(e)
        )

        st.info(
            "Pastikan MySQL/Docker sedang berjalan "
            "dan database earthquake_db tersedia."
        )

        st.stop()


engine = get_database_engine()


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_data():

    query = """
        SELECT
            usgs_id,
            event_time,
            magnitude,
            place,
            depth_km,
            longitude,
            latitude,
            mag_type,
            status,
            tsunami,
            emsc_id,
            emsc_magnitude,
            distance_km,
            magnitude_diff,
            event_date,
            weather_lat,
            weather_lon,
            temperature_2m_mean,
            precipitation_sum,
            wind_speed_10m_max,
            surface_pressure_mean
        FROM earthquake_data
        WHERE magnitude IS NOT NULL
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
    """

    data = pd.read_sql(
        query,
        engine
    )

    if data.empty:

        st.error(
            "Tabel earthquake_data tidak memiliki data."
        )

        st.stop()

    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    data["event_time"] = pd.to_datetime(
        data["event_time"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "magnitude",
        "depth_km",
        "longitude",
        "latitude",
        "tsunami",
        "emsc_magnitude",
        "distance_km",
        "magnitude_diff",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    data["year"] = data["event_time"].dt.year
    data["month"] = data["event_time"].dt.month
    data["day"] = data["event_time"].dt.day
    data["hour"] = data["event_time"].dt.hour
    data["day_of_week"] = data["event_time"].dt.dayofweek

    # --------------------------------------------------------
    # Magnitude category
    # --------------------------------------------------------

    data["magnitude_category"] = pd.cut(
        data["magnitude"],
        bins=[
            -np.inf,
            3,
            4,
            5,
            6,
            np.inf
        ],
        labels=[
            "Sangat Kecil",
            "Kecil",
            "Sedang",
            "Kuat",
            "Besar"
        ]
    )

    return data


df = load_data()


# ============================================================
# 5. LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None

    try:

        return joblib.load(
            MODEL_PATH
        )

    except Exception:

        return None


@st.cache_resource
def load_model_features():

    if not os.path.exists(FEATURE_PATH):

        return None

    try:

        return joblib.load(
            FEATURE_PATH
        )

    except Exception:

        return None


model = load_model()

model_features = load_model_features()


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🌎 Earthquake Insight"
    )

    st.caption(
        "Earthquake Intelligence & Analytics Dashboard"
    )

    st.divider()

    st.subheader(
        "Navigation"
    )

    page = st.radio(
        "Pilih halaman:",
        [
            "Overview",
            "Spatial Intelligence",
            "Seismic Analytics",
            "Environmental Analytics",
            "Magnitude Prediction",
            "Insights",
            "Data Explorer"
        ]
    )

    st.divider()

    st.subheader(
        "Dataset"
    )

    st.metric(
        "Total Events",
        f"{len(df):,}"
    )

    st.caption(
        "USGS + EMSC + Open-Meteo"
    )

    st.divider()

    if model is not None:

        st.success(
            "Model tersedia"
        )

    else:

        st.warning(
            "Model tidak ditemukan"
        )


# ============================================================
# 7. HEADER
# ============================================================

st.title(
    "🌎 Earthquake Insight"
)

st.subheader(
    "Earthquake Intelligence & Analytics Dashboard"
)

st.caption(
    "Analisis dan Prediksi Magnitudo Gempa Bumi "
    "Berbasis Integrasi Data Seismik dan Cuaca"
)

st.divider()


# ============================================================
# 8. OVERVIEW
# ============================================================

if page == "Overview":

    st.header(
        "📊 Overview"
    )

    st.write(
        "Halaman ini menyajikan ringkasan statistik "
        "dari dataset gempa hasil integrasi data "
        "USGS, EMSC, dan Open-Meteo."
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_events = len(df)

    avg_magnitude = df["magnitude"].mean()

    max_magnitude = df["magnitude"].max()

    avg_depth = df["depth_km"].mean()

    tsunami_total = int(
        df["tsunami"]
        .fillna(0)
        .sum()
    )

    unique_locations = df["place"].nunique()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🌋 Total Gempa",
            f"{total_events:,}"
        )

    with col2:

        st.metric(
            "📈 Rata-rata Magnitudo",
            f"{avg_magnitude:.2f}"
        )

    with col3:

        st.metric(
            "🔺 Magnitudo Maksimum",
            f"{max_magnitude:.2f}"
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "📍 Rata-rata Kedalaman",
            f"{avg_depth:.2f} km"
        )

    with col5:

        st.metric(
            "🌊 Indikasi Tsunami",
            f"{tsunami_total:,}"
        )

    with col6:

        st.metric(
            "📌 Lokasi",
            f"{unique_locations:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # MONTHLY DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        monthly = (
            df.groupby("month")
            .size()
            .reset_index(
                name="jumlah"
            )
        )

        month_names = {
            1: "Januari",
            2: "Februari",
            3: "Maret",
            4: "April",
            5: "Mei",
            6: "Juni",
            7: "Juli",
            8: "Agustus",
            9: "September",
            10: "Oktober",
            11: "November",
            12: "Desember"
        }

        monthly["bulan"] = (
            monthly["month"]
            .map(month_names)
        )

        fig = px.bar(
            monthly,
            x="bulan",
            y="jumlah",
            title="Jumlah Gempa Berdasarkan Bulan",
            labels={
                "bulan": "Bulan",
                "jumlah": "Jumlah Gempa"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="magnitude",
            nbins=35,
            title="Distribusi Magnitudo",
            labels={
                "magnitude": "Magnitudo",
                "count": "Jumlah"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # YEARLY TREND
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        yearly = (
            df.groupby("year")
            .size()
            .reset_index(
                name="jumlah"
            )
        )

        fig = px.line(
            yearly,
            x="year",
            y="jumlah",
            markers=True,
            title="Tren Jumlah Gempa per Tahun",
            labels={
                "year": "Tahun",
                "jumlah": "Jumlah Gempa"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        category = (
            df["magnitude_category"]
            .value_counts()
            .reset_index()
        )

        category.columns = [
            "kategori",
            "jumlah"
        ]

        fig = px.pie(
            category,
            names="kategori",
            values="jumlah",
            hole=0.45,
            title="Komposisi Kategori Magnitudo"
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 9. SPATIAL INTELLIGENCE
# ============================================================

elif page == "Spatial Intelligence":

    st.header(
        "🗺️ Spatial Intelligence"
    )

    st.write(
        "Eksplorasi persebaran lokasi gempa berdasarkan "
        "koordinat geografis, magnitudo, dan kedalaman."
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    map_data = df[
        [
            "latitude",
            "longitude",
            "magnitude",
            "depth_km",
            "place",
            "event_time"
        ]
    ].dropna()

    # Batasi data untuk menjaga performa browser
    if len(map_data) > 8000:

        map_data = map_data.sample(
            8000,
            random_state=42
        )

    fig = px.scatter_map(
        map_data,
        lat="latitude",
        lon="longitude",
        color="magnitude",
        size="magnitude",
        hover_name="place",
        hover_data={
            "latitude": ":.3f",
            "longitude": ":.3f",
            "magnitude": ":.2f",
            "depth_km": ":.2f",
            "event_time": True
        },
        zoom=1,
        height=650,
        color_continuous_scale="Turbo",
        title="Persebaran Gempa Bumi"
    )

    fig.update_layout(
        map_style="open-street-map",
        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Geographic analysis
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    sample = df.sample(
        min(len(df), 7000),
        random_state=100
    )

    with col1:

        fig = px.scatter(
            sample,
            x="longitude",
            y="latitude",
            color="magnitude",
            size="magnitude",
            title="Distribusi Magnitudo Secara Spasial",
            labels={
                "longitude": "Longitude",
                "latitude": "Latitude",
                "magnitude": "Magnitudo"
            },
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            sample,
            x="longitude",
            y="latitude",
            color="depth_km",
            size="magnitude",
            title="Distribusi Kedalaman Secara Spasial",
            labels={
                "longitude": "Longitude",
                "latitude": "Latitude",
                "depth_km": "Kedalaman (km)"
            },
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 10. SEISMIC ANALYTICS
# ============================================================

elif page == "Seismic Analytics":

    st.header(
        "📈 Seismic Analytics"
    )

    st.write(
        "Analisis karakteristik aktivitas gempa berdasarkan "
        "magnitudo, kedalaman, waktu, dan jenis magnitudo."
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="depth_km",
            nbins=40,
            title="Distribusi Kedalaman Gempa",
            labels={
                "depth_km": "Kedalaman (km)"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        sample = df.sample(
            min(len(df), 7000),
            random_state=200
        )

        fig = px.scatter(
            sample,
            x="depth_km",
            y="magnitude",
            color="magnitude",
            opacity=0.6,
            title="Hubungan Kedalaman dan Magnitudo",
            labels={
                "depth_km": "Kedalaman (km)",
                "magnitude": "Magnitudo"
            },
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        yearly_magnitude = (
            df.groupby("year")["magnitude"]
            .agg(
                rata_rata="mean",
                maksimum="max"
            )
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=yearly_magnitude["year"],
                y=yearly_magnitude["rata_rata"],
                mode="lines+markers",
                name="Rata-rata"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=yearly_magnitude["year"],
                y=yearly_magnitude["maksimum"],
                mode="lines+markers",
                name="Maksimum"
            )
        )

        fig.update_layout(
            title="Magnitudo Rata-rata dan Maksimum per Tahun",
            xaxis_title="Tahun",
            yaxis_title="Magnitudo",
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        mag_type = (
            df["mag_type"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .reset_index()
        )

        mag_type.columns = [
            "jenis",
            "jumlah"
        ]

        fig = px.bar(
            mag_type,
            x="jenis",
            y="jumlah",
            title="Jenis Magnitudo",
            labels={
                "jenis": "Magnitude Type",
                "jumlah": "Jumlah"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Statistik Deskriptif"
    )

    statistical_columns = [
        "magnitude",
        "depth_km",
        "distance_km",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean"
    ]

    statistics = (
        df[statistical_columns]
        .describe()
        .T
        .round(3)
    )

    st.dataframe(
        statistics,
        use_container_width=True
    )


# ============================================================
# 11. ENVIRONMENTAL ANALYTICS
# ============================================================

elif page == "Environmental Analytics":

    st.header(
        "🌦️ Environmental Analytics"
    )

    st.write(
        "Analisis kondisi cuaca yang terintegrasi dengan "
        "data kejadian gempa."
    )

    sample = df.sample(
        min(len(df), 7000),
        random_state=300
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            sample,
            x="temperature_2m_mean",
            y="magnitude",
            color="magnitude",
            opacity=0.6,
            title="Temperatur dan Magnitudo",
            labels={
                "temperature_2m_mean": "Temperatur",
                "magnitude": "Magnitudo"
            },
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            sample,
            x="precipitation_sum",
            y="magnitude",
            color="magnitude",
            opacity=0.6,
            title="Curah Hujan dan Magnitudo",
            labels={
                "precipitation_sum": "Curah Hujan",
                "magnitude": "Magnitudo"
            },
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            sample,
            x="wind_speed_10m_max",
            y="magnitude",
            color="magnitude",
            opacity=0.6,
            title="Kecepatan Angin dan Magnitudo",
            labels={
                "wind_speed_10m_max": "Kecepatan Angin",
                "magnitude": "Magnitudo"
            },
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            sample,
            x="surface_pressure_mean",
            y="magnitude",
            color="magnitude",
            opacity=0.6,
            title="Tekanan Permukaan dan Magnitudo",
            labels={
                "surface_pressure_mean": "Tekanan Permukaan",
                "magnitude": "Magnitudo"
            },
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Matriks Korelasi"
    )

    correlation_columns = [
        "magnitude",
        "depth_km",
        "distance_km",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean"
    ]

    correlation = (
        df[correlation_columns]
        .corr()
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Korelasi Variabel"
    )

    fig.update_layout(
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 12. MAGNITUDE PREDICTION
# ============================================================

elif page == "Magnitude Prediction":

    st.header(
        "🔮 Magnitude Prediction"
    )

    st.write(
        "Gunakan karakteristik kejadian untuk memperoleh "
        "estimasi nilai magnitudo menggunakan Random Forest."
    )

    st.info(
        "Catatan: model ini mengestimasi nilai magnitudo "
        "berdasarkan karakteristik yang dimasukkan. Model "
        "bukan alat untuk memprediksi kapan gempa akan terjadi."
    )

    if model is None:

        st.error(
            "Model Random Forest tidak ditemukan."
        )

        st.code(
            MODEL_PATH
        )

    elif model_features is None:

        st.error(
            "File model_features.pkl tidak ditemukan."
        )

        st.code(
            FEATURE_PATH
        )

    else:

        st.subheader(
            "Input Karakteristik Gempa"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=-4.0,
                step=0.1
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=122.5,
                step=0.1
            )

            depth = st.number_input(
                "Depth (km)",
                min_value=0.0,
                value=10.0,
                step=1.0
            )

        with col2:

            distance = st.number_input(
                "Distance (km)",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

            temperature = st.number_input(
                "Temperature",
                value=25.0,
                step=0.5
            )

            precipitation = st.number_input(
                "Rainfall",
                min_value=0.0,
                value=0.0,
                step=0.5
            )

        with col3:

            wind_speed = st.number_input(
                "Wind Speed",
                min_value=0.0,
                value=10.0,
                step=0.5
            )

            pressure = st.number_input(
                "Surface Pressure",
                min_value=0.0,
                value=1010.0,
                step=1.0
            )

        st.subheader(
            "Waktu Kejadian"
        )

        col1, col2 = st.columns(2)

        with col1:

            event_date = st.date_input(
                "Tanggal",
                value=date.today()
            )

        with col2:

            event_time = st.time_input(
                "Waktu",
                value=time(12, 0)
            )

        st.divider()

        predict = st.button(
            "🔮 PREDICT MAGNITUDE",
            type="primary",
            use_container_width=True
        )

        if predict:

            event_datetime = datetime.combine(
                event_date,
                event_time
            )

            # ------------------------------------------------
            # Prepare input
            # ------------------------------------------------

            input_data = {
                "depth_km": depth,
                "latitude": latitude,
                "longitude": longitude,
                "distance_km": distance,
                "temperature_2m_mean": temperature,
                "precipitation_sum": precipitation,
                "wind_speed_10m_max": wind_speed,
                "surface_pressure_mean": pressure,
                "year": event_datetime.year,
                "month": event_datetime.month,
                "day": event_datetime.day,
                "hour": event_datetime.hour,
                "day_of_week": event_datetime.weekday()
            }

            input_df = pd.DataFrame(
                [input_data]
            )

            # ------------------------------------------------
            # Check model features
            # ------------------------------------------------

            missing_features = [
                feature
                for feature in model_features
                if feature not in input_df.columns
            ]

            if missing_features:

                st.error(
                    "Fitur berikut dibutuhkan model tetapi "
                    "belum tersedia:"
                )

                st.write(
                    missing_features
                )

            else:

                try:

                    # Pastikan urutan fitur
                    # sama persis dengan training

                    input_df = input_df[
                        model_features
                    ]

                    prediction = float(
                        model.predict(
                            input_df
                        )[0]
                    )

                    # ------------------------------------------------
                    # Magnitude category
                    # ------------------------------------------------

                    if prediction < 3:

                        category = "Sangat Kecil"

                    elif prediction < 4:

                        category = "Kecil"

                    elif prediction < 5:

                        category = "Sedang"

                    elif prediction < 6:

                        category = "Kuat"

                    else:

                        category = "Besar"

                    # ------------------------------------------------
                    # Result
                    # ------------------------------------------------

                    st.success(
                        "Prediksi berhasil dilakukan."
                    )

                    st.subheader(
                        "Hasil Prediksi"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Predicted Magnitude",
                            f"{prediction:.2f}"
                        )

                    with col2:

                        st.metric(
                            "Kategori",
                            category
                        )

                    # ------------------------------------------------
                    # Model performance
                    # ------------------------------------------------

                    st.divider()

                    st.subheader(
                        "Performa Model"
                    )

                    m1, m2, m3 = st.columns(3)

                    with m1:

                        st.metric(
                            "MAE",
                            "0.2254"
                        )

                    with m2:

                        st.metric(
                            "RMSE",
                            "0.3241"
                        )

                    with m3:

                        st.metric(
                            "R²",
                            "0.3513"
                        )

                    st.info(
                        "Interpretasi: model memiliki R² sebesar "
                        "0.3513 pada data pengujian. Nilai prediksi "
                        "merupakan estimasi berdasarkan pola data "
                        "historis dan tidak menjamin terjadinya "
                        "gempa pada masa mendatang."
                    )

                    # ------------------------------------------------
                    # Input summary
                    # ------------------------------------------------

                    with st.expander(
                        "Lihat detail input prediksi"
                    ):

                        display_input = input_df.copy()

                        st.dataframe(
                            display_input,
                            use_container_width=True
                        )

                except Exception as e:

                    st.error(
                        "Prediksi gagal."
                    )

                    st.code(
                        str(e)
                    )


# ============================================================
# 13. INSIGHTS
# ============================================================

elif page == "Insights":

    st.header(
        "💡 Key Insights"
    )

    st.write(
        "Ringkasan temuan utama berdasarkan hasil "
        "eksplorasi dataset."
    )

    # --------------------------------------------------------
    # Maximum magnitude
    # --------------------------------------------------------

    max_row = df.loc[
        df["magnitude"].idxmax()
    ]

    # --------------------------------------------------------
    # Minimum depth
    # --------------------------------------------------------

    min_depth_row = df.loc[
        df["depth_km"].idxmin()
    ]

    # --------------------------------------------------------
    # Maximum depth
    # --------------------------------------------------------

    max_depth_row = df.loc[
        df["depth_km"].idxmax()
    ]

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation_columns = [
        "magnitude",
        "depth_km",
        "distance_km",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean"
    ]

    correlations = (
        df[correlation_columns]
        .corr()["magnitude"]
        .drop("magnitude")
        .sort_values(
            key=lambda x: abs(x),
            ascending=False
        )
    )

    strongest_feature = correlations.index[0]

    strongest_correlation = correlations.iloc[0]

    # --------------------------------------------------------
    # Insight 1
    # --------------------------------------------------------

    st.subheader(
        "1. Magnitudo maksimum"
    )

    st.write(
        f"Magnitudo tertinggi pada dataset adalah "
        f"**{max_row['magnitude']:.2f}**."
    )

    if pd.notna(max_row["place"]):

        st.write(
            f"Lokasi kejadian tercatat sebagai "
            f"**{max_row['place']}**."
        )

    # --------------------------------------------------------
    # Insight 2
    # --------------------------------------------------------

    st.subheader(
        "2. Kedalaman gempa"
    )

    st.write(
        f"Gempa terdangkal memiliki kedalaman sekitar "
        f"**{min_depth_row['depth_km']:.2f} km**, "
        f"sedangkan gempa terdalam mencapai sekitar "
        f"**{max_depth_row['depth_km']:.2f} km**."
    )

    # --------------------------------------------------------
    # Insight 3
    # --------------------------------------------------------

    st.subheader(
        "3. Variabel yang paling berkorelasi"
    )

    st.write(
        f"Variabel dengan hubungan linear paling kuat "
        f"terhadap magnitudo dalam dataset adalah "
        f"**{strongest_feature}** dengan koefisien korelasi "
        f"**{strongest_correlation:.3f}**."
    )

    # --------------------------------------------------------
    # Correlation chart
    # --------------------------------------------------------

    corr_display = correlations.reset_index()

    corr_display.columns = [
        "Variabel",
        "Korelasi"
    ]

    fig = px.bar(
        corr_display,
        x="Korelasi",
        y="Variabel",
        orientation="h",
        title="Korelasi Variabel terhadap Magnitudo"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Statistical summary
    # --------------------------------------------------------

    st.subheader(
        "4. Ringkasan Statistik"
    )

    summary = pd.DataFrame({
        "Indikator": [
            "Jumlah data",
            "Rata-rata magnitudo",
            "Magnitudo maksimum",
            "Rata-rata kedalaman",
            "Kedalaman maksimum"
        ],
        "Nilai": [
            f"{len(df):,}",
            f"{df['magnitude'].mean():.2f}",
            f"{df['magnitude'].max():.2f}",
            f"{df['depth_km'].mean():.2f} km",
            f"{df['depth_km'].max():.2f} km"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 14. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header(
        "🗃️ Data Explorer"
    )

    st.write(
        "Gunakan filter berikut untuk mengeksplorasi "
        "dataset hasil integrasi."
    )

    # --------------------------------------------------------
    # Magnitude filter
    # --------------------------------------------------------

    min_magnitude = float(
        df["magnitude"].min()
    )

    max_magnitude = float(
        df["magnitude"].max()
    )

    magnitude_filter = st.slider(
        "Rentang Magnitudo",
        min_value=min_magnitude,
        max_value=max_magnitude,
        value=(
            min_magnitude,
            max_magnitude
        )
    )

    # --------------------------------------------------------
    # Depth filter
    # --------------------------------------------------------

    min_depth = float(
        df["depth_km"].min()
    )

    max_depth = float(
        df["depth_km"].max()
    )

    depth_filter = st.slider(
        "Rentang Kedalaman (km)",
        min_value=min_depth,
        max_value=max_depth,
        value=(
            min_depth,
            max_depth
        )
    )

    # --------------------------------------------------------
    # Year filter
    # --------------------------------------------------------

    available_years = sorted(
        df["year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_years = st.multiselect(
        "Tahun",
        options=available_years,
        default=available_years
    )

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    filtered_df = df[
        df["magnitude"].between(
            magnitude_filter[0],
            magnitude_filter[1]
        )
        &
        df["depth_km"].between(
            depth_filter[0],
            depth_filter[1]
        )
        &
        df["year"].isin(
            selected_years
        )
    ]

    st.divider()

    # --------------------------------------------------------
    # Filter KPIs
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Data Terfilter",
            f"{len(filtered_df):,}"
        )

    with col2:

        if len(filtered_df) > 0:

            st.metric(
                "Rata-rata Magnitudo",
                f"{filtered_df['magnitude'].mean():.2f}"
            )

        else:

            st.metric(
                "Rata-rata Magnitudo",
                "-"
            )

    with col3:

        if len(filtered_df) > 0:

            st.metric(
                "Rata-rata Kedalaman",
                f"{filtered_df['depth_km'].mean():.2f} km"
            )

        else:

            st.metric(
                "Rata-rata Kedalaman",
                "-"
            )

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    display_columns = [
        "usgs_id",
        "event_time",
        "magnitude",
        "place",
        "depth_km",
        "latitude",
        "longitude",
        "distance_km",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    table = filtered_df[
        available_columns
    ].sort_values(
        "event_time",
        ascending=False
    )

    st.dataframe(
        table,
        use_container_width=True,
        height=600
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "Earthquake Insight | Data Science Project | "
    "Integrasi USGS, EMSC, dan Open-Meteo"
)