import os
from pathlib import Path
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Earthquake Insight",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "random_forest_magnitude.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.pkl"


# ============================================================
# DISPLAY LABELS
# ============================================================

LABELS = {
    "usgs_id": "ID USGS",
    "event_time": "Waktu Kejadian",
    "event_date": "Tanggal Kejadian",
    "magnitude": "Magnitudo",
    "place": "Lokasi",
    "depth_km": "Kedalaman (km)",
    "latitude": "Lintang",
    "longitude": "Bujur",
    "mag_type": "Tipe Magnitudo",
    "status": "Status",
    "tsunami": "Indikasi Tsunami",
    "emsc_id": "ID EMSC",
    "emsc_event_time": "Waktu EMSC",
    "emsc_magnitude": "Magnitudo EMSC",
    "emsc_place": "Lokasi EMSC",
    "emsc_depth_km": "Kedalaman EMSC (km)",
    "distance_km": "Jarak Antar Sumber (km)",
    "magnitude_diff": "Selisih Magnitudo",
    "weather_lat": "Lintang Cuaca",
    "weather_lon": "Bujur Cuaca",
    "temperature_2m_mean": "Suhu Rata-rata (°C)",
    "precipitation_sum": "Curah Hujan (mm)",
    "wind_speed_10m_max": "Kecepatan Angin Maksimum (km/jam)",
    "surface_pressure_mean": "Tekanan Permukaan Rata-rata (hPa)",
    "year": "Tahun",
    "month": "Bulan",
    "day": "Hari",
    "hour": "Jam",
    "day_of_week": "Hari dalam Minggu",
    "magnitude_category": "Kategori Magnitudo",
}


def label(column):
    return LABELS.get(column, column)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        opacity: 0.75;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 16px 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
        margin-bottom: 15px;
    }

    .small-note {
        font-size: 13px;
        opacity: 0.70;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.20);
        padding: 12px;
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_database_url():

    database_url = None

    try:
        database_url = st.secrets.get("DATABASE_URL")
    except Exception:
        database_url = None

    if not database_url:
        database_url = os.getenv("DATABASE_URL")

    return database_url


@st.cache_resource
def get_engine():

    database_url = get_database_url()

    if not database_url:
        return None, (
            "DATABASE_URL belum dikonfigurasi. "
            "Tambahkan DATABASE_URL pada Streamlit Secrets."
        )

    if database_url.startswith("mysql://"):
        database_url = database_url.replace(
            "mysql://",
            "mysql+pymysql://",
            1,
        )

    try:

        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={
                "connect_timeout": 15
            },
        )

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return engine, None

    except Exception as e:

        return None, str(e)


engine, db_error = get_engine()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None, f"Model tidak ditemukan: {MODEL_PATH}"

    try:

        model = joblib.load(MODEL_PATH)

        return model, None

    except Exception as e:

        return None, str(e)


@st.cache_resource
def load_model_features():

    if not FEATURES_PATH.exists():
        return None, (
            f"File fitur model tidak ditemukan: {FEATURES_PATH}"
        )

    try:

        features = joblib.load(FEATURES_PATH)

        if isinstance(features, np.ndarray):
            features = features.tolist()

        return list(features), None

    except Exception as e:

        return None, str(e)


model, model_error = load_model()
model_features, feature_error = load_model_features()


# ============================================================
# DATABASE TABLE CHECK
# ============================================================

def check_required_table(engine):

    if engine is None:
        return False, "Engine database tidak tersedia."

    try:

        with engine.connect() as connection:

            result = connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                    AND table_name = 'earthquake_data'
                    """
                )
            )

            exists = result.scalar()

        if exists == 1:
            return True, None

        return False, "Tabel earthquake_data tidak ditemukan."

    except Exception as e:

        return False, str(e)


table_exists, table_error = check_required_table(engine)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_data(_engine):

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
            emsc_event_time,
            emsc_magnitude,
            emsc_place,
            emsc_depth_km,

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

        ORDER BY event_time DESC
    """

    try:

        df = pd.read_sql(
            text(query),
            _engine,
        )

        return df, None

    except Exception as e:

        return pd.DataFrame(), str(e)


# ============================================================
# DATABASE ERROR HANDLING
# ============================================================

if engine is None:

    st.error("❌ Koneksi ke database gagal.")

    st.code(
        db_error if db_error else "DATABASE_URL belum tersedia.",
        language="text",
    )

    st.info(
        """
        Pastikan Streamlit Secrets memiliki DATABASE_URL Aiven.

        Jangan gunakan 127.0.0.1 atau localhost untuk deployment.
        """
    )

    st.stop()


if not table_exists:

    st.error("❌ Tabel earthquake_data tidak ditemukan.")

    st.code(
        table_error if table_error else "Tabel tidak tersedia.",
        language="text",
    )

    st.stop()


df, data_error = load_data(engine)


if data_error:

    st.error("❌ Data gagal dimuat dari database.")

    st.code(
        data_error,
        language="text",
    )

    st.stop()


if df.empty:

    st.warning(
        "Database berhasil terhubung, tetapi data gempa kosong."
    )

    st.stop()


# ============================================================
# DATA PREPROCESSING
# ============================================================

numeric_columns = [
    "magnitude",
    "depth_km",
    "longitude",
    "latitude",
    "tsunami",
    "emsc_magnitude",
    "emsc_depth_km",
    "distance_km",
    "magnitude_diff",
    "weather_lat",
    "weather_lon",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "surface_pressure_mean",
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


df["event_time"] = pd.to_datetime(
    df["event_time"],
    errors="coerce",
)

df["event_date"] = pd.to_datetime(
    df["event_date"],
    errors="coerce",
)

df["event_date"] = df["event_date"].fillna(
    df["event_time"]
)


# ============================================================
# DATE FEATURES
# ============================================================

df["year"] = df["event_time"].dt.year
df["month"] = df["event_time"].dt.month
df["day"] = df["event_time"].dt.day
df["hour"] = df["event_time"].dt.hour
df["day_of_week"] = df["event_time"].dt.dayofweek


# ============================================================
# MAGNITUDE CATEGORY
# ============================================================

def magnitude_category(value):

    if pd.isna(value):
        return "Tidak Diketahui"

    if value < 3:
        return "Sangat Kecil"

    if value < 4:
        return "Kecil"

    if value < 5:
        return "Sedang"

    if value < 6:
        return "Kuat"

    return "Besar"


df["magnitude_category"] = df[
    "magnitude"
].apply(magnitude_category)


# ============================================================
# DATA SUMMARY
# ============================================================

total_events = len(df)

avg_magnitude = df["magnitude"].mean()

max_magnitude = df["magnitude"].max()

avg_depth = df["depth_km"].mean()

min_date = df["event_time"].min()

max_date = df["event_time"].max()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 Earthquake Insight")

st.sidebar.caption(
    "Dashboard Analisis Data Gempa"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigasi",
    [
        "Overview",
        "Spatial Intelligence",
        "Seismic Analytics",
        "Environmental Analytics",
        "Magnitude Prediction",
        "Insights",
        "Data Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.metric(
    "Total Kejadian",
    f"{total_events:,}",
)

if pd.notna(min_date) and pd.notna(max_date):

    st.sidebar.caption(
        f"Periode data:\n"
        f"{min_date.strftime('%d %b %Y')} – "
        f"{max_date.strftime('%d %b %Y')}"
    )

st.sidebar.divider()

st.sidebar.caption(
    "Sumber Data"
)

st.sidebar.write(
    "🌎 USGS\n"
    "🌐 EMSC\n"
    "🌤️ Open-Meteo"
)

if model is not None:

    st.sidebar.success(
        "✓ Random Forest tersedia"
    )

else:

    st.sidebar.warning(
        "⚠ Model belum tersedia"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌍 Earthquake Insight</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Dashboard Data Science untuk eksplorasi, analisis,
    integrasi data lingkungan, dan prediksi magnitudo gempa.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"📊 {total_events:,} kejadian gempa • "
    "Integrasi USGS, EMSC, dan Open-Meteo"
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">📊 Overview Data Gempa</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Kejadian",
            f"{total_events:,}",
        )

    with col2:
        st.metric(
            "Rata-rata Magnitudo",
            f"{avg_magnitude:.2f}",
        )

    with col3:
        st.metric(
            "Magnitudo Tertinggi",
            f"{max_magnitude:.2f}",
        )

    with col4:
        st.metric(
            "Rata-rata Kedalaman",
            f"{avg_depth:.2f} km",
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        monthly = (
            df.dropna(
                subset=["event_time"]
            )
            .assign(
                month_period=lambda x:
                x["event_time"]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby("month_period")
            .size()
            .reset_index(
                name="jumlah_gempa"
            )
        )

        fig = px.line(
            monthly,
            x="month_period",
            y="jumlah_gempa",
            markers=True,
            title="Frekuensi Kejadian Gempa dari Waktu ke Waktu",
            labels={
                "month_period": "Periode",
                "jumlah_gempa": "Jumlah Gempa",
            },
        )

        fig.update_layout(
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        fig = px.histogram(
            df,
            x="magnitude",
            nbins=30,
            title="Distribusi Magnitudo Gempa",
            labels={
                "magnitude": "Magnitudo",
                "count": "Jumlah Kejadian",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    col1, col2 = st.columns(2)

    with col1:

        yearly = (
            df.dropna(
                subset=["year"]
            )
            .groupby("year")
            .agg(
                rata_rata=(
                    "magnitude",
                    "mean",
                ),
                maksimum=(
                    "magnitude",
                    "max",
                ),
            )
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["rata_rata"],
                mode="lines+markers",
                name="Rata-rata",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["maksimum"],
                mode="lines+markers",
                name="Maksimum",
            )
        )

        fig.update_layout(
            title="Perkembangan Magnitudo Berdasarkan Tahun",
            xaxis_title="Tahun",
            yaxis_title="Magnitudo",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        category = (
            df["magnitude_category"]
            .value_counts()
            .reindex(
                [
                    "Sangat Kecil",
                    "Kecil",
                    "Sedang",
                    "Kuat",
                    "Besar",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        category.columns = [
            "kategori",
            "jumlah",
        ]

        fig = px.pie(
            category,
            names="kategori",
            values="jumlah",
            title="Proporsi Kategori Magnitudo",
            hole=0.45,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE 2 — SPATIAL INTELLIGENCE
# ============================================================

elif page == "Spatial Intelligence":

    st.markdown(
        '<div class="section-title">🗺️ Spatial Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Visualisasi ini menunjukkan persebaran kejadian gempa
        berdasarkan lokasi geografis, magnitudo, dan kedalaman.
        """
    )

    map_df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "magnitude",
        ]
    ).copy()

    map_df["abs_magnitude"] = (
        map_df["magnitude"].abs()
    )

    st.subheader(
        "Peta Persebaran Kejadian Gempa"
    )

    fig = px.scatter_map(
        map_df,
        lat="latitude",
        lon="longitude",
        size="abs_magnitude",
        color="magnitude",
        hover_name="place",
        hover_data={
            "latitude": ":.3f",
            "longitude": ":.3f",
            "magnitude": ":.2f",
            "depth_km": ":.2f",
            "event_time": True,
        },
        zoom=1,
        height=600,
        labels={
            "magnitude": "Magnitudo",
            "depth_km": "Kedalaman (km)",
            "latitude": "Lintang",
            "longitude": "Bujur",
        },
    )

    fig.update_layout(
        map_style="open-street-map",
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "Hubungan Lokasi, Kedalaman, dan Magnitudo"
    )

    fig = px.scatter(
        df,
        x="longitude",
        y="latitude",
        size="magnitude",
        color="depth_km",
        hover_data=[
            "place",
            "magnitude",
            "depth_km",
        ],
        labels={
            "longitude": "Bujur",
            "latitude": "Lintang",
            "depth_km": "Kedalaman (km)",
            "magnitude": "Magnitudo",
            "place": "Lokasi",
        },
        title="Distribusi Spasial Berdasarkan Kedalaman",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# PAGE 3 — SEISMIC ANALYTICS
# ============================================================

elif page == "Seismic Analytics":

    st.markdown(
        '<div class="section-title">📈 Seismic Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Analisis karakteristik gempa berdasarkan magnitudo,
        kedalaman, waktu kejadian, dan tipe pengukuran.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="depth_km",
            nbins=40,
            title="Distribusi Kedalaman Gempa",
            labels={
                "depth_km": "Kedalaman (km)",
                "count": "Jumlah Gempa",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        fig = px.scatter(
            df,
            x="depth_km",
            y="magnitude",
            color="magnitude",
            hover_data=[
                "place",
                "event_time",
            ],
            labels={
                "depth_km": "Kedalaman (km)",
                "magnitude": "Magnitudo",
                "place": "Lokasi",
            },
            title="Hubungan Kedalaman dan Magnitudo",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "📋 Statistik Gempa Berdasarkan Tahun"
    )

    yearly = (
        df.dropna(
            subset=["year"]
        )
        .groupby("year")
        .agg(
            rata_rata_magnitudo=(
                "magnitude",
                "mean",
            ),
            maksimum_magnitudo=(
                "magnitude",
                "max",
            ),
            rata_rata_kedalaman=(
                "depth_km",
                "mean",
            ),
            jumlah_gempa=(
                "magnitude",
                "count",
            ),
        )
        .reset_index()
    )

    yearly.columns = [
        "Tahun",
        "Rata-rata Magnitudo",
        "Magnitudo Maksimum",
        "Rata-rata Kedalaman (km)",
        "Jumlah Gempa",
    ]

    st.dataframe(
        yearly.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Distribusi Tipe Magnitudo"
    )

    if "mag_type" in df.columns:

        mag_type = (
            df["mag_type"]
            .fillna("Tidak Diketahui")
            .value_counts()
            .reset_index()
        )

        mag_type.columns = [
            "tipe",
            "jumlah",
        ]

        fig = px.bar(
            mag_type,
            x="tipe",
            y="jumlah",
            title="Frekuensi Tipe Magnitudo",
            labels={
                "tipe": "Tipe Magnitudo",
                "jumlah": "Jumlah Kejadian",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "Statistik Deskriptif"
    )

    descriptive = df[
        [
            "magnitude",
            "depth_km",
            "latitude",
            "longitude",
        ]
    ].describe().round(3)

    descriptive.index = [
        "Jumlah",
        "Rata-rata",
        "Standar Deviasi",
        "Minimum",
        "Kuartil 25%",
        "Median",
        "Kuartil 75%",
        "Maksimum",
    ]

    descriptive.columns = [
        "Magnitudo",
        "Kedalaman (km)",
        "Lintang",
        "Bujur",
    ]

    st.dataframe(
        descriptive,
        use_container_width=True,
    )


# ============================================================
# PAGE 4 — ENVIRONMENTAL ANALYTICS
# ============================================================

elif page == "Environmental Analytics":

    st.markdown(
        '<div class="section-title">🌤️ Environmental Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Analisis hubungan antara karakteristik lingkungan
        dari Open-Meteo dengan magnitudo kejadian gempa.
        """
    )

    environmental_columns = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean",
    ]

    environmental_labels = {
        "temperature_2m_mean": "Suhu Rata-rata",
        "precipitation_sum": "Curah Hujan",
        "wind_speed_10m_max": "Kecepatan Angin Maksimum",
        "surface_pressure_mean": "Tekanan Permukaan Rata-rata",
    }

    available_environmental = [
        col
        for col in environmental_columns
        if col in df.columns
    ]

    if not available_environmental:

        st.warning(
            "Data lingkungan dari Open-Meteo tidak tersedia."
        )

    else:

        col1, col2 = st.columns(2)

        for index, column in enumerate(
            available_environmental
        ):

            temp_df = df[
                [
                    column,
                    "magnitude",
                ]
            ].dropna()

            if temp_df.empty:
                continue

            display_name = environmental_labels[
                column
            ]

            with (
                col1
                if index % 2 == 0
                else col2
            ):

                fig = px.scatter(
                    temp_df,
                    x=column,
                    y="magnitude",
                    trendline="ols",
                    labels={
                        column: display_name,
                        "magnitude": "Magnitudo Gempa",
                    },
                    title=(
                        f"{display_name} "
                        "vs Magnitudo Gempa"
                    ),
                )

                fig.update_layout(
                    height=420,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        st.subheader(
            "Korelasi Variabel Lingkungan dengan Magnitudo"
        )

        correlation_columns = [
            "magnitude"
        ] + available_environmental

        correlation = (
            df[correlation_columns]
            .corr(numeric_only=True)
            .round(3)
        )

        correlation_display = correlation.rename(
            index=label,
            columns=label,
        )

        fig = px.imshow(
            correlation_display,
            text_auto=True,
            aspect="auto",
            title="Matriks Korelasi Lingkungan dan Magnitudo",
            labels={
                "x": "Variabel",
                "y": "Variabel",
                "color": "Korelasi",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Nilai korelasi mendekati +1 menunjukkan hubungan "
            "positif kuat, sedangkan mendekati -1 menunjukkan "
            "hubungan negatif kuat. Korelasi tidak berarti "
            "hubungan sebab-akibat."
        )


# ============================================================
# PAGE 5 — MAGNITUDE PREDICTION
# ============================================================

elif page == "Magnitude Prediction":

    st.markdown(
        '<div class="section-title">🤖 AI Magnitude Prediction</div>',
        unsafe_allow_html=True,
    )

    st.info(
        """
        Model yang digunakan adalah **Random Forest Regressor**.
        Model memperkirakan magnitudo berdasarkan karakteristik
        lokasi, kedalaman, waktu kejadian, jarak, dan kondisi
        lingkungan.
        """
    )

    if model is None:

        st.error(
            "Model Random Forest tidak dapat dimuat."
        )

        if model_error:
            st.code(
                model_error,
                language="text",
            )

    elif model_features is None:

        st.error(
            "Fitur model tidak dapat dimuat."
        )

        if feature_error:
            st.code(
                feature_error,
                language="text",
            )

    else:

        st.subheader(
            "Masukkan Karakteristik Kejadian"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            latitude = st.number_input(
                "Lintang",
                value=-4.0,
                min_value=-90.0,
                max_value=90.0,
                step=0.01,
            )

            longitude = st.number_input(
                "Bujur",
                value=122.5,
                min_value=-180.0,
                max_value=180.0,
                step=0.01,
            )

            depth_km = st.number_input(
                "Kedalaman (km)",
                value=10.0,
                min_value=0.0,
                step=0.1,
            )

        with col2:

            distance_km = st.number_input(
                "Jarak Antar Sumber (km)",
                value=0.0,
                min_value=0.0,
                step=0.1,
            )

            temperature = st.number_input(
                "Suhu Rata-rata (°C)",
                value=25.0,
                step=0.1,
            )

            precipitation = st.number_input(
                "Curah Hujan (mm)",
                value=0.0,
                min_value=0.0,
                step=0.1,
            )

        with col3:

            wind_speed = st.number_input(
                "Kecepatan Angin Maksimum (km/jam)",
                value=10.0,
                min_value=0.0,
                step=0.1,
            )

            pressure = st.number_input(
                "Tekanan Permukaan Rata-rata (hPa)",
                value=1013.0,
                step=0.1,
            )

            event_date_input = st.date_input(
                "Tanggal Kejadian",
                value=date.today(),
            )

            event_time_input = st.time_input(
                "Waktu Kejadian",
                value=datetime.now().time(),
            )

        predict_button = st.button(
            "🔮 Prediksi Magnitudo",
            type="primary",
            use_container_width=True,
        )

        if predict_button:

            event_datetime = datetime.combine(
                event_date_input,
                event_time_input,
            )

            input_dict = {

                "latitude": latitude,

                "longitude": longitude,

                "depth_km": depth_km,

                "distance_km": distance_km,

                "temperature_2m_mean":
                    temperature,

                "precipitation_sum":
                    precipitation,

                "wind_speed_10m_max":
                    wind_speed,

                "surface_pressure_mean":
                    pressure,

                "year":
                    event_datetime.year,

                "month":
                    event_datetime.month,

                "day":
                    event_datetime.day,

                "hour":
                    event_datetime.hour,

                "day_of_week":
                    event_datetime.weekday(),
            }

            input_df = pd.DataFrame(
                [input_dict]
            )

            for feature in model_features:

                if feature not in input_df.columns:

                    input_df[feature] = 0.0

            input_df = input_df[
                model_features
            ]

            try:

                prediction = model.predict(
                    input_df
                )[0]

                prediction = float(
                    prediction
                )

                category = magnitude_category(
                    prediction
                )

                st.success(
                    "Prediksi berhasil dilakukan."
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Prediksi Magnitudo",
                        f"{prediction:.2f}",
                    )

                with col2:

                    st.metric(
                        "Kategori",
                        category,
                    )

                st.divider()

                st.subheader(
                    "Parameter yang Digunakan"
                )

                display_input = pd.DataFrame(
                    {
                        "Parameter": [
                            "Lintang",
                            "Bujur",
                            "Kedalaman (km)",
                            "Jarak Antar Sumber (km)",
                            "Suhu Rata-rata (°C)",
                            "Curah Hujan (mm)",
                            "Kecepatan Angin Maksimum (km/jam)",
                            "Tekanan Permukaan Rata-rata (hPa)",
                            "Tanggal",
                            "Waktu",
                        ],
                        "Nilai": [
                            latitude,
                            longitude,
                            depth_km,
                            distance_km,
                            temperature,
                            precipitation,
                            wind_speed,
                            pressure,
                            event_date_input,
                            event_time_input,
                        ],
                    }
                )

                st.dataframe(
                    display_input,
                    use_container_width=True,
                    hide_index=True,
                )

                st.warning(
                    """
                    Catatan: prediksi ini merupakan hasil model
                    machine learning berdasarkan data yang digunakan
                    saat pelatihan. Model bukan alat untuk menentukan
                    secara pasti kapan, di mana, atau apakah gempa
                    akan terjadi.
                    """
                )

            except Exception as e:

                st.error(
                    "Prediksi gagal dilakukan."
                )

                st.code(
                    str(e),
                    language="text",
                )


# ============================================================
# PAGE 6 — INSIGHTS
# ============================================================

elif page == "Insights":

    st.markdown(
        '<div class="section-title">💡 Insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Halaman ini merangkum temuan penting yang diperoleh
        dari proses eksplorasi dan analisis data.
        """
    )

    max_row = df.loc[
        df["magnitude"].idxmax()
    ]

    min_depth = df["depth_km"].min()

    max_depth = df["depth_km"].max()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Magnitudo Tertinggi",
            f"{max_row['magnitude']:.2f}",
        )

    with col2:

        st.metric(
            "Kedalaman Terdangkal",
            f"{min_depth:.2f} km",
        )

    with col3:

        st.metric(
            "Kedalaman Terdalam",
            f"{max_depth:.2f} km",
        )

    st.divider()

    numeric_df = df.select_dtypes(
        include=np.number
    )

    correlation = (
        numeric_df
        .corr()["magnitude"]
        .drop("magnitude")
        .dropna()
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    if not correlation.empty:

        strongest_feature = correlation.index[0]

        strongest_value = correlation.iloc[0]

        strongest_label = label(
            strongest_feature
        )

        st.subheader(
            "Variabel dengan Korelasi Terkuat"
        )

        if strongest_value > 0:

            relationship = "positif"

        else:

            relationship = "negatif"

        st.markdown(
            f"""
            Berdasarkan dataset yang dianalisis,
            **{strongest_label}** memiliki korelasi
            paling kuat dengan magnitudo, dengan nilai
            **{strongest_value:.3f}**.

            Hubungan tersebut bersifat **{relationship}**.
            """
        )

        corr_plot = (
            correlation
            .reset_index()
        )

        corr_plot.columns = [
            "variabel",
            "korelasi",
        ]

        corr_plot["variabel"] = (
            corr_plot["variabel"]
            .map(label)
        )

        fig = px.bar(
            corr_plot,
            x="variabel",
            y="korelasi",
            title="Korelasi Variabel terhadap Magnitudo",
            labels={
                "variabel": "Variabel",
                "korelasi": "Koefisien Korelasi",
            },
        )

        fig.update_layout(
            xaxis_tickangle=-35,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "📌 Ringkasan Temuan"
    )

    location = (
        max_row["place"]
        if pd.notna(max_row["place"])
        else "tidak tersedia"
    )

    st.markdown(
        f"""
        - Dataset terdiri dari **{total_events:,} kejadian gempa**.
        - Magnitudo tertinggi yang tercatat adalah
          **{max_row["magnitude"]:.2f}**.
        - Kejadian tersebut tercatat di lokasi
          **{location}**.
        - Kedalaman gempa pada dataset memiliki rentang
          **{min_depth:.2f}–{max_depth:.2f} km**.
        - Data Open-Meteo digunakan sebagai variabel lingkungan
          untuk menganalisis hubungan kondisi lingkungan
          dengan karakteristik gempa.
        - Model **Random Forest Regressor** digunakan sebagai
          pendekatan machine learning untuk memperkirakan
          magnitudo.
        """
    )

    st.info(
        """
        **Catatan interpretasi:** korelasi menunjukkan hubungan
        statistik antarvariabel dan tidak dapat langsung
        diartikan sebagai hubungan sebab-akibat.
        """
    )


# ============================================================
# PAGE 7 — DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="section-title">🔎 Data Explorer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Gunakan filter berikut untuk mengeksplorasi data gempa
        berdasarkan magnitudo, kedalaman, dan tahun kejadian.
        """
    )

    st.subheader(
        "Filter Data"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        min_mag = float(
            df["magnitude"].min()
        )

        max_mag = float(
            df["magnitude"].max()
        )

        magnitude_range = st.slider(
            "Rentang Magnitudo",
            min_value=min_mag,
            max_value=max_mag,
            value=(
                min_mag,
                max_mag,
            ),
        )

    with col2:

        min_depth_value = float(
            df["depth_km"].min()
        )

        max_depth_value = float(
            df["depth_km"].max()
        )

        depth_range = st.slider(
            "Rentang Kedalaman (km)",
            min_value=min_depth_value,
            max_value=max_depth_value,
            value=(
                min_depth_value,
                max_depth_value,
            ),
        )

    with col3:

        year_values = sorted(
            df["year"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if year_values:

            selected_year = st.selectbox(
                "Tahun Kejadian",
                ["Semua"] + year_values,
            )

        else:

            selected_year = "Semua"

    filtered_df = df[
        df["magnitude"].between(
            magnitude_range[0],
            magnitude_range[1],
        )
        &
        df["depth_km"].between(
            depth_range[0],
            depth_range[1],
        )
    ].copy()

    if selected_year != "Semua":

        filtered_df = filtered_df[
            filtered_df["year"]
            == selected_year
        ]

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Data Terpilih",
            f"{len(filtered_df):,}",
        )

    with col2:

        if not filtered_df.empty:

            st.metric(
                "Rata-rata Magnitudo",
                f"{filtered_df['magnitude'].mean():.2f}",
            )

        else:

            st.metric(
                "Rata-rata Magnitudo",
                "-",
            )

    with col3:

        if not filtered_df.empty:

            st.metric(
                "Magnitudo Tertinggi",
                f"{filtered_df['magnitude'].max():.2f}",
            )

        else:

            st.metric(
                "Magnitudo Tertinggi",
                "-",
            )

    st.subheader(
        "📋 Data Kejadian Gempa"
    )

    display_columns = [
        "usgs_id",
        "event_time",
        "magnitude",
        "magnitude_category",
        "depth_km",
        "latitude",
        "longitude",
        "place",
        "emsc_id",
        "emsc_magnitude",
        "distance_km",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    display_df = filtered_df[
        display_columns
    ].copy()

    display_df = display_df.rename(
        columns=LABELS
    )

    if "Waktu Kejadian" in display_df.columns:

        display_df["Waktu Kejadian"] = (
            display_df["Waktu Kejadian"]
            .dt.strftime("%d-%m-%Y %H:%M")
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=550,
    )

    st.caption(
        f"Menampilkan {len(display_df):,} data "
        "sesuai filter yang dipilih."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 Earthquake Insight | Data Science Project | "
    "USGS • EMSC • Open-Meteo"
)