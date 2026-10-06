from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import requests
import pendulum

from sqlalchemy import create_engine, text, inspect

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path("/usr/local/airflow")
INCLUDE_DIR = BASE_DIR / "include"
INCLUDE_DIR.mkdir(parents=True, exist_ok=True)

# Koneksi MySQL dari dalam container Airflow
MYSQL_URI = os.getenv(
    "MYSQL_URI",
    "mysql+pymysql://root:root@mysql:3306/earthquake_db"
)

# Database
DB_NAME = "earthquake_db"

# ============================================================
# API
# ============================================================

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
EMSC_URL = "https://www.seismicportal.eu/fdsnws/event/1/query"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# ============================================================
# PERIODE DATA
# ============================================================

INITIAL_START_DATE = "2025-10-01"
INITIAL_END_DATE = "2026-09-30"

MIN_MAGNITUDE = 4.0

# ============================================================
# OPEN-METEO
# ============================================================

WEATHER_BATCH_SIZE = 50
WEATHER_DAYS_PER_REQUEST = 7

WEATHER_REQUEST_DELAY = 5

WEATHER_MAX_RETRIES = 8
WEATHER_INITIAL_WAIT = 30
WEATHER_MAX_WAIT = 300


# ============================================================
# ENGINE MYSQL
# ============================================================

def get_engine():
    """
    Membuat koneksi SQLAlchemy ke MySQL.
    """

    return create_engine(
        MYSQL_URI,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
    )


# ============================================================
# CEK TABLE
# ============================================================

def table_exists(engine, table_name):
    """
    Mengecek apakah tabel tersedia di database.
    """

    inspector = inspect(engine)

    return inspector.has_table(table_name)


# ============================================================
# REQUEST JSON
# ============================================================

def request_json(
    url,
    params=None,
    timeout=120,
    max_retries=5,
    initial_wait=15,
    max_wait=300,
):
    """
    Request API dengan retry untuk error sementara.
    """

    for attempt in range(1, max_retries + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={
                    "User-Agent": "EarthquakeDataScience-UHO/1.0"
                },
            )

            if response.status_code == 200:

                return response.json()

            if response.status_code == 429:

                wait_time = min(
                    initial_wait * (2 ** (attempt - 1)),
                    max_wait,
                )

                print(
                    f"HTTP 429 - Too Many Requests. "
                    f"Percobaan {attempt}/{max_retries}. "
                    f"Menunggu {wait_time} detik."
                )

                time.sleep(wait_time)

                continue

            if 500 <= response.status_code < 600:

                wait_time = min(
                    initial_wait * (2 ** (attempt - 1)),
                    max_wait,
                )

                print(
                    f"HTTP {response.status_code}. "
                    f"Percobaan {attempt}/{max_retries}. "
                    f"Menunggu {wait_time} detik."
                )

                time.sleep(wait_time)

                continue

            print(
                f"Request gagal. "
                f"Status={response.status_code}, "
                f"URL={url}"
            )

            print(response.text[:500])

            return None

        except requests.exceptions.Timeout:

            wait_time = min(
                initial_wait * (2 ** (attempt - 1)),
                max_wait,
            )

            print(
                f"Timeout pada percobaan "
                f"{attempt}/{max_retries}. "
                f"Menunggu {wait_time} detik."
            )

            time.sleep(wait_time)

        except requests.exceptions.ConnectionError as e:

            wait_time = min(
                initial_wait * (2 ** (attempt - 1)),
                max_wait,
            )

            print(
                f"Connection error pada percobaan "
                f"{attempt}/{max_retries}: {e}"
            )

            time.sleep(wait_time)

        except Exception as e:

            print(
                f"Error tidak terduga pada request: {e}"
            )

            return None

    print(
        f"Request gagal setelah {max_retries} percobaan."
    )

    return None


# ============================================================
# WINDOW EKSTRAKSI
# ============================================================

def get_extraction_window(engine):
    """
    Menentukan periode pengambilan data.

    Jika tabel sumber belum ada / kosong:
        gunakan periode historis.

    Jika sudah ada:
        lakukan incremental extraction.
    """

    if not table_exists(engine, "usgs_earthquakes"):

        return INITIAL_START_DATE, INITIAL_END_DATE

    try:

        with engine.connect() as conn:

            result = conn.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS total,
                        MAX(event_time) AS max_event_time
                    FROM usgs_earthquakes
                    """
                )
            ).mappings().first()

        total = result["total"]
        max_event_time = result["max_event_time"]

        if total == 0 or max_event_time is None:

            return INITIAL_START_DATE, INITIAL_END_DATE

        max_date = pd.to_datetime(max_event_time)

        start_date = (
            max_date - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")

        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        return start_date, end_date

    except Exception as e:

        print(
            f"Gagal menentukan incremental window: {e}"
        )

        return INITIAL_START_DATE, INITIAL_END_DATE


# ============================================================
# APPEND DATA BARU
# ============================================================

def append_new_data(
    df,
    table_name,
    unique_column,
):
    """
    Menambahkan hanya data yang belum ada.
    """

    if df is None or df.empty:

        print(
            f"Tidak ada data baru untuk {table_name}."
        )

        return 0

    engine = get_engine()

    try:

        if not table_exists(engine, table_name):

            df.to_sql(
                table_name,
                con=engine,
                if_exists="append",
                index=False,
                chunksize=1000,
                method="multi",
            )

            print(
                f"Tabel {table_name} dibuat. "
                f"Rows={len(df)}"
            )

            return len(df)

        inspector = inspect(engine)

        columns = [
            col["name"]
            for col in inspector.get_columns(table_name)
        ]

        if unique_column not in columns:

            raise ValueError(
                f"Kolom {unique_column} tidak ditemukan "
                f"pada tabel {table_name}."
            )

        with engine.connect() as conn:

            existing = pd.read_sql(
                text(
                    f"""
                    SELECT `{unique_column}`
                    FROM `{table_name}`
                    """
                ),
                conn,
            )

        existing_ids = set(
            existing[unique_column]
            .dropna()
            .astype(str)
        )

        df = df.copy()

        df[unique_column] = (
            df[unique_column]
            .astype(str)
        )

        df_new = df[
            ~df[unique_column].isin(existing_ids)
        ].copy()

        if df_new.empty:

            print(
                f"Tidak ada data baru untuk {table_name}."
            )

            return 0

        df_new.to_sql(
            table_name,
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )

        print(
            f"{table_name}: "
            f"{len(df_new)} data baru ditambahkan."
        )

        return len(df_new)

    finally:

        engine.dispose()


# ============================================================
# TASK 1
# EXTRACT USGS
# ============================================================

def extract_usgs():

    print("=" * 70)
    print("TASK 1 - EXTRACT USGS")
    print("=" * 70)

    engine = get_engine()

    try:

        start_date, end_date = get_extraction_window(
            engine
        )

        print(
            f"USGS window: "
            f"{start_date} sampai {end_date}"
        )

        params = {
            "format": "geojson",
            "starttime": start_date,
            "endtime": end_date,
            "minmagnitude": MIN_MAGNITUDE,
            "orderby": "time-asc",
            "limit": 20000,
        }

        payload = request_json(
            USGS_URL,
            params=params,
            timeout=180,
            max_retries=5,
            initial_wait=20,
            max_wait=300,
        )

        if payload is None:

            raise RuntimeError(
                "USGS API tidak mengembalikan data."
            )

        features = payload.get("features", [])

        print(
            f"Jumlah feature USGS: {len(features)}"
        )

        rows = []

        for feature in features:

            try:

                properties = feature.get(
                    "properties",
                    {}
                )

                geometry = feature.get(
                    "geometry",
                    {}
                )

                coordinates = geometry.get(
                    "coordinates",
                    [None, None, None]
                )

                usgs_id = feature.get("id")

                if not usgs_id:

                    continue

                longitude = coordinates[0]
                latitude = coordinates[1]
                depth = coordinates[2]

                event_time = properties.get("time")

                if event_time is not None:

                    event_time = pd.to_datetime(
                        event_time,
                        unit="ms",
                        utc=True,
                    ).tz_convert(
                        "Asia/Makassar"
                    ).tz_localize(None)

                rows.append(
                    {
                        "usgs_id": str(usgs_id),
                        "event_time": event_time,
                        "magnitude": properties.get("mag"),
                        "place": properties.get("place"),
                        "depth_km": depth,
                        "longitude": longitude,
                        "latitude": latitude,
                        "mag_type": properties.get("magType"),
                        "status": properties.get("status"),
                        "tsunami": properties.get("tsunami"),
                    }
                )

            except Exception as e:

                print(
                    f"Gagal memproses feature USGS: {e}"
                )

        df = pd.DataFrame(rows)

        if df.empty:

            print("USGS tidak menghasilkan data baru.")

            return

        numeric_columns = [
            "magnitude",
            "depth_km",
            "longitude",
            "latitude",
            "tsunami",
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df = df.dropna(
            subset=[
                "usgs_id",
                "event_time",
                "magnitude",
                "latitude",
                "longitude",
            ]
        )

        df = df.drop_duplicates(
            subset=["usgs_id"]
        )

        print(
            f"Data USGS setelah cleaning: "
            f"{len(df)} rows"
        )

        append_new_data(
            df,
            "usgs_earthquakes",
            "usgs_id",
        )

    finally:

        engine.dispose()


# ============================================================
# TASK 2
# EXTRACT EMSC
# ============================================================

def parse_emsc_payload(payload):

    rows = []

    if payload is None:

        return rows

    features = payload.get(
        "features",
        []
    )

    for feature in features:

        try:

            properties = feature.get(
                "properties",
                {}
            )

            geometry = feature.get(
                "geometry",
                {}
            )

            coordinates = geometry.get(
                "coordinates",
                [None, None, None]
            )

            emsc_id = (
                feature.get("id")
                or properties.get("id")
                or properties.get("unid")
            )

            if not emsc_id:

                continue

            longitude = coordinates[0]
            latitude = coordinates[1]
            depth = coordinates[2]

            event_time = (
                properties.get("time")
                or properties.get("origin_time")
            )

            if event_time is not None:

                try:

                    event_time = pd.to_datetime(
                        event_time,
                        utc=True,
                    ).tz_convert(
                        "Asia/Makassar"
                    ).tz_localize(None)

                except Exception:

                    event_time = pd.NaT

            rows.append(
                {
                    "emsc_id": str(emsc_id),
                    "emsc_event_time": event_time,
                    "emsc_magnitude": properties.get("mag"),
                    "emsc_place": (
                        properties.get("place")
                        or properties.get("region")
                    ),
                    "emsc_depth_km": depth,
                    "emsc_longitude": longitude,
                    "emsc_latitude": latitude,
                    "emsc_mag_type": (
                        properties.get("magtype")
                        or properties.get("magType")
                    ),
                    "emsc_status": properties.get(
                        "status"
                    ),
                }
            )

        except Exception as e:

            print(
                f"Gagal memproses EMSC feature: {e}"
            )

    return rows


def extract_emsc():

    print("=" * 70)
    print("TASK 2 - EXTRACT EMSC")
    print("=" * 70)

    engine = get_engine()

    try:

        start_date, end_date = get_extraction_window(
            engine
        )

        print(
            f"EMSC window: "
            f"{start_date} sampai {end_date}"
        )

        params = {
            "format": "json",
            "starttime": start_date,
            "endtime": end_date,
            "minmagnitude": MIN_MAGNITUDE,
            "orderby": "time-asc",
            "limit": 20000,
        }

        payload = request_json(
            EMSC_URL,
            params=params,
            timeout=180,
            max_retries=5,
            initial_wait=20,
            max_wait=300,
        )

        if payload is None:

            print(
                "EMSC API tidak mengembalikan data."
            )

            return

        rows = parse_emsc_payload(payload)

        df = pd.DataFrame(rows)

        if df.empty:

            print("Tidak ada data EMSC.")

            return

        numeric_columns = [
            "emsc_magnitude",
            "emsc_depth_km",
            "emsc_longitude",
            "emsc_latitude",
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df = df.dropna(
            subset=[
                "emsc_id",
                "emsc_event_time",
                "emsc_magnitude",
                "emsc_latitude",
                "emsc_longitude",
            ]
        )

        df = df.drop_duplicates(
            subset=["emsc_id"]
        )

        print(
            f"Data EMSC setelah cleaning: "
            f"{len(df)} rows"
        )

        append_new_data(
            df,
            "emsc_earthquakes",
            "emsc_id",
        )

    finally:

        engine.dispose()


# ============================================================
# HAVERSINE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Menghitung jarak antar koordinat dalam kilometer.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 6371.0 * (
        2 * np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# TASK 3
# INTEGRATE USGS + EMSC
# ============================================================

def integrate_seismic():

    print("=" * 70)
    print("TASK 3 - INTEGRATE USGS + EMSC")
    print("=" * 70)

    engine = get_engine()

    try:

        if not table_exists(
            engine,
            "usgs_earthquakes",
        ):

            raise RuntimeError(
                "Tabel usgs_earthquakes belum tersedia."
            )

        if not table_exists(
            engine,
            "emsc_earthquakes",
        ):

            raise RuntimeError(
                "Tabel emsc_earthquakes belum tersedia."
            )

        with engine.connect() as conn:

            usgs = pd.read_sql(
                text(
                    """
                    SELECT *
                    FROM usgs_earthquakes
                    """
                ),
                conn,
            )

            emsc = pd.read_sql(
                text(
                    """
                    SELECT *
                    FROM emsc_earthquakes
                    """
                ),
                conn,
            )

        print(
            f"USGS rows : {len(usgs)}"
        )

        print(
            f"EMSC rows : {len(emsc)}"
        )

        if usgs.empty or emsc.empty:

            raise RuntimeError(
                "Data USGS atau EMSC kosong."
            )

        # ----------------------------------------------------
        # NORMALISASI
        # ----------------------------------------------------

        usgs["event_time"] = pd.to_datetime(
            usgs["event_time"],
            errors="coerce",
        )

        emsc["emsc_event_time"] = pd.to_datetime(
            emsc["emsc_event_time"],
            errors="coerce",
        )

        for col in [
            "magnitude",
            "depth_km",
            "longitude",
            "latitude",
        ]:

            usgs[col] = pd.to_numeric(
                usgs[col],
                errors="coerce",
            )

        for col in [
            "emsc_magnitude",
            "emsc_depth_km",
            "emsc_longitude",
            "emsc_latitude",
        ]:

            emsc[col] = pd.to_numeric(
                emsc[col],
                errors="coerce",
            )

        usgs = usgs.dropna(
            subset=[
                "event_time",
                "latitude",
                "longitude",
                "magnitude",
            ]
        )

        emsc = emsc.dropna(
            subset=[
                "emsc_event_time",
                "emsc_latitude",
                "emsc_longitude",
                "emsc_magnitude",
            ]
        )

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        usgs = usgs.sort_values(
            "event_time"
        )

        emsc = emsc.sort_values(
            "emsc_event_time"
        )

        # ----------------------------------------------------
        # MERGE AS OF BERDASARKAN WAKTU
        # ----------------------------------------------------

        merged = pd.merge_asof(
            usgs,
            emsc,
            left_on="event_time",
            right_on="emsc_event_time",
            direction="nearest",
            tolerance=pd.Timedelta(
                seconds=120
            ),
        )

        # ----------------------------------------------------
        # HITUNG JARAK
        # ----------------------------------------------------

        merged["distance_km"] = haversine_km(
            merged["latitude"],
            merged["longitude"],
            merged["emsc_latitude"],
            merged["emsc_longitude"],
        )

        merged["magnitude_diff"] = (
            merged["magnitude"]
            - merged["emsc_magnitude"]
        ).abs()

        # ----------------------------------------------------
        # FILTER MATCHING
        # ----------------------------------------------------

        merged = merged[
            merged["emsc_id"].notna()
            &
            (merged["distance_km"] <= 50)
            &
            (merged["magnitude_diff"] <= 0.5)
        ].copy()

        print(
            f"Data setelah matching USGS + EMSC: "
            f"{len(merged)} rows"
        )

        if merged.empty:

            raise RuntimeError(
                "Tidak ada data yang berhasil "
                "diintegrasikan antara USGS dan EMSC."
            )

        # ----------------------------------------------------
        # EVENT DATE
        # ----------------------------------------------------

        merged["event_date"] = (
            pd.to_datetime(
                merged["event_time"],
                errors="coerce",
            )
            .dt.date
        )

        # ----------------------------------------------------
        # KOORDINAT WEATHER
        # Dibulatkan agar dapat digunakan sebagai
        # key penghubung dengan Open-Meteo.
        # ----------------------------------------------------

        merged["weather_lat"] = (
            merged["latitude"]
            .round(1)
        )

        merged["weather_lon"] = (
            merged["longitude"]
            .round(1)
        )

        # ----------------------------------------------------
        # DROP DUPLICATE
        # ----------------------------------------------------

        merged = merged.sort_values(
            "event_time"
        )

        merged = merged.drop_duplicates(
            subset=["usgs_id"],
            keep="first",
        )

        # ----------------------------------------------------
        # SIMPAN MYSQL
        # ----------------------------------------------------

        merged.to_sql(
            "earthquake_seismic_integrated",
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=1000,
            method="multi",
        )

        print(
            "Tabel earthquake_seismic_integrated berhasil "
            "diperbarui."
        )

        print(
            f"Jumlah data seismic integrated: "
            f"{len(merged)}"
        )

    finally:

        engine.dispose()


# ============================================================
# TASK 4
# EXTRACT OPEN-METEO
# ============================================================

def extract_openmeteo():

    print("=" * 70)
    print("TASK 4 - EXTRACT OPEN-METEO")
    print("=" * 70)

    engine = get_engine()

    try:

        if not table_exists(
            engine,
            "earthquake_seismic_integrated",
        ):

            raise RuntimeError(
                "Tabel earthquake_seismic_integrated "
                "belum tersedia."
            )

        with engine.connect() as conn:

            seismic = pd.read_sql(
                text(
                    """
                    SELECT
                        event_date,
                        weather_lat,
                        weather_lon
                    FROM earthquake_seismic_integrated
                    WHERE event_date IS NOT NULL
                      AND weather_lat IS NOT NULL
                      AND weather_lon IS NOT NULL
                    """
                ),
                conn,
            )

        if seismic.empty:

            raise RuntimeError(
                "Tidak ada data untuk Open-Meteo."
            )

        # ----------------------------------------------------
        # NORMALISASI
        # ----------------------------------------------------

        seismic["event_date"] = pd.to_datetime(
            seismic["event_date"],
            errors="coerce",
        ).dt.date

        seismic["weather_lat"] = pd.to_numeric(
            seismic["weather_lat"],
            errors="coerce",
        ).round(1)

        seismic["weather_lon"] = pd.to_numeric(
            seismic["weather_lon"],
            errors="coerce",
        ).round(1)

        seismic = seismic.dropna(
            subset=[
                "event_date",
                "weather_lat",
                "weather_lon",
            ]
        )

        weather_keys = (
            seismic[
                [
                    "event_date",
                    "weather_lat",
                    "weather_lon",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        print(
            f"Unique weather keys: "
            f"{len(weather_keys)}"
        )

        # ----------------------------------------------------
        # LOAD CACHE
        # ----------------------------------------------------

        if table_exists(
            engine,
            "openmeteo_weather",
        ):

            with engine.connect() as conn:

                cached = pd.read_sql(
                    text(
                        """
                        SELECT
                            event_date,
                            weather_lat,
                            weather_lon
                        FROM openmeteo_weather
                        """
                    ),
                    conn,
                )

            if not cached.empty:

                cached["event_date"] = pd.to_datetime(
                    cached["event_date"],
                    errors="coerce",
                ).dt.date

                cached["weather_lat"] = pd.to_numeric(
                    cached["weather_lat"],
                    errors="coerce",
                ).round(1)

                cached["weather_lon"] = pd.to_numeric(
                    cached["weather_lon"],
                    errors="coerce",
                ).round(1)

            else:

                cached = pd.DataFrame(
                    columns=[
                        "event_date",
                        "weather_lat",
                        "weather_lon",
                    ]
                )

        else:

            cached = pd.DataFrame(
                columns=[
                    "event_date",
                    "weather_lat",
                    "weather_lon",
                ]
            )

        # ----------------------------------------------------
        # CARI DATA YANG BELUM ADA
        # ----------------------------------------------------

        if not cached.empty:

            missing = (
                weather_keys
                .merge(
                    cached.drop_duplicates(),
                    on=[
                        "event_date",
                        "weather_lat",
                        "weather_lon",
                    ],
                    how="left",
                    indicator=True,
                )
            )

            missing = missing[
                missing["_merge"] == "left_only"
            ].drop(
                columns=["_merge"]
            )

        else:

            missing = weather_keys.copy()

        print(
            f"Weather keys belum tersedia: "
            f"{len(missing)}"
        )

        if missing.empty:

            print(
                "Semua data Open-Meteo sudah tersedia "
                "di cache."
            )

            return

        # ----------------------------------------------------
        # REQUEST PER KOORDINAT
        # ----------------------------------------------------

        new_rows = []

        coordinate_groups = (
            missing[
                [
                    "weather_lat",
                    "weather_lon",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        print(
            f"Unique koordinat yang perlu diambil: "
            f"{len(coordinate_groups)}"
        )

        for index, coordinate in coordinate_groups.iterrows():

            lat = float(
                coordinate["weather_lat"]
            )

            lon = float(
                coordinate["weather_lon"]
            )

            dates_for_coordinate = missing[
                (
                    missing["weather_lat"] == lat
                )
                &
                (
                    missing["weather_lon"] == lon
                )
            ]["event_date"]

            min_date = min(
                dates_for_coordinate
            )

            max_date = max(
                dates_for_coordinate
            )

            print(
                f"[{index + 1}/"
                f"{len(coordinate_groups)}] "
                f"Open-Meteo "
                f"lat={lat}, lon={lon}, "
                f"{min_date} -> {max_date}"
            )

            # ------------------------------------------------
            # BAGI PER 7 HARI
            # ------------------------------------------------

            current_start = pd.Timestamp(
                min_date
            )

            end_timestamp = pd.Timestamp(
                max_date
            )

            while current_start <= end_timestamp:

                current_end = min(
                    current_start
                    + pd.Timedelta(
                        days=WEATHER_DAYS_PER_REQUEST - 1
                    ),
                    end_timestamp,
                )

                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": current_start.strftime(
                        "%Y-%m-%d"
                    ),
                    "end_date": current_end.strftime(
                        "%Y-%m-%d"
                    ),
                    "daily": (
                        "temperature_2m_mean,"
                        "precipitation_sum,"
                        "wind_speed_10m_max,"
                        "surface_pressure_mean"
                    ),
                    "timezone": "Asia/Makassar",
                }

                payload = request_json(
                    OPENMETEO_URL,
                    params=params,
                    timeout=120,
                    max_retries=WEATHER_MAX_RETRIES,
                    initial_wait=WEATHER_INITIAL_WAIT,
                    max_wait=WEATHER_MAX_WAIT,
                )

                if payload is None:

                    print(
                        f"Gagal mengambil Open-Meteo "
                        f"untuk {lat}, {lon}, "
                        f"{current_start.date()} - "
                        f"{current_end.date()}"
                    )

                    current_start = (
                        current_end
                        + pd.Timedelta(days=1)
                    )

                    continue

                daily = payload.get(
                    "daily",
                    {}
                )

                dates = daily.get(
                    "time",
                    []
                )

                temperatures = daily.get(
                    "temperature_2m_mean",
                    []
                )

                precipitation = daily.get(
                    "precipitation_sum",
                    []
                )

                wind_speed = daily.get(
                    "wind_speed_10m_max",
                    []
                )

                pressure = daily.get(
                    "surface_pressure_mean",
                    []
                )

                for i, date_value in enumerate(
                    dates
                ):

                    event_date = pd.to_datetime(
                        date_value
                    ).date()

                    # Hanya simpan tanggal yang memang
                    # dibutuhkan oleh dataset gempa.
                    if event_date not in set(
                        dates_for_coordinate
                    ):

                        continue

                    new_rows.append(
                        {
                            "event_date": event_date,
                            "weather_lat": lat,
                            "weather_lon": lon,
                            "temperature_2m_mean": (
                                temperatures[i]
                                if i < len(temperatures)
                                else None
                            ),
                            "precipitation_sum": (
                                precipitation[i]
                                if i < len(precipitation)
                                else None
                            ),
                            "wind_speed_10m_max": (
                                wind_speed[i]
                                if i < len(wind_speed)
                                else None
                            ),
                            "surface_pressure_mean": (
                                pressure[i]
                                if i < len(pressure)
                                else None
                            ),
                        }
                    )

                current_start = (
                    current_end
                    + pd.Timedelta(days=1)
                )

                time.sleep(
                    WEATHER_REQUEST_DELAY
                )

        # ----------------------------------------------------
        # SIMPAN HASIL
        # ----------------------------------------------------

        weather_df = pd.DataFrame(
            new_rows
        )

        if weather_df.empty:

            print(
                "Tidak ada data Open-Meteo baru."
            )

            return

        weather_df = weather_df.drop_duplicates(
            subset=[
                "event_date",
                "weather_lat",
                "weather_lon",
            ]
        )

        weather_df.to_sql(
            "openmeteo_weather",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=500,
            method="multi",
        )

        print(
            f"Open-Meteo berhasil disimpan: "
            f"{len(weather_df)} rows"
        )

    finally:

        engine.dispose()


# ============================================================
# TASK 5
# BUILD FINAL DATASET
# ============================================================

def build_final_dataset():

    print("=" * 70)
    print("TASK 5 - BUILD FINAL DATASET")
    print("=" * 70)

    engine = get_engine()

    try:

        # ----------------------------------------------------
        # CEK TABLE
        # ----------------------------------------------------

        required_tables = [
            "earthquake_seismic_integrated",
            "openmeteo_weather",
        ]

        for table in required_tables:

            if not table_exists(
                engine,
                table,
            ):

                raise RuntimeError(
                    f"Tabel {table} belum tersedia."
                )

        # ----------------------------------------------------
        # LOAD SEISMIC
        # ----------------------------------------------------

        with engine.connect() as conn:

            seismic = pd.read_sql(
                text(
                    """
                    SELECT *
                    FROM earthquake_seismic_integrated
                    """
                ),
                conn,
            )

            weather = pd.read_sql(
                text(
                    """
                    SELECT *
                    FROM openmeteo_weather
                    """
                ),
                conn,
            )

        print(
            f"Seismic rows : {len(seismic)}"
        )

        print(
            f"Weather rows : {len(weather)}"
        )

        if seismic.empty:

            raise RuntimeError(
                "Data seismic kosong."
            )

        if weather.empty:

            raise RuntimeError(
                "Data Open-Meteo kosong."
            )

        # ----------------------------------------------------
        # NORMALISASI
        # ----------------------------------------------------

        seismic["event_date"] = pd.to_datetime(
            seismic["event_date"],
            errors="coerce",
        ).dt.date

        weather["event_date"] = pd.to_datetime(
            weather["event_date"],
            errors="coerce",
        ).dt.date

        for col in [
            "weather_lat",
            "weather_lon",
        ]:

            seismic[col] = pd.to_numeric(
                seismic[col],
                errors="coerce",
            ).round(1)

            weather[col] = pd.to_numeric(
                weather[col],
                errors="coerce",
            ).round(1)

        # ----------------------------------------------------
        # REMOVE DUPLICATE WEATHER
        # ----------------------------------------------------

        weather = weather.drop_duplicates(
            subset=[
                "event_date",
                "weather_lat",
                "weather_lon",
            ]
        )

        # ----------------------------------------------------
        # INNER JOIN
        # ----------------------------------------------------

        final_df = seismic.merge(
            weather,
            on=[
                "event_date",
                "weather_lat",
                "weather_lon",
            ],
            how="inner",
        )

        print(
            f"Rows setelah INNER JOIN: "
            f"{len(final_df)}"
        )

        if final_df.empty:

            raise RuntimeError(
                "Hasil INNER JOIN kosong. "
                "Periksa kecocokan tanggal dan koordinat."
            )

        # ----------------------------------------------------
        # NORMALISASI EVENT TIME
        # ----------------------------------------------------

        if "event_time" in final_df.columns:

            final_df["event_time"] = pd.to_datetime(
                final_df["event_time"],
                errors="coerce",
            )

        # ----------------------------------------------------
        # DROP DUPLICATE GEMPA
        # ----------------------------------------------------

        if "usgs_id" in final_df.columns:

            final_df = final_df.drop_duplicates(
                subset=["usgs_id"],
                keep="first",
            )

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        if "event_time" in final_df.columns:

            final_df = final_df.sort_values(
                "event_time"
            )

        # ----------------------------------------------------
        # CEK MISSING VALUE
        # ----------------------------------------------------

        print()
        print(
            "MISSING VALUE DATASET FINAL"
        )
        print("-" * 50)

        missing_summary = (
            final_df.isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        print(
            missing_summary[
                missing_summary > 0
            ]
        )

        # ----------------------------------------------------
        # BUAT TABLE FINAL JIKA BELUM ADA
        # ----------------------------------------------------

        if not table_exists(
            engine,
            "earthquake_data",
        ):

            print(
                "Tabel earthquake_data belum ada."
            )

            final_df.to_sql(
                "earthquake_data",
                con=engine,
                if_exists="append",
                index=False,
                chunksize=1000,
                method="multi",
            )

        else:

            # ------------------------------------------------
            # PENTING:
            # TRUNCATE hanya tabel hasil akhir.
            # Source tables TIDAK dihapus.
            #
            # Berbeda dengan if_exists='replace',
            # struktur tabel tetap dipertahankan.
            # ------------------------------------------------

            with engine.begin() as conn:

                conn.execute(
                    text(
                        "TRUNCATE TABLE earthquake_data"
                    )
                )

            final_df.to_sql(
                "earthquake_data",
                con=engine,
                if_exists="append",
                index=False,
                chunksize=1000,
                method="multi",
            )

        # ----------------------------------------------------
        # VALIDASI FINAL
        # ----------------------------------------------------

        with engine.connect() as conn:

            result = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM earthquake_data
                    """
                )
            )

            total = result.scalar()

        print()
        print("=" * 70)
        print("FINAL DATASET BERHASIL")
        print("=" * 70)

        print(
            f"Total rows earthquake_data: {total}"
        )

        print(
            f"Total kolom: {len(final_df.columns)}"
        )

        print()
        print(
            "Sumber API:"
        )

        print(
            "1. USGS"
        )

        print(
            "2. EMSC"
        )

        print(
            "3. Open-Meteo"
        )

        print()
        print(
            "JOIN:"
        )

        print(
            "USGS + EMSC -> seismic integration"
        )

        print(
            "seismic integration INNER JOIN Open-Meteo"
        )

        print(
            "Tidak menggunakan CSV."
        )

        print(
            "Tidak menggunakan NASA POWER."
        )

    finally:

        engine.dispose()


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="earthquake_multisource_etl",

    start_date=pendulum.datetime(
        2026,
        10,
        1,
        tz="Asia/Makassar",
    ),

    schedule="0 1 * * *",

    catchup=False,

    max_active_runs=1,

    tags=[
        "earthquake",
        "data-science",
        "etl",
        "usgs",
        "emsc",
        "open-meteo",
    ],

    default_args={
        "owner": "anissa",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": pendulum.duration(
            minutes=2
        ),
    },

) as dag:

    task_usgs = PythonOperator(
        task_id="extract_usgs",
        python_callable=extract_usgs,
        retries=2,
        retry_delay=pendulum.duration(
            minutes=5
        ),
    )

    task_emsc = PythonOperator(
        task_id="extract_emsc",
        python_callable=extract_emsc,
        retries=1,
        retry_delay=pendulum.duration(
            minutes=2
        ),
    )

    task_integrate = PythonOperator(
        task_id="integrate_seismic",
        python_callable=integrate_seismic,
        retries=1,
        retry_delay=pendulum.duration(
            minutes=2
        ),
    )

    task_openmeteo = PythonOperator(
        task_id="extract_openmeteo",
        python_callable=extract_openmeteo,
        retries=2,
        retry_delay=pendulum.duration(
            minutes=30
        ),
        retry_exponential_backoff=True,
        max_retry_delay=pendulum.duration(
            hours=2
        ),
        execution_timeout=pendulum.duration(
            hours=12
        ),
    )

    task_final = PythonOperator(
        task_id="build_final_dataset",
        python_callable=build_final_dataset,
        retries=1,
        retry_delay=pendulum.duration(
            minutes=2
        ),
    )

    # ========================================================
    # DEPENDENCY
    # ========================================================

    [
        task_usgs,
        task_emsc,
    ] >> task_integrate >> task_openmeteo >> task_final