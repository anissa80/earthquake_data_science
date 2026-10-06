import os
import requests
import pandas as pd

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.operators.python import PythonOperator

from sqlalchemy import create_engine, text


# ============================================================
# KONFIGURASI API
# ============================================================

USGS_URL = (
    "https://earthquake.usgs.gov/"
    "fdsnws/event/1/query"
)

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/"
    "v1/archive"
)

NASA_URL = (
    "https://power.larc.nasa.gov/"
    "api/temporal/daily/point"
)


# ============================================================
# KONFIGURASI MYSQL
# ============================================================

MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "mysql"
)

MYSQL_PORT = os.getenv(
    "MYSQL_PORT",
    "3306"
)

MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)

MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    "root"
)

MYSQL_DB = os.getenv(
    "MYSQL_DB",
    "earthquake_db"
)


ENGINE_URL = (
    f"mysql+pymysql://"
    f"{MYSQL_USER}:"
    f"{MYSQL_PASSWORD}@"
    f"{MYSQL_HOST}:"
    f"{MYSQL_PORT}/"
    f"{MYSQL_DB}"
)


# ============================================================
# KONFIGURASI DATA
# ============================================================

# Ambil data gempa 7 hari terakhir
RAW_DAYS = 7

# Magnitude minimum
MIN_MAGNITUDE = 4.5

# Maksimal event yang diambil dalam satu proses
MAX_EVENTS = 100


# ============================================================
# DATABASE ENGINE
# ============================================================

def get_engine():

    return create_engine(
        ENGINE_URL,
        pool_pre_ping=True
    )


# ============================================================
# JOIN KEY
# ============================================================

def make_join_key(
    date_value,
    latitude,
    longitude
):

    return (
        f"{date_value}_"
        f"{round(float(latitude), 2)}_"
        f"{round(float(longitude), 2)}"
    )


# ============================================================
# FUNGSI MEMBERSIHKAN NaN
# ============================================================

def clean_nan(value):

    if pd.isna(value):
        return None

    return value


# ============================================================
# 1. LOAD USGS EARTHQUAKE
# ============================================================

def load_usgs():

    print("========================================")
    print("MEMULAI PROSES USGS")
    print("========================================")

    # --------------------------------------------------------
    # Tentukan periode data
    # --------------------------------------------------------

    end_date = datetime.utcnow()

    start_date = (
        end_date -
        timedelta(days=RAW_DAYS)
    )

    # --------------------------------------------------------
    # Parameter API USGS
    # --------------------------------------------------------

    params = {

        "format": "geojson",

        "starttime":
            start_date.strftime(
                "%Y-%m-%d"
            ),

        "endtime":
            end_date.strftime(
                "%Y-%m-%d"
            ),

        "minmagnitude":
            MIN_MAGNITUDE,

        "limit":
            MAX_EVENTS,

        "orderby":
            "time"
    }

    print(
        "Mengambil data USGS..."
    )

    # --------------------------------------------------------
    # Request API
    # --------------------------------------------------------

    response = requests.get(
        USGS_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    # --------------------------------------------------------
    # Transformasi data
    # --------------------------------------------------------

    rows = []

    for feature in data.get(
        "features",
        []
    ):

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
            []
        )

        if len(coordinates) < 3:
            continue

        longitude = coordinates[0]

        latitude = coordinates[1]

        depth = coordinates[2]

        event_id = feature.get(
            "id"
        )

        event_timestamp = (
            properties.get("time")
        )

        if event_timestamp is None:
            continue

        # ----------------------------------------------------
        # Konversi timestamp
        # ----------------------------------------------------

        event_datetime = (
            datetime.fromtimestamp(
                event_timestamp / 1000
            )
        )

        event_date = (
            event_datetime.strftime(
                "%Y-%m-%d"
            )
        )

        # ----------------------------------------------------
        # Join key
        # ----------------------------------------------------

        join_key = make_join_key(
            event_date,
            latitude,
            longitude
        )

        # ----------------------------------------------------
        # Masukkan data
        # ----------------------------------------------------

        rows.append({

            "event_id":
                event_id,

            "event_time":
                event_datetime,

            "magnitude":
                properties.get(
                    "mag"
                ),

            "depth_km":
                depth,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "place":
                properties.get(
                    "place"
                ),

            "tsunami":
                properties.get(
                    "tsunami"
                ),

            "mag_type":
                properties.get(
                    "magType"
                ),

            "status":
                properties.get(
                    "status"
                ),

            "join_key":
                join_key
        })

    # --------------------------------------------------------
    # Tidak ada data
    # --------------------------------------------------------

    if not rows:

        print(
            "Tidak ada data USGS."
        )

        return

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Membersihkan NaN
    # --------------------------------------------------------

    df = df.astype(object).where(
        pd.notna(df),
        None
    )

    print(
        f"Data USGS yang diproses: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    engine = get_engine()

    # --------------------------------------------------------
    # Buat tabel jika belum ada
    # --------------------------------------------------------

    create_table_sql = """

    CREATE TABLE IF NOT EXISTS
    earthquake_data (

        event_id VARCHAR(100)
        PRIMARY KEY,

        event_time DATETIME,

        magnitude DOUBLE,

        depth_km DOUBLE,

        latitude DOUBLE,

        longitude DOUBLE,

        place TEXT,

        tsunami INT,

        mag_type VARCHAR(50),

        status VARCHAR(50),

        join_key VARCHAR(150)

    )

    """

    # --------------------------------------------------------
    # INSERT / UPDATE
    # --------------------------------------------------------

    insert_sql = """

    INSERT INTO earthquake_data (

        event_id,
        event_time,
        magnitude,
        depth_km,
        latitude,
        longitude,
        place,
        tsunami,
        mag_type,
        status,
        join_key

    )

    VALUES (

        :event_id,
        :event_time,
        :magnitude,
        :depth_km,
        :latitude,
        :longitude,
        :place,
        :tsunami,
        :mag_type,
        :status,
        :join_key

    )

    ON DUPLICATE KEY UPDATE

        event_time =
            VALUES(event_time),

        magnitude =
            VALUES(magnitude),

        depth_km =
            VALUES(depth_km),

        latitude =
            VALUES(latitude),

        longitude =
            VALUES(longitude),

        place =
            VALUES(place),

        tsunami =
            VALUES(tsunami),

        mag_type =
            VALUES(mag_type),

        status =
            VALUES(status),

        join_key =
            VALUES(join_key)

    """

    with engine.begin() as conn:

        conn.execute(
            text(create_table_sql)
        )

        for _, row in df.iterrows():

            row_data = row.to_dict()

            # ----------------------------------------------
            # Pastikan NaN menjadi NULL
            # ----------------------------------------------

            for key in row_data:

                row_data[key] = clean_nan(
                    row_data[key]
                )

            conn.execute(
                text(insert_sql),
                row_data
            )

    print(
        "USGS selesai."
    )

    print(
        "Data lama dipertahankan."
    )

    print(
        "Data baru ditambahkan."
    )

    print(
        "Data event yang sama diperbarui."
    )


# ============================================================
# 2. LOAD OPEN-METEO
# ============================================================

def load_openmeteo():

    print("========================================")
    print("MEMULAI PROSES OPEN-METEO")
    print("========================================")

    engine = get_engine()

    # --------------------------------------------------------
    # Ambil event dari tabel USGS
    # --------------------------------------------------------

    query = """

    SELECT

        event_id,

        DATE(event_time)
        AS weather_date,

        latitude,

        longitude

    FROM earthquake_data

    WHERE event_time IS NOT NULL

    """

    events = pd.read_sql(
        query,
        engine
    )

    if events.empty:

        print(
            "Tidak ada event untuk Open-Meteo."
        )

        return

    rows = []

    # --------------------------------------------------------
    # Loop event
    # --------------------------------------------------------

    for _, event in events.iterrows():

        date = pd.to_datetime(
            event["weather_date"]
        ).strftime(
            "%Y-%m-%d"
        )

        try:

            params = {

                "latitude":
                    float(
                        event["latitude"]
                    ),

                "longitude":
                    float(
                        event["longitude"]
                    ),

                "start_date":
                    date,

                "end_date":
                    date,

                "daily": [
                    "temperature_2m_mean",
                    "precipitation_sum",
                    "wind_speed_10m_max"
                ],

                "timezone":
                    "UTC"
            }

            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            daily = data.get(
                "daily",
                {}
            )

            if not daily.get(
                "time"
            ):

                continue

            rows.append({

                "event_id":
                    event["event_id"],

                "weather_date":
                    date,

                "latitude":
                    event["latitude"],

                "longitude":
                    event["longitude"],

                "temperature_mean":
                    daily[
                        "temperature_2m_mean"
                    ][0],

                "precipitation_sum":
                    daily[
                        "precipitation_sum"
                    ][0],

                "wind_speed_max":
                    daily[
                        "wind_speed_10m_max"
                    ][0],

                "join_key":
                    make_join_key(
                        date,
                        event["latitude"],
                        event["longitude"]
                    )
            })

        except Exception as e:

            print(
                "Gagal Open-Meteo "
                f"{event['event_id']}: "
                f"{e}"
            )

    if not rows:

        print(
            "Tidak ada data Open-Meteo."
        )

        return

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # NaN menjadi None
    # --------------------------------------------------------

    df = df.astype(object).where(
        pd.notna(df),
        None
    )

    print(
        f"Data Open-Meteo diproses: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Buat tabel
    # --------------------------------------------------------

    create_table_sql = """

    CREATE TABLE IF NOT EXISTS
    openmeteo_weather (

        event_id VARCHAR(100)
        PRIMARY KEY,

        weather_date DATE,

        latitude DOUBLE,

        longitude DOUBLE,

        temperature_mean DOUBLE,

        precipitation_sum DOUBLE,

        wind_speed_max DOUBLE,

        join_key VARCHAR(150)

    )

    """

    # --------------------------------------------------------
    # Insert / update
    # --------------------------------------------------------

    insert_sql = """

    INSERT INTO openmeteo_weather (

        event_id,
        weather_date,
        latitude,
        longitude,
        temperature_mean,
        precipitation_sum,
        wind_speed_max,
        join_key

    )

    VALUES (

        :event_id,
        :weather_date,
        :latitude,
        :longitude,
        :temperature_mean,
        :precipitation_sum,
        :wind_speed_max,
        :join_key

    )

    ON DUPLICATE KEY UPDATE

        weather_date =
            VALUES(weather_date),

        latitude =
            VALUES(latitude),

        longitude =
            VALUES(longitude),

        temperature_mean =
            VALUES(temperature_mean),

        precipitation_sum =
            VALUES(precipitation_sum),

        wind_speed_max =
            VALUES(wind_speed_max),

        join_key =
            VALUES(join_key)

    """

    with engine.begin() as conn:

        conn.execute(
            text(create_table_sql)
        )

        for _, row in df.iterrows():

            row_data = row.to_dict()

            for key in row_data:

                row_data[key] = clean_nan(
                    row_data[key]
                )

            conn.execute(
                text(insert_sql),
                row_data
            )

    print(
        "Open-Meteo selesai."
    )


# ============================================================
# 3. LOAD NASA POWER
# ============================================================

def load_nasa_power():

    print("========================================")
    print("MEMULAI PROSES NASA POWER")
    print("========================================")

    engine = get_engine()

    # --------------------------------------------------------
    # Ambil event dari USGS
    # --------------------------------------------------------

    query = """

    SELECT

        event_id,

        DATE(event_time)
        AS weather_date,

        latitude,

        longitude

    FROM earthquake_data

    WHERE event_time IS NOT NULL

    """

    events = pd.read_sql(
        query,
        engine
    )

    if events.empty:

        print(
            "Tidak ada event untuk NASA POWER."
        )

        return

    rows = []

    # --------------------------------------------------------
    # Loop event
    # --------------------------------------------------------

    for _, event in events.iterrows():

        date = pd.to_datetime(
            event["weather_date"]
        ).strftime(
            "%Y%m%d"
        )

        try:

            params = {

                "parameters":
                    (
                        "T2M,"
                        "PRECTOTCORR,"
                        "WS10M,"
                        "ALLSKY_SFC_SW_DWN"
                    ),

                "community":
                    "AG",

                "longitude":
                    float(
                        event["longitude"]
                    ),

                "latitude":
                    float(
                        event["latitude"]
                    ),

                "start":
                    date,

                "end":
                    date,

                "format":
                    "JSON"
            }

            response = requests.get(
                NASA_URL,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            properties = data.get(
                "properties",
                {}
            )

            parameter = properties.get(
                "parameter",
                {}
            )

            # ------------------------------------------------
            # Fungsi mengambil nilai NASA
            # ------------------------------------------------

            def get_value(name):

                values = parameter.get(
                    name,
                    {}
                )

                if not values:

                    return None

                value = list(
                    values.values()
                )[0]

                # --------------------------------------------
                # -999 = missing value
                # --------------------------------------------

                if value == -999:

                    return None

                return value

            # ------------------------------------------------
            # Format tanggal
            # ------------------------------------------------

            date_normal = pd.to_datetime(
                date,
                format="%Y%m%d"
            ).strftime(
                "%Y-%m-%d"
            )

            # ------------------------------------------------
            # Masukkan data
            # ------------------------------------------------

            rows.append({

                "event_id":
                    event["event_id"],

                "weather_date":
                    date_normal,

                "latitude":
                    event["latitude"],

                "longitude":
                    event["longitude"],

                "temperature_mean":
                    get_value("T2M"),

                "precipitation":
                    get_value(
                        "PRECTOTCORR"
                    ),

                "wind_speed":
                    get_value(
                        "WS10M"
                    ),

                "solar_radiation":
                    get_value(
                        "ALLSKY_SFC_SW_DWN"
                    ),

                "join_key":
                    make_join_key(
                        date_normal,
                        event["latitude"],
                        event["longitude"]
                    )
            })

        except Exception as e:

            print(
                "Gagal NASA POWER "
                f"{event['event_id']}: "
                f"{e}"
            )

    if not rows:

        print(
            "Tidak ada data NASA POWER."
        )

        return

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # PENTING:
    # NaN dari Pandas diubah menjadi None
    # sebelum dikirim ke MySQL.
    # --------------------------------------------------------

    df = df.astype(object).where(
        pd.notna(df),
        None
    )

    print(
        f"Data NASA POWER diproses: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Buat tabel jika belum ada
    # --------------------------------------------------------

    create_table_sql = """

    CREATE TABLE IF NOT EXISTS
    nasa_power_weather (

        event_id VARCHAR(100)
        PRIMARY KEY,

        weather_date DATE,

        latitude DOUBLE,

        longitude DOUBLE,

        temperature_mean DOUBLE,

        precipitation DOUBLE,

        wind_speed DOUBLE,

        solar_radiation DOUBLE,

        join_key VARCHAR(150)

    )

    """

    # --------------------------------------------------------
    # INSERT / UPDATE
    # --------------------------------------------------------

    insert_sql = """

    INSERT INTO nasa_power_weather (

        event_id,
        weather_date,
        latitude,
        longitude,
        temperature_mean,
        precipitation,
        wind_speed,
        solar_radiation,
        join_key

    )

    VALUES (

        :event_id,
        :weather_date,
        :latitude,
        :longitude,
        :temperature_mean,
        :precipitation,
        :wind_speed,
        :solar_radiation,
        :join_key

    )

    ON DUPLICATE KEY UPDATE

        weather_date =
            VALUES(weather_date),

        latitude =
            VALUES(latitude),

        longitude =
            VALUES(longitude),

        temperature_mean =
            VALUES(temperature_mean),

        precipitation =
            VALUES(precipitation),

        wind_speed =
            VALUES(wind_speed),

        solar_radiation =
            VALUES(solar_radiation),

        join_key =
            VALUES(join_key)

    """

    # --------------------------------------------------------
    # Eksekusi database
    # --------------------------------------------------------

    with engine.begin() as conn:

        conn.execute(
            text(create_table_sql)
        )

        # PENTING:
        # row hanya digunakan DI DALAM loop ini.

        for _, row in df.iterrows():

            row_data = row.to_dict()

            # ----------------------------------------------
            # Pastikan semua NaN menjadi None
            # ----------------------------------------------

            for key, value in row_data.items():

                if pd.isna(value):

                    row_data[key] = None

            # ----------------------------------------------
            # INSERT / UPDATE
            # ----------------------------------------------

            conn.execute(
                text(insert_sql),
                row_data
            )

    print(
        "NASA POWER selesai."
    )

    print(
        "Nilai -999 diubah menjadi NULL."
    )

    print(
        "Nilai NaN diubah menjadi NULL."
    )


# ============================================================
# AIRFLOW DAG
# ============================================================

with DAG(

    dag_id=
        "earthquake_multisource_etl",

    start_date=
        datetime(
            2026,
            1,
            1
        ),

    # Jalankan setiap hari pukul 03.00
    schedule=
        "0 3 * * *",

    catchup=False,

    tags=[
        "earthquake",
        "etl",
        "multisource"
    ]

) as dag:

    # --------------------------------------------------------
    # TASK 1 - USGS
    # --------------------------------------------------------

    extract_transform_load_usgs = PythonOperator(

        task_id=
            "extract_transform_load_usgs",

        python_callable=
            load_usgs

    )

    # --------------------------------------------------------
    # TASK 2 - OPEN-METEO
    # --------------------------------------------------------

    extract_transform_load_openmeteo = PythonOperator(

        task_id=
            "extract_transform_load_openmeteo",

        python_callable=
            load_openmeteo

    )

    # --------------------------------------------------------
    # TASK 3 - NASA POWER
    # --------------------------------------------------------

    extract_transform_load_nasa = PythonOperator(

        task_id=
            "extract_transform_load_nasa",

        python_callable=
            load_nasa_power

    )

    # --------------------------------------------------------
    # URUTAN ETL
    # --------------------------------------------------------

    (
        extract_transform_load_usgs
        >>
        extract_transform_load_openmeteo
        >>
        extract_transform_load_nasa
    )