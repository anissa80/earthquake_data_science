CREATE DATABASE IF NOT EXISTS earthquake_db;
USE earthquake_db;

CREATE TABLE IF NOT EXISTS earthquake_data (
    event_id VARCHAR(50) PRIMARY KEY,
    event_time DATETIME,
    magnitude DOUBLE,
    depth_km DOUBLE,
    latitude DOUBLE,
    longitude DOUBLE,
    place VARCHAR(255),
    tsunami INT,
    mag_type VARCHAR(20),
    status VARCHAR(50),
    join_key VARCHAR(120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS openmeteo_weather (
    event_id VARCHAR(50) PRIMARY KEY,
    weather_date DATE,
    latitude DOUBLE,
    longitude DOUBLE,
    temperature_mean DOUBLE,
    precipitation_sum DOUBLE,
    wind_speed_max DOUBLE,
    join_key VARCHAR(120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nasa_power_weather (
    event_id VARCHAR(50) PRIMARY KEY,
    weather_date DATE,
    latitude DOUBLE,
    longitude DOUBLE,
    temperature_mean DOUBLE,
    precipitation DOUBLE,
    wind_speed DOUBLE,
    solar_radiation DOUBLE,
    join_key VARCHAR(120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
