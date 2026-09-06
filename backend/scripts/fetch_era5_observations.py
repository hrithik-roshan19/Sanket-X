"""Build the observation-side sample for Sanket from ERA5 (via Open-Meteo).

Source
------
ERA5 reanalysis, served by the Open-Meteo Historical Weather API
(https://open-meteo.com/en/docs/historical-weather-api). ERA5 is ECMWF's reanalysis -
the standard "ground truth" that operational forecasts are verified against. Open-Meteo
redistributes it under CC-BY 4.0; underlying ERA5 is Copernicus Climate Change Service
information. No API key required; non-commercial use.

What this script does
---------------------
For the same Indian city points as the forecast sample, it pulls hourly ERA5 for a
window covering all of 2019 plus a two-week margin into 2020 (so Day-10 forecasts issued
in mid-December 2019 still have a verifying observation), aggregates to one row per
(city, date) in canonical units, and writes CSV + parquet.

Aggregation per calendar day (UTC):
  * temperature, RH, MSLP, surface pressure, TCWV, soil moisture ... daily mean
  * precipitation .................................................. daily sum
  * 10 m wind ..................................................... vector daily mean
    (mean of u,v components, then back to speed/direction - a scalar mean of degrees
    would be wrong across the 0/360 wrap)

This pairs with gefs_reforecast_india_2019.csv on (city, valid_date) at
feature-engineering time - exactly the join the plan describes.

Usage
-----
    python backend/scripts/fetch_era5_observations.py
    python backend/scripts/fetch_era5_observations.py --start 2019-06-01 --end 2019-09-30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
OUT_DIR = BACKEND_DIR / "data" / "samples"
CITIES_JSON = SCRIPT_DIR / "india_cities.json"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "soil_moisture_0_to_7cm",
    "total_column_integrated_water_vapour",
]

DEFAULT_START = "2019-01-01"
DEFAULT_END = "2020-01-15"

HTTP_RETRIES = 5
HTTP_BACKOFF = 3.0
POLITE_GAP_S = 1.5  # between city requests, to stay well under the public rate limit

_session = requests.Session()
_session.headers["User-Agent"] = "Sanket/phase2-sample (SIH 2026)"


def _get_json(params: dict) -> dict:
    last = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = _session.get(ARCHIVE_URL, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 429:
                time.sleep(HTTP_BACKOFF * (attempt + 2))
                continue
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(HTTP_BACKOFF * (attempt + 1))
    raise RuntimeError(f"archive request failed after {HTTP_RETRIES} tries: {last}")


def fetch_city(city: dict, start: str, end: str) -> pd.DataFrame:
    data = _get_json({
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "windspeed_unit": "ms",
        "timezone": "UTC",
        "cell_selection": "nearest",
    })
    h = data.get("hourly")
    if not h:
        raise RuntimeError(f"no hourly block for {city['city']}: {data.get('reason', data)}")

    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.floor("D")

    # wind -> components for a correct circular daily mean
    wd = np.deg2rad(df["wind_direction_10m"].to_numpy(dtype=float))
    ws = df["wind_speed_10m"].to_numpy(dtype=float)
    df["_u"] = -ws * np.sin(wd)   # meteorological "from" convention
    df["_v"] = -ws * np.cos(wd)

    g = df.groupby("date")
    daily = pd.DataFrame({
        "t2m_c": g["temperature_2m"].mean(),
        "rh2m_pct": g["relative_humidity_2m"].mean(),
        "precip_mm": g["precipitation"].sum(),
        "mslp_hpa": g["pressure_msl"].mean(),
        "psfc_hpa": g["surface_pressure"].mean(),
        "pwat_kgm2": g["total_column_integrated_water_vapour"].mean(),
        # ERA5 soil moisture is m3/m3 (0..1); canonical soil_moisture_pct is % volumetric
        # (0..100), matching the GEFS soilw_vol_pct column -> scale to percent here.
        "soil_moisture_pct": g["soil_moisture_0_to_7cm"].mean() * 100.0,
        "_u": g["_u"].mean(),
        "_v": g["_v"].mean(),
        "hours_in_day": g.size(),
    }).reset_index()

    daily["wspd10m_ms"] = np.sqrt(daily["_u"] ** 2 + daily["_v"] ** 2)
    daily["wdir10m_deg"] = (270.0 - np.degrees(np.arctan2(daily["_v"], daily["_u"]))) % 360.0
    daily = daily.drop(columns=["_u", "_v"])

    daily.insert(0, "city", city["city"])
    daily.insert(1, "state", city["state"])
    daily.insert(2, "region", city["region"])
    daily.insert(3, "latitude", data.get("latitude", city["lat"]))
    daily.insert(4, "longitude", data.get("longitude", city["lon"]))
    daily["source"] = "ERA5 via Open-Meteo archive-api (CC-BY 4.0; Copernicus C3S)"
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    return daily


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()

    # Name the output from the span, and fetch one year per run. The filename used to be
    # hardcoded to 2019, so `--start 2010-01-01` would silently overwrite the committed
    # 2019 observations - the exact trap that corrupted the GEFS sample once already.
    start_year = int(args.start[:4])
    span_days = (date.fromisoformat(args.end) - date.fromisoformat(args.start)).days
    if span_days > 400:
        sys.exit(f"span is {span_days} days; fetch one year per run so the output is "
                 f"named honestly (--start {start_year}-01-01 --end {start_year + 1}-01-15)")

    if not CITIES_JSON.exists():
        sys.exit(f"missing {CITIES_JSON}")
    cities = json.loads(CITIES_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ERA5 observations  {args.start} .. {args.end}  for {len(cities)} cities")
    frames = []
    for i, c in enumerate(cities, 1):
        t0 = time.time()
        d = fetch_city(c, args.start, args.end)
        frames.append(d)
        print(f"  [{i:2}/{len(cities)}] {c['city']:<20} {len(d):4} days  {time.time()-t0:4.1f}s")
        time.sleep(POLITE_GAP_S)

    full = pd.concat(frames, ignore_index=True)
    short = full[full["hours_in_day"] < 24]
    full = full.drop(columns=["hours_in_day"])

    csv_path = OUT_DIR / f"era5_observations_india_{start_year}.csv"
    pq_path = OUT_DIR / f"era5_observations_india_{start_year}.parquet"
    full.to_csv(csv_path, index=False)
    full.to_parquet(pq_path, index=False)

    print("\n" + "=" * 78)
    print(f"OBSERVATION SAMPLE  ->  {csv_path.relative_to(BACKEND_DIR)}")
    print(f"                        {pq_path.relative_to(BACKEND_DIR)}")
    print("=" * 78)
    print(f"rows        : {len(full):,}")
    print(f"cities      : {full['city'].nunique()}")
    print(f"date range  : {full['date'].min()} .. {full['date'].max()}")
    if len(short):
        print(f"note        : {len(short)} city-days had <24 hourly values (edge of archive window)")
    print("\ncanonical variable ranges:")
    for col in ["t2m_c", "rh2m_pct", "precip_mm", "mslp_hpa", "psfc_hpa",
                "pwat_kgm2", "wspd10m_ms", "wdir10m_deg", "soil_moisture_pct"]:
        s = full[col].dropna()
        print(f"  {col:18} min={s.min():8.2f}  max={s.max():8.2f}  mean={s.mean():8.2f}")


if __name__ == "__main__":
    main()
