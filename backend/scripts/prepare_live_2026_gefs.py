from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cfgrib
import numpy as np
import pandas as pd
import requests

BACKEND_DIR = Path(__file__).resolve().parents[1]
SAMPLE_FORECAST = BACKEND_DIR / "data" / "samples" / "gefs_reforecast_india_2019.parquet"
DEFAULT_OUT = BACKEND_DIR / "data" / "live_2026"

MEMBERS = ("gec00", "gep01", "gep02", "gep03", "gep04")
LEADS = (24, 48, 72, 96, 120)
PRODUCT_DIR = "pgrb2sp25"
PRODUCT_FILE = "pgrb2s.0p25"
NOMADS_ROOT = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"

RAW_NAMES = {
    "2t": "temperature_c",
    "2r": "humidity_pct",
    "10u": "wind_u10_ms",
    "10v": "wind_v10_ms",
    "sp": "pressure_hpa",
    "tp": "rainfall_mm",
    "pwat": "atmospheric_moisture_kgm2",
    "soilw": "soil_moisture_pct",
}

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 45
DOWNLOAD_RETRIES = 5
CHUNK_SIZE = 1024 * 1024
MIN_GRIB_BYTES = 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare a real 2026 operational GEFS cycle for the frozen 2019 Sanket-X model."
    )
    p.add_argument("--date", required=True, help="GEFS initialization date: YYYY-MM-DD")
    p.add_argument("--cycle", default="00Z", choices=("00Z", "06Z", "12Z", "18Z"))
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--retries", type=int, default=DOWNLOAD_RETRIES)
    return p.parse_args()


def load_training_points():
    if not SAMPLE_FORECAST.exists():
        raise FileNotFoundError(f"Missing validated 2019 sample: {SAMPLE_FORECAST}")

    df = pd.read_parquet(SAMPLE_FORECAST)
    required = {"city", "state", "latitude", "longitude", "region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"2019 sample missing columns: {sorted(missing)}")

    points = (
        df[["city", "state", "latitude", "longitude", "region"]]
        .drop_duplicates()
        .sort_values(["region", "city"])
        .reset_index(drop=True)
    )

    if points.empty or points["region"].isna().any():
        raise ValueError("Training-compatible city/region map is empty or contains null regions.")

    return points


def url_for(day, cycle, member, lead):
    hh = cycle[:2]
    filename = f"{member}.t{hh}z.{PRODUCT_FILE}.f{lead:03d}"
    return (
        f"{NOMADS_ROOT}?file={quote(filename)}"
        f"&dir=%2Fgefs.{day:%Y%m%d}%2F{hh}%2Fatmos%2F{PRODUCT_DIR}"
    )


def _remote_size(url):
    """Best-effort remote size discovery using a one-byte range request."""
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    try:
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            total = content_range.rsplit("/", 1)[1]
            if total.isdigit():
                return int(total)

        length = response.headers.get("Content-Length")
        if length and length.isdigit():
            # If server ignored Range, this is the full object size.
            return int(length)
        return None
    finally:
        response.close()


def _validate_partial(path, minimum=MIN_GRIB_BYTES):
    return path.exists() and path.stat().st_size >= minimum


def download(url, destination, retries=DOWNLOAD_RETRIES):
    """
    Resumable, atomic downloader.

    A failed transfer leaves .part intact. The next attempt resumes from
    the existing byte count. A final file is only published after basic
    size/integrity checks.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")

    expected_size = None
    try:
        expected_size = _remote_size(url)
    except requests.RequestException:
        # Size discovery is advisory; the actual transfer still has its own
        # HTTP/status/integrity checks.
        expected_size = None

    for attempt in range(1, retries + 1):
        existing = part.stat().st_size if part.exists() else 0

        if expected_size is not None and existing > expected_size:
            part.unlink(missing_ok=True)
            existing = 0

        headers = {"Accept-Encoding": "identity"}
        if existing:
            headers["Range"] = f"bytes={existing}-"

        try:
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )

            if existing and response.status_code == 200:
                # Server ignored Range. Restart from zero instead of corrupting
                # the partial file by appending the complete object.
                response.close()
                part.unlink(missing_ok=True)
                existing = 0
                response = requests.get(
                    url,
                    headers={"Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )

            response.raise_for_status()

            if existing:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Resume requested but NOAA returned HTTP {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                expected_prefix = f"bytes {existing}-"
                if not content_range.startswith(expected_prefix):
                    raise RuntimeError(
                        f"Invalid Content-Range for resume: {content_range!r}"
                    )
            else:
                if response.status_code not in (200, 206):
                    raise RuntimeError(
                        f"Unexpected NOAA HTTP status {response.status_code}"
                    )

            content_type = (response.headers.get("Content-Type") or "").lower()
            if "html" in content_type or "text/html" in content_type:
                raise RuntimeError(f"NOAA returned HTML instead of GRIB: {content_type}")

            mode = "ab" if existing else "wb"
            with part.open(mode) as handle:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)

            response.close()

            final_size = part.stat().st_size if part.exists() else 0
            if final_size < MIN_GRIB_BYTES:
                raise RuntimeError(
                    f"Downloaded object is too small: {final_size} bytes"
                )

            if expected_size is not None and final_size != expected_size:
                raise RuntimeError(
                    f"Incomplete download: received {final_size} bytes, "
                    f"expected {expected_size}"
                )

            # GRIB files begin with GRIB. Check the first four bytes before
            # publishing the file atomically.
            with part.open("rb") as handle:
                magic = handle.read(4)
            if magic != b"GRIB":
                raise RuntimeError(
                    f"Downloaded file does not start with GRIB magic: {magic!r}"
                )

            os.replace(part, destination)
            return {
                "status": "downloaded",
                "attempt": attempt,
                "bytes": final_size,
                "resumed": existing > 0,
                "expected_bytes": expected_size,
            }

        except (requests.RequestException, OSError, RuntimeError) as exc:
            try:
                response.close()
            except Exception:
                pass

            if attempt >= retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {url}\n"
                    f"Last error: {exc}"
                ) from exc

            sleep_seconds = min(30, 2 ** (attempt - 1))
            print(
                f"  download retry {attempt}/{retries - 1} after "
                f"{sleep_seconds}s: {exc}"
            )
            time.sleep(sleep_seconds)

    raise AssertionError("unreachable")


def _find_coord(ds, names):
    wanted = {n.lower() for n in names}
    for name in ds.coords:
        if name.lower() in wanted:
            return name
    for name in ds.dims:
        if name.lower() in wanted:
            return name
    return None


def _find_var(datasets, short_name):
    for ds in datasets:
        for name, var in ds.data_vars.items():
            if name == short_name or str(var.attrs.get("GRIB_shortName", "")) == short_name:
                return ds, name
    return None, None


def _sample_nearest(da, lat_name, lon_name, lat, lon):
    lat_values = np.asarray(da[lat_name].values, dtype=float)
    lon_values = np.asarray(da[lon_name].values, dtype=float)

    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError("Live GEFS coordinate arrays are not 1-D.")

    ilat = int(np.abs(lat_values - lat).argmin())
    ilon = int(np.abs(lon_values - lon).argmin())

    return float(np.asarray(
        da.isel({lat_name: ilat, lon_name: ilon}).values
    ).squeeze())


def extract_one(path, points):
    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )

    output = []
    try:
        for _, row in points.iterrows():
            item = {
                "city": row.city,
                "state": row.state,
                "region": row.region,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
            }

            for raw_name, canonical in RAW_NAMES.items():
                ds, variable_name = _find_var(datasets, raw_name)
                if ds is None:
                    raise ValueError(f"{raw_name} not found in {path.name}")

                lat_name = _find_coord(ds, ("latitude", "lat"))
                lon_name = _find_coord(ds, ("longitude", "lon"))
                if lat_name is None or lon_name is None:
                    raise ValueError(
                        f"Missing coordinates for {raw_name} in {path.name}"
                    )

                value = _sample_nearest(
                    ds[variable_name],
                    lat_name,
                    lon_name,
                    item["latitude"],
                    item["longitude"],
                )

                units = str(
                    ds[variable_name].attrs.get("GRIB_units", "")
                ).lower()

                if raw_name == "2t" and units in {"k", "kelvin"}:
                    value -= 273.15
                elif raw_name == "sp" and value > 2000:
                    value /= 100.0
                elif raw_name == "tp":
                    # GEFS accumulated precipitation is kg m-2, numerically
                    # equivalent to mm of water depth.
                    value = max(0.0, value)
                elif raw_name == "soilw":
                    # GEFS inventory reports soilw in "Proportion".
                    # Sanket-X canonical contract stores percent.
                    # Missing values are preserved; never fabricate them.
                    if np.isfinite(value):
                        if not (0.0 <= value <= 1.0):
                            raise ValueError(
                                f"Unexpected GEFS soilw proportion: {value}"
                            )
                        value *= 100.0

                item[canonical] = value

            output.append(item)
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass

    return pd.DataFrame(output)


def build_live_rows(day, cycle, points, raw_dir, keep_raw, retries):
    pieces = []

    for member in MEMBERS:
        lead_frames = {}

        for lead in LEADS:
            path = raw_dir / (
                f"{member}.t{cycle[:2]}z.{PRODUCT_FILE}.f{lead:03d}"
            )

            if not path.exists():
                print(f"  {member} f{lead:03d}: downloading")
                download(url_for(day, cycle, member, lead), path, retries)
            else:
                print(f"  {member} f{lead:03d}: using existing GRIB")

            lead_frames[lead] = extract_one(path, points)

        for lead, frame in lead_frames.items():
            frame = frame.copy()

            # The frozen 2019 model uses wind speed/direction, not raw u/v.
            # Derive both strictly from the forecast-side GEFS fields.
            u = pd.to_numeric(frame["wind_u10_ms"], errors="coerce")
            v = pd.to_numeric(frame["wind_v10_ms"], errors="coerce")
            frame["wind_speed_ms"] = np.hypot(u, v)
            frame["wind_direction_deg"] = (
                np.degrees(np.arctan2(-u, -v)) + 360.0
            ) % 360.0

            # tp is accumulated from initialization. Convert it to a
            # 24-hour bucket by differencing consecutive forecast times.
            if lead == 24:
                frame["rainfall_mm"] = frame["rainfall_mm"].clip(lower=0)
            else:
                previous = lead_frames[lead - 24][
                    ["city", "rainfall_mm"]
                ].rename(
                    columns={"rainfall_mm": "previous_rainfall_mm"}
                )

                frame = frame.merge(
                    previous,
                    on="city",
                    how="left",
                    validate="one_to_one",
                )

                frame["rainfall_mm"] = (
                    frame["rainfall_mm"] -
                    frame["previous_rainfall_mm"]
                ).clip(lower=0)

                frame = frame.drop(columns=["previous_rainfall_mm"])

            frame["member"] = member
            frame["init_date"] = pd.Timestamp(day)
            frame["valid_date"] = (
                pd.Timestamp(day) + pd.to_timedelta(lead, unit="h")
            )
            frame["lead_day"] = lead // 24
            pieces.append(frame)

        if not keep_raw:
            for path in raw_dir.glob(
                f"{member}.t{cycle[:2]}z.{PRODUCT_FILE}.f*"
            ):
                path.unlink(missing_ok=True)

    wide = pd.concat(pieces, ignore_index=True)

    long_parts = []
    for variable in (
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "rainfall_mm",
        "wind_speed_ms",
        "wind_direction_deg",
        "atmospheric_moisture_kgm2",
        "soil_moisture_pct",
    ):
        columns = [
            "city", "state", "region", "latitude", "longitude",
            "init_date", "valid_date", "lead_day", "member", variable,
        ]

        part = wide[columns].copy()
        part["variable"] = variable
        part["value"] = pd.to_numeric(
            part.pop(variable), errors="coerce"
        )
        part["value_type"] = "forecast"
        part["ensemble_member_id"] = part.pop("member")
        part["verification_status"] = "pending"
        part["source"] = (
            "NOAA GEFS operational 0.25-degree NOMADS"
        )

        part["source_url"] = [
            url_for(
                day,
                cycle,
                member,
                int(lead_day) * 24,
            )
            for member, lead_day in zip(
                part["ensemble_member_id"],
                part["lead_day"],
            )
        ]

        long_parts.append(part)

    result = pd.concat(long_parts, ignore_index=True)
    result["cycle"] = cycle
    return result


def validate_output(df, day, cycle):
    required = {
        "city", "state", "region", "latitude", "longitude",
        "init_date", "valid_date", "lead_day",
        "ensemble_member_id", "variable", "value",
        "value_type", "verification_status",
        "source", "source_url", "cycle",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Live canonical output missing columns: {sorted(missing)}"
        )

    if set(df["ensemble_member_id"]) != set(MEMBERS):
        raise ValueError(
            "Live member set does not match the frozen 5-member model contract."
        )

    if set(df["lead_day"]) != {1, 2, 3, 4, 5}:
        raise ValueError("Live lead-day coverage is incomplete.")

    if df["cycle"].nunique() != 1 or df["cycle"].iloc[0] != cycle:
        raise ValueError("Cycle mismatch.")

    init_dates = pd.to_datetime(df["init_date"]).dt.date
    if init_dates.nunique() != 1 or init_dates.iloc[0] != day:
        raise ValueError("Initialization date mismatch.")

    # GEFS operational soil-water can legitimately be missing for the
    # selected soil level/grid cells. Missing source values are preserved as
    # NaN and may be consumed by XGBoost as missing features. Every other
    # model input must be finite; we never fill, interpolate, or fabricate.
    non_finite = ~np.isfinite(df["value"].to_numpy(dtype=float))
    if non_finite.any():
        bad = df.loc[non_finite, "variable"].value_counts().to_dict()
        allowed = {"soil_moisture_pct"}
        unexpected = sorted(set(bad) - allowed)
        if unexpected:
            raise ValueError(
                "Non-finite live forecast values outside explicitly allowed "
                f"soil moisture field: {unexpected}; counts={bad}"
            )

    duplicate_keys = [
        "city", "init_date", "lead_day",
        "ensemble_member_id", "variable",
    ]
    if df.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate live canonical keys detected.")

    expected = (
        df["city"].nunique()
        * len(MEMBERS)
        * len(LEADS)
        * len(RAW_NAMES)
    )

    if len(df) != expected:
        raise ValueError(
            f"Unexpected row count {len(df)}; expected {expected}."
        )

    non_finite_counts = (
        df.loc[~np.isfinite(df["value"].to_numpy(dtype=float)), "variable"]
        .value_counts()
        .astype(int)
        .to_dict()
    )

    return {
        "rows": int(len(df)),
        "cities": int(df["city"].nunique()),
        "non_finite_value_counts": {str(k): int(v) for k, v in non_finite_counts.items()},
        "members": sorted(df["ensemble_member_id"].unique()),
        "lead_days": [int(x) for x in sorted(df["lead_day"].unique())],
        "variables": sorted(df["variable"].unique()),
        "init_date": str(day),
        "cycle": cycle,
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    args = parse_args()
    day = datetime.strptime(args.date, "%Y-%m-%d").date()

    points = load_training_points()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SANKET-X — 2026 OPERATIONAL GEFS PREPARATION")
    print("=" * 72)
    print(f"Date       : {day}")
    print(f"Cycle      : {args.cycle}")
    print(f"Cities     : {len(points)}")
    print(f"Members    : {', '.join(MEMBERS)}")
    print("Lead days  : 1–5")
    print("Model base : validated 2019 training contract")
    print("Download   : resumable + timeout + retry + atomic publish")
    print("Fabrication/interpolation/regridding: NO")
    print()

    frame = build_live_rows(
        day=day,
        cycle=args.cycle,
        points=points,
        raw_dir=raw_dir,
        keep_raw=args.keep_raw,
        retries=max(1, args.retries),
    )

    report = validate_output(frame, day, args.cycle)

    parquet_path = output_root / "live_forecast_canonical.parquet"
    report_path = output_root / "live_forecast_report.json"

    # Atomic output publication.
    parquet_part = parquet_path.with_name(parquet_path.name + ".part")
    frame.to_parquet(parquet_part, index=False)
    os.replace(parquet_part, parquet_path)

    report.update({
        "status": "PASS",
        "output": str(parquet_path),
        "model_contract": {
            "training_period": "2019",
            "members": list(MEMBERS),
            "leads_hours": list(LEADS),
            "variables": ["temperature_c", "humidity_pct", "pressure_hpa", "rainfall_mm", "wind_speed_ms", "wind_direction_deg", "atmospheric_moisture_kgm2", "soil_moisture_pct"],
        },
        "download_contract": {
            "connect_timeout_seconds": CONNECT_TIMEOUT,
            "read_timeout_seconds": READ_TIMEOUT,
            "retries": max(1, args.retries),
            "resume_enabled": True,
            "atomic_publish": True,
            "remote_size_check": True,
        },
    })

    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if not args.keep_raw:
        for path in raw_dir.glob("*.part"):
            path.unlink(missing_ok=True)
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    print(json.dumps(report, indent=2))
    print("\nLIVE GEFS PREPARATION: PASS")


if __name__ == "__main__":
    main()
