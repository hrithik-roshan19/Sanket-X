from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import eccodes
except Exception as exc:
    raise SystemExit(
        "eccodes is required. Install with: python -m pip install eccodes"
    ) from exc

try:
    import certifi
except Exception as exc:
    raise SystemExit(
        "certifi is required. Install with: python -m pip install certifi"
    ) from exc


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

OUTPUT_ROOT = BACKEND_DIR / "data" / "canonical_gefs_historical"
REPORT_PATH = BACKEND_DIR / "data" / "analysis" / "gefs_m2_6_extraction.json"
CHECKPOINT_PATH = OUTPUT_ROOT / "_checkpoint.json"

AWS_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"

PRODUCT_DIR = "pgrb2sp25"
FILE_PRODUCT = "pgrb2s.0p25"

MEMBERS = ["gec00", "gep01", "gep02", "gep03", "gep04"]
LEADS = [24, 48, 72, 96, 120]
CYCLE = "00"

INDIA_LAT_MIN = 6.5
INDIA_LAT_MAX = 38.5
INDIA_LON_MIN = 66.5
INDIA_LON_MAX = 100.0

REQUIRED = {
    "2t": "temperature_c",
    "2r": "humidity_pct",
    "10u": "wind_u10_ms",
    "10v": "wind_v10_ms",
    "sp": "pressure_hpa",
    "tp": "rainfall_mm",
    "pwat": "atmospheric_moisture_kgm2",
}

UNIT_RULES = {
    "2t": {"k", "kelvin", "c", "degc", "celsius"},
    "2r": {"%", "percent", "percentage"},
    "10u": {"m s-1", "m/s", "m s**-1", "ms-1"},
    "10v": {"m s-1", "m/s", "m s**-1", "ms-1"},
    "sp": {"pa", "pascal", "pascals", "hpa", "mb", "mbar"},
    "tp": {"kg m**-2", "kg m-2", "kg/m2", "kg m^-2", "mm"},
    "pwat": {"kg m**-2", "kg m-2", "kg/m2", "kg m^-2"},
}

FILENAME_RE = re.compile(
    r"^(gec00|gep\d{2})\.t00z\.pgrb2s\.0p25\.f(\d{3})$"
)

# ecCodes keys used to identify the expected message when shortName alone
# is not sufficiently discriminating.
LEVEL_RULES = {
    "2t": {"typeOfLevel": "heightAboveGround", "level": 2},
    "2r": {"typeOfLevel": "heightAboveGround", "level": 2},
    "10u": {"typeOfLevel": "heightAboveGround", "level": 10},
    "10v": {"typeOfLevel": "heightAboveGround", "level": 10},
    "sp": {"typeOfLevel": "surface"},
    "tp": {},
    # PWAT is identified by shortName + validated units. Do not impose a
    # typeOfLevel gate because GEFS/ecCodes can expose the same physical field
    # with differing level metadata across messages/files.
    "pwat": {},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanket-X M2.6 resumable historical GEFS extraction"
    )
    parser.add_argument("--start-date", default="2020-09-23")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Process at most N dates; useful for canary testing.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most N GRIB files; useful for canary testing.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded GRIB files for debugging instead of deleting them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild an already completed partition.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Maximum download attempts per GRIB file.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=180,
        help="HTTP read timeout in seconds per download attempt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan work without downloading or writing extracted data.",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end-date must be >= start-date")
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def file_url(day: date, member: str, lead: int) -> str:
    stamp = day.strftime("%Y%m%d")
    filename = f"{member}.t00z.{FILE_PRODUCT}.f{lead:03d}"
    return (
        f"{AWS_BASE}/gefs.{stamp}/{CYCLE}/atmos/"
        f"{PRODUCT_DIR}/{filename}"
    )


def expected_filename(day: date, member: str, lead: int) -> str:
    return f"{member}.t00z.{FILE_PRODUCT}.f{lead:03d}"


def output_partition(day: date, member: str, lead: int) -> Path:
    return (
        OUTPUT_ROOT
        / f"init_date={day.isoformat()}"
        / f"member={member}"
        / f"lead_day={lead // 24}"
        / "forecast.parquet"
    )


def safe_get(handle: Any, key: str, default: Any = None) -> Any:
    try:
        return eccodes.codes_get(handle, key)
    except Exception:
        return default


def release(handle: Any) -> None:
    if handle is not None:
        try:
            eccodes.codes_release(handle)
        except Exception:
            pass


def crop(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    if not (
        values.size == latitudes.size == longitudes.size
    ):
        raise ValueError("GRIB values/latitudes/longitudes size mismatch.")

    mask = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & (latitudes >= INDIA_LAT_MIN)
        & (latitudes <= INDIA_LAT_MAX)
        & (longitudes >= INDIA_LON_MIN)
        & (longitudes <= INDIA_LON_MAX)
    )

    if not np.any(mask):
        raise ValueError("India crop is empty.")

    return np.asarray(values[mask], dtype=np.float64)


def convert_units(
    short_name: str,
    values: np.ndarray,
    units: str | None,
) -> tuple[np.ndarray, str, str]:
    unit = (units or "").strip().lower()

    if short_name == "2t":
        if unit in {"k", "kelvin"}:
            return values - 273.15, "degC", "K_to_C"
        if unit in {"c", "degc", "celsius"}:
            return values, "degC", "none"

    elif short_name == "2r":
        if unit in {"%", "percent", "percentage"}:
            return values, "%", "none"

    elif short_name in {"10u", "10v"}:
        if unit in {"m s-1", "m/s", "m s**-1", "ms-1"}:
            return values, "m s-1", "none"

    elif short_name == "sp":
        if unit in {"pa", "pascal", "pascals"}:
            return values / 100.0, "hPa", "Pa_to_hPa"
        if unit in {"hpa", "mb", "mbar"}:
            return values, "hPa", "none"

    elif short_name == "tp":
        if unit in {
            "kg m**-2",
            "kg m-2",
            "kg/m2",
            "kg m^-2",
            "mm",
        }:
            return values, "mm", "none"

    elif short_name == "pwat":
        if unit in {
            "kg m**-2",
            "kg m-2",
            "kg/m2",
            "kg m^-2",
        }:
            return values, "kg m-2", "none"

    raise ValueError(
        f"Unsupported units for {short_name}: {units!r}"
    )


def message_matches(handle: Any, short_name: str) -> bool:
    if str(safe_get(handle, "shortName", "")) != short_name:
        return False

    rule = LEVEL_RULES.get(short_name, {})
    if not rule:
        return True

    for key, expected in rule.items():
        actual = safe_get(handle, key)
        if actual is None:
            return False
        try:
            if float(actual) != float(expected):
                return False
        except (TypeError, ValueError):
            if str(actual) != str(expected):
                return False

    return True


def extract_grib(path: Path, day: date, member: str, lead: int) -> pd.DataFrame:
    recovered: dict[str, dict[str, Any]] = {}

    with path.open("rb") as fh:
        while True:
            handle = None
            try:
                handle = eccodes.codes_grib_new_from_file(fh)
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}".lower()
                if (
                    "prematureendoffile" in text
                    or "end of resource reached" in text
                    or "end of file" in text
                    or "no message found" in text
                ):
                    break
                raise

            if handle is None:
                break

            try:
                short_name = str(safe_get(handle, "shortName", "") or "")
                if short_name not in REQUIRED:
                    continue
                if not message_matches(handle, short_name):
                    continue

                canonical = REQUIRED[short_name]
                if canonical in recovered:
                    continue

                values = np.asarray(
                    eccodes.codes_get_array(handle, "values"),
                    dtype=np.float64,
                )
                latitudes = np.asarray(
                    eccodes.codes_get_array(handle, "latitudes"),
                    dtype=np.float64,
                )
                longitudes = np.asarray(
                    eccodes.codes_get_array(handle, "longitudes"),
                    dtype=np.float64,
                )

                cropped = crop(values, latitudes, longitudes)

                units = safe_get(handle, "units")
                units = str(units) if units is not None else None

                converted, canonical_units, conversion = convert_units(
                    short_name, cropped, units
                )

                finite = np.isfinite(converted)
                if not np.any(finite):
                    raise ValueError(
                        f"{short_name} India crop contains no finite values."
                    )

                recovered[canonical] = {
                    "values": converted,
                    "units": canonical_units,
                    "source_units": units,
                    "conversion": conversion,
                    "step_type": safe_get(handle, "stepType"),
                    "start_step": safe_get(handle, "startStep"),
                    "end_step": safe_get(handle, "endStep"),
                }
            finally:
                release(handle)

            if len(recovered) == len(REQUIRED):
                break

    missing = [
        name for name in REQUIRED.values()
        if name not in recovered
    ]
    if missing:
        raise ValueError(
            "Required fields missing from GRIB: " + ", ".join(missing)
        )

    # The seven fields must share the same grid after the same geographic crop.
    sizes = {
        name: int(item["values"].size)
        for name, item in recovered.items()
    }
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Required field crop sizes differ: {sizes}")

    n = next(iter(sizes.values()))

    # GEFS lead is expressed in hours; Sanket-X canonical valid_date is a
    # calendar date and lead_day is an integer day.
    valid_date = day + timedelta(days=lead // 24)
    lead_day = lead // 24

    data: dict[str, Any] = {
        "init_date": [day.isoformat()] * n,
        "valid_date": [valid_date.isoformat()] * n,
        "cycle": [CYCLE] * n,
        "member": [member] * n,
        "lead_hour": [lead] * n,
        "lead_day": [lead_day] * n,
        "latitude": np.nan,   # populated below from one reference message
        "longitude": np.nan,
    }

    # Re-read one required message's coordinates for the output grid.
    # This is intentionally not a regrid: the coordinates are the native GEFS
    # grid coordinates, cropped to the India bbox.
    ref_lat = None
    ref_lon = None
    with path.open("rb") as fh:
        while True:
            handle = None
            try:
                handle = eccodes.codes_grib_new_from_file(fh)
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}".lower()
                if (
                    "prematureendoffile" in text
                    or "end of resource reached" in text
                    or "end of file" in text
                ):
                    break
                raise
            if handle is None:
                break
            try:
                short_name = str(safe_get(handle, "shortName", "") or "")
                if short_name in REQUIRED and message_matches(handle, short_name):
                    lat = np.asarray(
                        eccodes.codes_get_array(handle, "latitudes"),
                        dtype=np.float64,
                    )
                    lon = np.asarray(
                        eccodes.codes_get_array(handle, "longitudes"),
                        dtype=np.float64,
                    )
                    mask = (
                        np.isfinite(lat)
                        & np.isfinite(lon)
                        & (lat >= INDIA_LAT_MIN)
                        & (lat <= INDIA_LAT_MAX)
                        & (lon >= INDIA_LON_MIN)
                        & (lon <= INDIA_LON_MAX)
                    )
                    if np.any(mask):
                        ref_lat = lat[mask]
                        ref_lon = lon[mask]
                        break
            finally:
                release(handle)

    if ref_lat is None or ref_lon is None:
        raise ValueError("Could not recover native GEFS crop coordinates.")

    if ref_lat.size != n or ref_lon.size != n:
        raise ValueError(
            f"Coordinate count {ref_lat.size}/{ref_lon.size} does not match "
            f"field count {n}."
        )

    data["latitude"] = ref_lat
    data["longitude"] = ref_lon

    for canonical, item in recovered.items():
        data[canonical] = item["values"]

    df = pd.DataFrame(data)

    # Keep provenance needed for exact historical reconstruction.
    df["source"] = "NOAA GEFSv12 operational archive"
    df["source_url"] = file_url(day, member, lead)
    df["gefs_grid"] = "0.25-degree native grid"
    df["interpolated"] = False
    df["regridded"] = False
    df["fabricated"] = False

    # Strict physical sanity checks; do not repair outliers.
    checks = {
        "temperature_c": (-90.0, 60.0),
        "humidity_pct": (0.0, 120.0),
        "pressure_hpa": (250.0, 1100.0),
        "rainfall_mm": (0.0, 2000.0),
        "atmospheric_moisture_kgm2": (0.0, 100.0),
    }
    for column, (low, high) in checks.items():
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} contains non-finite values.")
        if (values < low).any() or (values > high).any():
            raise ValueError(
                f"{column} contains values outside [{low}, {high}]."
            )

    for column in ("wind_u10_ms", "wind_v10_ms"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} contains non-finite values.")
        if (values.abs() > 100.0).any():
            raise ValueError(f"{column} contains implausible wind components.")

    return df


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _http_request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 180,
):
    """Open one HTTPS request using the project's CA bundle."""
    import urllib.request

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    request_headers = {
        "User-Agent": "Sanket-X-M2.6/6.0",
        "Accept": "application/octet-stream,*/*",
        "Connection": "close",
    }
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url, headers=request_headers)
    return urllib.request.urlopen(
        request,
        context=ssl_context,
        timeout=timeout,
    )


def _remote_size(
    url: str,
    *,
    timeout: int = 60,
) -> int | None:
    """Discover the S3 object size without downloading the object."""
    from urllib.error import HTTPError

    # Prefer a one-byte Range request because S3 returns the total size in
    # Content-Range. Fall back to Content-Length if the server answers 200.
    try:
        with _http_request(
            url,
            headers={"Range": "bytes=0-0"},
            timeout=timeout,
        ) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.search(r"/(\d+)$", content_range)
            if match:
                return int(match.group(1))

            content_length = response.headers.get("Content-Length")
            if content_length:
                value = int(content_length)
                if value > 1:
                    return value
    except HTTPError:
        pass
    except Exception:
        pass

    return None


def _request_range(
    url: str,
    start: int,
    end: int,
    *,
    timeout: int,
):
    """Request exactly the inclusive byte range [start, end]."""
    from urllib.error import HTTPError

    response = _http_request(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=timeout,
    )

    status = getattr(response, "status", None)
    content_range = response.headers.get("Content-Range", "")

    # For a range request against a normal S3 object we require 206.
    # A 200 for a multi-byte range means the server ignored Range; accepting
    # it would risk writing the whole object into a chunk.
    if status != 206:
        response.close()
        raise IOError(
            f"Expected HTTP 206 for Range {start}-{end}, "
            f"received {status}."
        )

    match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+)", content_range)
    if not match:
        response.close()
        raise IOError(
            f"Missing/invalid Content-Range for Range {start}-{end}: "
            f"{content_range!r}"
        )

    actual_start = int(match.group(1))
    actual_end = int(match.group(2))
    total = int(match.group(3))

    if actual_start != start or actual_end > end:
        response.close()
        raise IOError(
            f"Server returned wrong byte range: {content_range!r}; "
            f"requested {start}-{end}."
        )

    return response, actual_start, actual_end, total


def _download_with_chunked_resume(
    url: str,
    part_path: Path,
    *,
    timeout: int,
    remote_size: int,
    chunk_size: int = 8 * 1024 * 1024,
    chunk_retries: int = 5,
    backoff_seconds: float = 2.0,
) -> int:
    """Download a known-size S3 object in independently verified byte ranges.

    This is deliberately chunked rather than one long streaming request.
    If a connection dies after N bytes, only the incomplete chunk is retried.
    The already written chunks remain intact in .part.
    """
    if remote_size <= 0:
        raise IOError("Remote GRIB object has invalid size.")

    part_path.parent.mkdir(parents=True, exist_ok=True)

    existing = part_path.stat().st_size if part_path.exists() else 0

    if existing > remote_size:
        part_path.unlink(missing_ok=True)
        existing = 0

    if existing == remote_size:
        return existing

    # The partial file is allowed to end inside a chunk. We resume exactly
    # from its current byte count; the requested Range starts there.
    position = existing

    while position < remote_size:
        end = min(position + chunk_size - 1, remote_size - 1)
        last_error: Exception | None = None

        for attempt in range(1, chunk_retries + 1):
            response = None
            try:
                response, actual_start, actual_end, total = _request_range(
                    url,
                    position,
                    end,
                    timeout=timeout,
                )

                if total != remote_size:
                    raise IOError(
                        f"Remote size changed during download: "
                        f"Range total={total}, expected={remote_size}."
                    )

                expected_chunk_bytes = end - position + 1

                with response:
                    written_this_chunk = 0
                    with part_path.open("ab") as out:
                        while written_this_chunk < expected_chunk_bytes:
                            block = response.read(
                                min(
                                    1024 * 1024,
                                    expected_chunk_bytes - written_this_chunk,
                                )
                            )
                            if not block:
                                break

                            remaining = (
                                expected_chunk_bytes - written_this_chunk
                            )
                            if len(block) > remaining:
                                block = block[:remaining]

                            out.write(block)
                            written_this_chunk += len(block)

                        out.flush()
                        os.fsync(out.fileno())

                if written_this_chunk != expected_chunk_bytes:
                    raise IOError(
                        f"Incomplete byte-range download: received "
                        f"{written_this_chunk} bytes, expected "
                        f"{expected_chunk_bytes} "
                        f"for Range {position}-{end}."
                    )

                position += written_this_chunk
                break

            except Exception as exc:
                last_error = exc
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass

                # Never leave a partially written final chunk without using
                # its actual size as the next resume position.
                actual_size = (
                    part_path.stat().st_size if part_path.exists() else 0
                )
                if actual_size > remote_size:
                    part_path.unlink(missing_ok=True)
                    actual_size = 0

                position = actual_size

                if position >= remote_size:
                    break

                if attempt < chunk_retries:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))

        else:
            raise RuntimeError(
                f"Chunk download failed after {chunk_retries} attempts: "
                f"Range {position}-{end}; last error: {last_error}"
            )

        if position >= remote_size:
            break

    final_size = part_path.stat().st_size if part_path.exists() else 0
    if final_size != remote_size:
        raise IOError(
            f"Final download size mismatch: received {final_size} bytes, "
            f"expected {remote_size}."
        )

    return final_size


def download(
    url: str,
    destination: Path,
    timeout: int = 180,
    retries: int = 4,
    backoff_seconds: float = 2.0,
) -> int:
    """Reliably download a GEFS object using verified S3 byte ranges.

    Each chunk is independently requested and verified. Partial data remains
    in .part, and the final GRIB path appears only after the complete object
    size has been verified.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")

    remote_size = _remote_size(url, timeout=min(timeout, 60))
    if remote_size is None:
        raise RuntimeError(
            "Could not determine remote GEFS object size; refusing "
            "unverified streaming download."
        )

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            size = _download_with_chunked_resume(
                url,
                part_path,
                timeout=timeout,
                remote_size=remote_size,
                chunk_size=8 * 1024 * 1024,
                chunk_retries=5,
                backoff_seconds=backoff_seconds,
            )

            if size != remote_size:
                raise IOError(
                    f"Final byte count mismatch: {size} != {remote_size}"
                )

            os.replace(part_path, destination)
            return size

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"Download failed after {retries} attempts: {url}; "
        f"last error: {last_error}"
    )


def write_partition(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: incomplete Parquet must never look completed.
    with tempfile.NamedTemporaryFile(
        prefix="forecast_",
        suffix=".parquet.part",
        dir=destination.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {"completed": [], "failed": []}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": [], "failed": []}


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(checkpoint, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, CHECKPOINT_PATH)


def task_key(day: date, member: str, lead: int) -> str:
    return f"{day.isoformat()}|{CYCLE}|{member}|f{lead:03d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.retries <= 0:
        raise SystemExit("--retries must be > 0")
    if args.download_timeout <= 0:
        raise SystemExit("--download-timeout must be > 0")

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)

    # GEFSv12 operational start.
    if start < date(2020, 9, 23):
        raise SystemExit(
            "M2.6 only targets GEFSv12 operational data from 2020-09-23 onward."
        )

    days = date_range(start, end)
    if args.max_days is not None:
        if args.max_days <= 0:
            raise SystemExit("--max-days must be > 0")
        days = days[: args.max_days]

    tasks = [
        (day, member, lead)
        for day in days
        for member in MEMBERS
        for lead in LEADS
    ]

    if args.max_files is not None:
        if args.max_files <= 0:
            raise SystemExit("--max-files must be > 0")
        tasks = tasks[: args.max_files]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SANKET-X M2.6 v5 â€” HISTORICAL GEFS EXTRACTION PIPELINE")
    print("=" * 72)
    print()
    print(f"Date range       : {start} â†’ {end}")
    print(f"Dates selected   : {len(days)}")
    print(f"Cycle            : {CYCLE}Z")
    print(f"Members          : {', '.join(MEMBERS)}")
    print(f"Leads            : {LEADS}")
    print(f"Tasks selected   : {len(tasks)}")
    print(f"Output           : {OUTPUT_ROOT}")
    print("Raw GRIB         : temporary")
    print("Interpolation    : NO")
    print("Regridding       : NO")
    print("Fabrication      : NO")
    print()

    if args.dry_run:
        print("DRY RUN â€” no download or extraction will occur.")
        print(f"Planned tasks: {len(tasks)}")
        print("M2.6 DRY-RUN STATUS: PASS")
        return 0

    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed", []))

    processed = 0
    skipped = 0
    failed: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    for index, (day, member, lead) in enumerate(tasks, start=1):
        key = task_key(day, member, lead)
        destination = output_partition(day, member, lead)

        if not args.force and (key in completed or destination.exists()):
            skipped += 1
            print(
                f"[{index:05d}/{len(tasks):05d}] SKIP "
                f"{day} {member} f{lead:03d}"
            )
            continue

        filename = expected_filename(day, member, lead)
        url = file_url(day, member, lead)

        raw_dir = OUTPUT_ROOT / "_raw"
        raw_path = raw_dir / f"{day:%Y%m%d}_{member}_f{lead:03d}.grib2"

        t0 = time.perf_counter()

        try:
            print(
                f"[{index:05d}/{len(tasks):05d}] "
                f"{day} {member} f{lead:03d} | download"
            )

            if raw_path.exists():
                # Never trust a leftover final GRIB from an interrupted/older
                # downloader. Compare against the remote object size when
                # available; otherwise let extraction remain the content gate.
                remote_size = _remote_size(
                    url,
                    timeout=min(args.download_timeout, 60),
                )
                local_size = raw_path.stat().st_size
                if remote_size is not None and local_size != remote_size:
                    raw_path.unlink()
                elif local_size <= 0:
                    raw_path.unlink()

            if not raw_path.exists():
                download(
                    url,
                    raw_path,
                    timeout=args.download_timeout,
                    retries=args.retries,
                )

            raw_size_bytes = raw_path.stat().st_size
            if raw_size_bytes == 0:
                raise ValueError("Downloaded GRIB is empty.")

            # Extraction itself is the acceptance gate.
            df = extract_grib(raw_path, day, member, lead)

            write_partition(df, destination)

            # Verify the completed artifact before deleting raw GRIB.
            check = pd.read_parquet(destination)
            if check.empty:
                raise ValueError("Written Parquet partition is empty.")

            required_output = {
                "temperature_c",
                "humidity_pct",
                "wind_u10_ms",
                "wind_v10_ms",
                "pressure_hpa",
                "rainfall_mm",
                "atmospheric_moisture_kgm2",
            }
            missing_output = required_output - set(check.columns)
            if missing_output:
                raise ValueError(
                    f"Written partition missing columns: {sorted(missing_output)}"
                )

            if not args.keep_raw:
                raw_path.unlink(missing_ok=True)

            completed.add(key)
            checkpoint["completed"] = sorted(completed)
            prior_failed = checkpoint.get("failed", [])
            checkpoint["failed"] = [
                item for item in prior_failed
                if item.get("task") != key
            ]
            save_checkpoint(checkpoint)

            processed += 1
            elapsed = time.perf_counter() - t0

            print(
                f"       PASS | rows={len(df):,} | "
                f"GRIB={raw_size_bytes / (1024**2):.2f} MB | "
                f"{elapsed:.2f}s"
            )

        except Exception as exc:
            failed.append(
                {
                    "task": key,
                    "file": filename,
                    "url": url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            checkpoint.setdefault("failed", []).append(failed[-1])
            save_checkpoint(checkpoint)

            print(
                f"       FAIL | {type(exc).__name__}: {exc}"
            )

            # Do not delete failed raw data: it is needed for diagnosis.
            continue

    total_seconds = time.perf_counter() - started_all

    # Count completed artifacts for the selected task set.
    successful = 0
    for day, member, lead in tasks:
        key = task_key(day, member, lead)
        destination = output_partition(day, member, lead)
        if key in completed and destination.exists():
            successful += 1

    status = (
        "PASS"
        if successful == len(tasks) and not failed
        else "NOT_PASS"
    )

    report = {
        "benchmark": "Sanket-X M2.6 v5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days_selected": len(days),
        },
        "cycle": CYCLE,
        "members": MEMBERS,
        "lead_hours": LEADS,
        "tasks_selected": len(tasks),
        "processed_this_run": processed,
        "skipped_existing_or_completed": skipped,
        "successful_artifacts": successful,
        "failed_tasks": len(failed),
        "required_variables": REQUIRED,
        "india_bbox": {
            "latitude_min": INDIA_LAT_MIN,
            "latitude_max": INDIA_LAT_MAX,
            "longitude_min": INDIA_LON_MIN,
            "longitude_max": INDIA_LON_MAX,
        },
        "interpolation": False,
        "regridding": False,
        "nearest_neighbor": False,
        "fabricated_values": False,
        "raw_grib_deleted_after_success": not args.keep_raw,
        "download_atomic": True,
        "download_resume": True,
        "download_chunked_ranges": True,
        "download_chunk_size_bytes": 8388608,
        "remote_size_validation": True,
        "download_user_agent": "Sanket-X-M2.6/6.0",
        "download_retries": args.retries,
        "download_timeout_seconds": args.download_timeout,
        "output_root": str(OUTPUT_ROOT),
        "checkpoint": str(CHECKPOINT_PATH),
        "total_seconds": round(total_seconds, 3),
        "failures": failed,
        "status": status,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("M2.6 VERDICT")
    print("=" * 72)
    print()
    print(f"Tasks selected : {len(tasks)}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {len(failed)}")
    print(f"Skipped        : {skipped}")
    print(f"Total time     : {total_seconds:.2f}s")
    print()
    print(f"STATUS: {status}")
    print(f"Report: {REPORT_PATH}")
    print()

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
