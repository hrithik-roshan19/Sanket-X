from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
import pandas as pd


# ============================================================
# Sanket-X M2.1
# Operational GEFS Smoke-Test Downloader
# ============================================================
#
# Purpose:
#   Download a SMALL, controlled slice of operational GEFS
#   from NOAA's public AWS GEFS archive and verify that the
#   downloaded GRIB files are readable.
#
# Default:
#   Date    : 2020-09-24
#   Cycle   : 00Z
#   Members : gec00, gep01, gep02, gep03, gep04
#   Leads   : 1..5
#
# Total:
#   5 members x 5 lead days = 25 GRIB files
#
# This is NOT the bulk downloader.
#
# M2.1 PASS requires:
#   downloaded_files == expected_files
#   AND
#   readable_files == expected_files
#
# IMPORTANT:
#   GEFS GRIB files contain multiple logical GRIB message groups.
#   Therefore we use cfgrib.open_datasets(), NOT
#   xarray.open_dataset(..., engine="cfgrib") directly.
# ============================================================


# ============================================================
# Configuration
# ============================================================

AWS_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"

MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]

DEFAULT_DATE = "2020-09-24"
DEFAULT_CYCLE = "00"

LEAD_DAYS = [1, 2, 3, 4, 5]

# Correct NOAA GEFS 0.25-degree product directory.
PRODUCT = "pgrb2sp25"

TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3
CHUNK_SIZE = 1024 * 1024

USER_AGENT = (
    "Sanket-X-M2.1-GEFS-Smoke-Test/1.0 "
    "(research; controlled validation)"
)


# ============================================================
# SSL
# ============================================================


def build_ssl_context() -> ssl.SSLContext:
    """
    Build a secure TLS context.

    Priority:
      1. SSL_CERT_FILE environment variable
      2. certifi CA bundle

    Certificate verification remains enabled.
    """

    custom_ca = os.environ.get("SSL_CERT_FILE")

    if custom_ca:
        ca_path = Path(custom_ca)

        if not ca_path.exists():
            raise FileNotFoundError(
                f"SSL_CERT_FILE points to a missing file: {custom_ca}"
            )

        print(f"TLS CA bundle : {ca_path}")

        return ssl.create_default_context(
            cafile=str(ca_path)
        )

    ca_path = certifi.where()

    print(f"TLS CA bundle : {ca_path}")

    return ssl.create_default_context(
        cafile=ca_path
    )


# ============================================================
# URL
# ============================================================


def build_url(
    yyyymmdd: str,
    cycle: str,
    member: str,
    forecast_hour: int,
) -> str:
    """
    Build the exact NOAA AWS GEFS object URL.

    Example:

    https://noaa-gefs-pds.s3.amazonaws.com/
    gefs.20200924/
    00/
    atmos/
    pgrb2sp25/
    gec00.t00z.pgrb2s.0p25.f024
    """

    filename = (
        f"{member}.t{cycle}z."
        f"pgrb2s.0p25.f{forecast_hour:03d}"
    )

    return (
        f"{AWS_BUCKET}/"
        f"gefs.{yyyymmdd}/"
        f"{cycle}/"
        f"atmos/"
        f"{PRODUCT}/"
        f"{filename}"
    )


# ============================================================
# File validation
# ============================================================


def is_nonempty_file(path: Path) -> bool:
    """
    Return True only when the file exists and has non-zero size.
    """

    try:
        return (
            path.exists()
            and path.is_file()
            and path.stat().st_size > 0
        )
    except OSError:
        return False


# ============================================================
# Download
# ============================================================


def download_file(
    url: str,
    destination: Path,
    ssl_context: ssl.SSLContext,
) -> dict[str, Any]:
    """
    Download one file securely.

    Existing non-empty files are reused.

    Partial downloads are stored as .part files and removed
    after failed attempts.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result: dict[str, Any] = {
        "url": url,
        "destination": str(destination),
        "downloaded": False,
        "reused_existing": False,
        "size_bytes": 0,
        "attempts": 0,
        "error": None,
    }

    # --------------------------------------------------------
    # Reuse existing file.
    # --------------------------------------------------------

    if is_nonempty_file(destination):
        size = destination.stat().st_size

        print(
            f"  EXISTS: {destination.name} "
            f"({size / 1024 / 1024:.1f} MB)"
        )

        result["downloaded"] = True
        result["reused_existing"] = True
        result["size_bytes"] = size

        return result

    # --------------------------------------------------------
    # Retry loop.
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        result["attempts"] = attempt

        temp_path = destination.with_suffix(
            destination.suffix + ".part"
        )

        temp_path.unlink(
            missing_ok=True
        )

        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
            },
            method="GET",
        )

        try:

            print(
                f"  Download attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            with urlopen(
                request,
                timeout=TIMEOUT,
                context=ssl_context,
            ) as response:

                status = getattr(
                    response,
                    "status",
                    None,
                )

                if status is not None and status != 200:
                    raise RuntimeError(
                        f"Unexpected HTTP status: {status}"
                    )

                with temp_path.open("wb") as output:

                    while True:

                        chunk = response.read(
                            CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        output.write(chunk)

            # ------------------------------------------------
            # Validate downloaded file.
            # ------------------------------------------------

            if not is_nonempty_file(temp_path):
                raise RuntimeError(
                    "Downloaded file is empty."
                )

            size = temp_path.stat().st_size

            # Replace only after successful download.
            temp_path.replace(destination)

            print(
                f"  DOWNLOADED: {destination.name} "
                f"({size / 1024 / 1024:.1f} MB)"
            )

            result["downloaded"] = True
            result["size_bytes"] = size

            return result

        except HTTPError as exc:

            result["error"] = (
                f"HTTPError {exc.code}: {exc.reason}"
            )

            print(
                f"  FAILED: {destination.name}"
            )

            print(
                f"          HTTP {exc.code}: {exc.reason}"
            )

        except URLError as exc:

            result["error"] = (
                f"URLError: {exc.reason}"
            )

            print(
                f"  FAILED: {destination.name}"
            )

            print(
                f"          URL error: {exc.reason}"
            )

        except ssl.SSLError as exc:

            result["error"] = (
                f"SSL error: {exc}"
            )

            print(
                f"  FAILED: {destination.name}"
            )

            print(
                f"          SSL error: {exc}"
            )

            # SSL errors normally are not fixed by retrying.
            break

        except TimeoutError as exc:

            result["error"] = (
                f"TimeoutError: {exc}"
            )

            print(
                f"  FAILED: {destination.name}"
            )

            print(
                "          Request timed out."
            )

        except OSError as exc:

            result["error"] = (
                f"OSError: {exc}"
            )

            print(
                f"  FAILED: {destination.name}"
            )

            print(
                f"          OS error: {exc}"
            )

        except Exception as exc:

            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"  FAILED: {destination.name}"
            )

            print(
                f"          {type(exc).__name__}: {exc}"
            )

        finally:

            temp_path.unlink(
                missing_ok=True
            )

        if attempt < MAX_RETRIES:

            print(
                f"  Retrying in "
                f"{RETRY_DELAY_SECONDS}s..."
            )

            time.sleep(
                RETRY_DELAY_SECONDS
            )

    return result


# ============================================================
# Safe metadata conversion
# ============================================================


def safe_json_value(value: Any) -> Any:
    """
    Convert common numpy/xarray/cfgrib values into JSON-safe
    Python values.

    This intentionally avoids serializing large arrays.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, bytes):
        try:
            return value.decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            return str(value)

    if hasattr(value, "item"):
        try:
            return safe_json_value(
                value.item()
            )
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        return [
            safe_json_value(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): safe_json_value(val)
            for key, val in value.items()
        }

    return str(value)


# ============================================================
# GRIB inspection
# ============================================================


def inspect_grib(
    path: Path,
) -> dict[str, Any]:
    """
    Inspect one GEFS GRIB file using cfgrib.open_datasets().

    WHY open_datasets():
        GEFS GRIB files contain multiple logical message groups,
        for example:

          - surface fields
          - heightAboveGround level 2
          - heightAboveGround level 10
          - meanSea
          - pressure-level fields

        Calling:

            xr.open_dataset(path, engine="cfgrib")

        can fail because incompatible GRIB messages are forced
        into one Dataset.

        cfgrib.open_datasets() partitions the file into compatible
        Dataset groups.

    This function validates GRIB readability and inventories
    metadata only.

    It does NOT create the Sanket-X canonical dataset.
    """

    result: dict[str, Any] = {
        "file": str(path),
        "exists": path.exists(),
        "size_bytes": (
            path.stat().st_size
            if path.exists()
            else 0
        ),
        "readable": False,
        "engine": "cfgrib",
        "reader_mode": "cfgrib.open_datasets",
        "dataset_count": 0,
        "datasets": [],
        "variables": [],
        "coordinates": [],
        "errors": [],
        "error": None,
    }

    if not path.exists():

        result["error"] = (
            "File does not exist."
        )

        return result

    if path.stat().st_size == 0:

        result["error"] = (
            "File is empty."
        )

        return result

    datasets = []

    try:

        import cfgrib

        # ----------------------------------------------------
        # IMPORTANT:
        # Do not use xr.open_dataset() here.
        # ----------------------------------------------------

        datasets = cfgrib.open_datasets(
            str(path)
        )

        if not datasets:

            result["error"] = (
                "cfgrib.open_datasets() returned "
                "no readable datasets."
            )

            return result

        result["readable"] = True
        result["dataset_count"] = len(datasets)

        all_variables: set[str] = set()
        all_coordinates: set[str] = set()

        dataset_reports: list[dict[str, Any]] = []

        for index, ds in enumerate(datasets):

            try:

                dataset_variables = [
                    str(name)
                    for name in ds.data_vars
                ]

                dataset_coordinates = [
                    str(name)
                    for name in ds.coords
                ]

                dimensions = {
                    str(key): int(value)
                    for key, value in ds.sizes.items()
                }

                # ------------------------------------------------
                # Dataset-level GRIB attributes.
                # ------------------------------------------------

                attrs: dict[str, Any] = {}

                for key, value in ds.attrs.items():

                    attrs[str(key)] = safe_json_value(
                        value
                    )

                # ------------------------------------------------
                # Variable metadata.
                # ------------------------------------------------

                variable_metadata: dict[str, Any] = {}

                for variable_name in ds.data_vars:

                    data = ds[variable_name]

                    variable_attrs: dict[str, Any] = {}

                    for key, value in data.attrs.items():

                        variable_attrs[
                            str(key)
                        ] = safe_json_value(
                            value
                        )

                    variable_metadata[
                        str(variable_name)
                    ] = {
                        "dims": [
                            str(dim)
                            for dim in data.dims
                        ],
                        "shape": [
                            int(size)
                            for size in data.shape
                        ],
                        "dtype": str(
                            data.dtype
                        ),
                        "attrs": variable_attrs,
                    }

                dataset_report = {
                    "dataset_index": index,
                    "dimensions": dimensions,
                    "coordinates": dataset_coordinates,
                    "variables": dataset_variables,
                    "attrs": attrs,
                    "variable_metadata": variable_metadata,
                }

                dataset_reports.append(
                    dataset_report
                )

                all_variables.update(
                    dataset_variables
                )

                all_coordinates.update(
                    dataset_coordinates
                )

            except Exception as exc:

                result["errors"].append(
                    {
                        "dataset_index": index,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

            finally:

                try:
                    ds.close()
                except Exception:
                    pass

        result["datasets"] = dataset_reports

        result["variables"] = sorted(
            all_variables
        )

        result["coordinates"] = sorted(
            all_coordinates
        )

        # If cfgrib returned datasets but all inspection
        # groups failed, mark the file unreadable.
        if not dataset_reports:

            result["readable"] = False

            result["error"] = (
                "cfgrib opened the GRIB file but "
                "no dataset group could be inspected."
            )

        return result

    except ImportError as exc:

        result["error"] = (
            "cfgrib is not installed/importable: "
            f"{exc}"
        )

        return result

    except Exception as exc:

        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return result

    finally:

        for ds in datasets:

            try:
                ds.close()
            except Exception:
                pass


# ============================================================
# GRIB dependency diagnostic
# ============================================================


def diagnose_cfgrib() -> dict[str, Any]:
    """
    Determine whether cfgrib and eccodes are importable.
    """

    result: dict[str, Any] = {
        "cfgrib_importable": False,
        "cfgrib_version": None,
        "eccodes_importable": False,
        "eccodes_version": None,
        "error": None,
    }

    try:

        import cfgrib

        result["cfgrib_importable"] = True

        result["cfgrib_version"] = getattr(
            cfgrib,
            "__version__",
            None,
        )

    except Exception as exc:

        result["error"] = (
            f"cfgrib: {type(exc).__name__}: {exc}"
        )

    try:

        import eccodes

        result["eccodes_importable"] = True

        result["eccodes_version"] = getattr(
            eccodes,
            "__version__",
            None,
        )

    except Exception as exc:

        existing = result.get("error")

        message = (
            f"eccodes: {type(exc).__name__}: {exc}"
        )

        result["error"] = (
            f"{existing}; {message}"
            if existing
            else message
        )

    return result


# ============================================================
# Smoke Test
# ============================================================


def run_smoke_test(
    yyyymmdd: str,
    cycle: str,
    output_dir: Path,
) -> dict[str, Any]:

    print()
    print("=" * 72)
    print("M2.1 GEFS OPERATIONAL SMOKE TEST")
    print("=" * 72)

    print()
    print(f"Date    : {yyyymmdd}")
    print(f"Cycle   : {cycle}Z")
    print(f"Members : {MEMBERS}")
    print(f"Leads   : {LEAD_DAYS}")
    print(f"Product : {PRODUCT}")

    # --------------------------------------------------------
    # TLS
    # --------------------------------------------------------

    print()
    print("Checking TLS configuration...")

    try:

        ssl_context = build_ssl_context()

        print(
            "TLS verification : ENABLED"
        )

    except Exception as exc:

        print()
        print(
            "TLS INITIALIZATION FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "stage": "M2.1",
            "status": "NOT_PASS",
            "error": (
                "TLS initialization failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    # --------------------------------------------------------
    # cfgrib dependencies
    # --------------------------------------------------------

    print()
    print("Checking GRIB reader dependencies...")

    cfgrib_status = diagnose_cfgrib()

    print(
        f"cfgrib importable  : "
        f"{cfgrib_status['cfgrib_importable']}"
    )

    print(
        f"eccodes importable : "
        f"{cfgrib_status['eccodes_importable']}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    grib_dir = (
        output_dir
        / "grib"
        / yyyymmdd
        / cycle
    )

    grib_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_reports: list[dict[str, Any]] = []

    total_files = 0
    successful_files = 0
    readable_files = 0

    # --------------------------------------------------------
    # Download + inspect
    # --------------------------------------------------------

    print()
    print(
        "Downloading and inspecting representative "
        "GRIB files..."
    )

    for member in MEMBERS:

        for lead_day in LEAD_DAYS:

            forecast_hour = lead_day * 24

            filename = (
                f"{member}.t{cycle}z."
                f"pgrb2s.0p25.f"
                f"{forecast_hour:03d}"
            )

            url = build_url(
                yyyymmdd=yyyymmdd,
                cycle=cycle,
                member=member,
                forecast_hour=forecast_hour,
            )

            destination = (
                grib_dir
                / member
                / filename
            )

            total_files += 1

            print()
            print(
                f"[{member}] "
                f"lead {lead_day} "
                f"({forecast_hour:03d}h)"
            )

            print(
                f"  URL: {url}"
            )

            download_result = download_file(
                url=url,
                destination=destination,
                ssl_context=ssl_context,
            )

            file_report: dict[str, Any] = {
                "member": member,
                "lead_day": lead_day,
                "forecast_hour": forecast_hour,
                "filename": filename,
                "url": url,
                "download": download_result,
                "inspection": None,
            }

            if not download_result.get(
                "downloaded",
                False,
            ):

                file_reports.append(
                    file_report
                )

                continue

            successful_files += 1

            # ------------------------------------------------
            # Inspect downloaded GRIB.
            # ------------------------------------------------

            inspection = inspect_grib(
                destination
            )

            file_report[
                "inspection"
            ] = inspection

            if inspection.get(
                "readable",
                False,
            ):

                readable_files += 1

                print(
                    "  GRIB inspection: READABLE"
                )

                print(
                    f"  Dataset groups: "
                    f"{inspection['dataset_count']}"
                )

                print(
                    f"  Coordinates: "
                    f"{inspection['coordinates']}"
                )

                print(
                    f"  Variables: "
                    f"{inspection['variables']}"
                )

            else:

                print(
                    "  GRIB inspection: "
                    "NOT READABLE"
                )

                if inspection.get(
                    "error"
                ):

                    print(
                        "  Error: "
                        f"{inspection['error']}"
                    )

            file_reports.append(
                file_report
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("-" * 72)
    print("SMOKE TEST FILE SUMMARY")
    print("-" * 72)

    print(
        f"Expected files : {total_files}"
    )

    print(
        f"Downloaded     : {successful_files}"
    )

    print(
        f"Readable GRIB  : {readable_files}"
    )

    # --------------------------------------------------------
    # Inventory of readable files
    # --------------------------------------------------------

    inventory: list[dict[str, Any]] = []

    for report in file_reports:

        inspection = report.get(
            "inspection"
        )

        if not isinstance(
            inspection,
            dict,
        ):
            continue

        if not inspection.get(
            "readable",
            False,
        ):
            continue

        inventory.append(
            {
                "member": report[
                    "member"
                ],
                "lead_day": report[
                    "lead_day"
                ],
                "forecast_hour": report[
                    "forecast_hour"
                ],
                "filename": report[
                    "filename"
                ],
                "url": report[
                    "url"
                ],
                "dataset_count": inspection[
                    "dataset_count"
                ],
                "coordinates": inspection[
                    "coordinates"
                ],
                "variables": inspection[
                    "variables"
                ],
                "datasets": inspection[
                    "datasets"
                ],
            }
        )

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    download_complete = (
        successful_files == total_files
    )

    grib_readable = (
        readable_files == total_files
    )

    verdict_pass = (
        download_complete
        and grib_readable
    )

    verdict = {
        "pass": verdict_pass,
        "download_complete": download_complete,
        "grib_readable": grib_readable,
        "expected_files": total_files,
        "downloaded_files": successful_files,
        "readable_files": readable_files,
    }

    # --------------------------------------------------------
    # Machine-readable report
    # --------------------------------------------------------

    report: dict[str, Any] = {
        "stage": "M2.1",
        "status": (
            "PASS"
            if verdict_pass
            else "NOT_PASS"
        ),
        "purpose": (
            "Operational GEFS small-slice "
            "download and GRIB readability smoke test"
        ),
        "source": (
            "NOAA GEFS Public Dataset on AWS"
        ),
        "source_bucket": AWS_BUCKET,
        "date": yyyymmdd,
        "cycle": cycle,
        "members": MEMBERS,
        "lead_days": LEAD_DAYS,
        "forecast_hours": [
            lead * 24
            for lead in LEAD_DAYS
        ],
        "product": PRODUCT,
        "grid_resolution": "0.25 degree",
        "tls": {
            "verification_enabled": True,
            "ca_source": (
                os.environ.get(
                    "SSL_CERT_FILE"
                )
                or certifi.where()
            ),
        },
        "grib_reader": cfgrib_status,
        "reader_mode": "cfgrib.open_datasets",
        "download_is_smoke_test_only": True,
        "verdict": verdict,
        "files": file_reports,
        "inventory": inventory,
    }

    report_file = (
        output_dir
        / "gefs_operational_smoke_test.json"
    )

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("M2.1 SMOKE TEST VERDICT")
    print("=" * 72)

    if verdict_pass:

        print()
        print("PASS")
        print()

        print(
            "All requested GEFS GRIB files "
            "downloaded successfully."
        )

        print(
            "All requested GRIB files were "
            "readable through cfgrib."
        )

    else:

        print()
        print("NOT PASS")
        print()

        if not download_complete:

            print(
                "- One or more files could not "
                "be downloaded."
            )

        if not grib_readable:

            print(
                "- One or more downloaded GRIB files "
                "could not be read."
            )

    print()
    print(
        f"Report: {report_file}"
    )

    return report


# ============================================================
# CLI
# ============================================================


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Sanket-X M2.1 operational GEFS "
            "smoke-test downloader."
        )
    )

    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--cycle",
        default=DEFAULT_CYCLE,
        choices=[
            "00",
            "06",
            "12",
            "18",
        ],
        help="GEFS cycle.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data"
            "/operational_gefs_smoke"
        ),
        help="Output directory.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate date
    # --------------------------------------------------------

    try:

        parsed = pd.Timestamp(
            args.date
        )

        yyyymmdd = parsed.strftime(
            "%Y%m%d"
        )

    except Exception as exc:

        print(
            f"Invalid date: {args.date}"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    # --------------------------------------------------------
    # Validate members/leads
    # --------------------------------------------------------

    if not MEMBERS:

        print(
            "No GEFS members configured."
        )

        return 1

    if not LEAD_DAYS:

        print(
            "No lead days configured."
        )

        return 1

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    report = run_smoke_test(
        yyyymmdd=yyyymmdd,
        cycle=args.cycle,
        output_dir=output_dir,
    )

    verdict = report.get(
        "verdict",
        {},
    )

    return (
        0
        if verdict.get(
            "pass",
            False,
        )
        else 1
    )


# ============================================================
# Entry Point
# ============================================================


if __name__ == "__main__":
    sys.exit(
        main()
    )