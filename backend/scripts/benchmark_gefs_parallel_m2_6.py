"""
Sanket-X M2.6 — 3-worker performance benchmark

Purpose:
- Benchmark controlled parallel GEFS downloads/extraction before large archive processing.
- Does NOT modify the existing M2.6 pipeline.
- Uses 3 independent GEFS files from one cycle.
- Keeps integrity checks and atomic .part downloads.
- Downloads to a separate benchmark directory.

Example:
python .\scripts\benchmark_gefs_parallel_m2_6.py `
  --date 2020-09-27 `
  --workers 3 `
  --keep-raw

The benchmark intentionally uses only 3 files.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import certifi
except ImportError:
    certifi = None

try:
    import cfgrib
except ImportError:
    cfgrib = None


BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
PRODUCT = "pgrb2sp25"
FILENAME_PRODUCT = "pgrb2s.0p25"

MEMBERS = ["gec00", "gep01", "gep02"]
LEADS = [24, 48, 72]

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "analysis"
    / "gefs_parallel_benchmark_m2_6"
)


def build_url(date: str, cycle: str, member: str, lead: int) -> str:
    ymd = date.replace("-", "")
    cc = cycle[:2]
    filename = f"{member}.t{cc}z.{FILENAME_PRODUCT}.f{lead:03d}"
    return (
        f"{BUCKET}/gefs.{ymd}/{cc}/atmos/"
        f"{PRODUCT}/{filename}"
    )


def make_ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def remote_size(url: str, timeout: int = 60) -> int | None:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "Sanket-X-M2.6-parallel-benchmark/1.0",
        },
    )
    ctx = make_ssl_context()

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                tail = content_range.rsplit("/", 1)[1]
                if tail.isdigit():
                    return int(tail)

            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                # A normal 200 response to the range request may return the
                # full object; in that case this is the remote size.
                return int(content_length)
    except Exception:
        return None

    return None


def download_one(
    date: str,
    cycle: str,
    member: str,
    lead: int,
    keep_raw: bool,
) -> dict:
    started = time.perf_counter()
    url = build_url(date, cycle, member, lead)

    raw_dir = BENCHMARK_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = url.rsplit("/", 1)[1]
    final_path = raw_dir / filename
    part_path = raw_dir / f"{filename}.part"

    expected = remote_size(url)

    if final_path.exists():
        if expected is None or final_path.stat().st_size == expected:
            if keep_raw:
                return {
                    "member": member,
                    "lead": lead,
                    "status": "SKIP",
                    "seconds": time.perf_counter() - started,
                    "bytes": final_path.stat().st_size,
                    "url": url,
                }
            final_path.unlink()

    if part_path.exists() and expected is not None:
        current = part_path.stat().st_size
        if current > expected:
            part_path.unlink()
    elif part_path.exists() and expected is None:
        part_path.unlink()

    ctx = make_ssl_context()

    for attempt in range(1, 5):
        try:
            existing = part_path.stat().st_size if part_path.exists() else 0

            headers = {
                "User-Agent": "Sanket-X-M2.6-parallel-benchmark/1.0",
                "Accept-Encoding": "identity",
            }
            if existing:
                headers["Range"] = f"bytes={existing}-"

            req = urllib.request.Request(url, headers=headers, method="GET")

            with urllib.request.urlopen(req, timeout=180, context=ctx) as response:
                status = getattr(response, "status", 200)

                # If server ignores Range, restart from zero safely.
                if existing and status == 200:
                    existing = 0
                    part_path.unlink(missing_ok=True)
                    req = urllib.request.Request(
                        url,
                        headers={
                            "User-Agent": "Sanket-X-M2.6-parallel-benchmark/1.0",
                            "Accept-Encoding": "identity",
                        },
                        method="GET",
                    )
                    with urllib.request.urlopen(
                        req, timeout=180, context=ctx
                    ) as response2:
                        with open(part_path, "wb") as out:
                            while True:
                                chunk = response2.read(1024 * 1024)
                                if not chunk:
                                    break
                                out.write(chunk)
                else:
                    mode = "ab" if existing else "wb"
                    with open(part_path, mode) as out:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)

            actual = part_path.stat().st_size
            if expected is not None and actual != expected:
                raise RuntimeError(
                    f"size mismatch: received {actual}, expected {expected}"
                )

            os.replace(part_path, final_path)

            elapsed = time.perf_counter() - started
            size_mb = final_path.stat().st_size / (1024 * 1024)

            # Readability check. cfgrib.open_datasets avoids the mixed-level
            # DatasetBuildError seen with a single xarray.open_dataset call.
            readable = False
            groups = 0
            read_error = None

            if cfgrib is not None:
                try:
                    datasets = cfgrib.open_datasets(str(final_path))
                    groups = len(datasets)
                    for ds in datasets:
                        ds.close()
                    readable = groups > 0
                except Exception as exc:
                    read_error = str(exc)

            result = {
                "member": member,
                "lead": lead,
                "status": "PASS",
                "seconds": round(elapsed, 2),
                "bytes": final_path.stat().st_size,
                "size_mb": round(size_mb, 2),
                "remote_size_bytes": expected,
                "readable": readable,
                "dataset_groups": groups,
                "url": url,
            }

            if read_error:
                result["read_error"] = read_error

            if not keep_raw:
                final_path.unlink(missing_ok=True)

            return result

        except Exception as exc:
            if attempt == 4:
                return {
                    "member": member,
                    "lead": lead,
                    "status": "FAIL",
                    "seconds": round(time.perf_counter() - started, 2),
                    "error": str(exc),
                    "url": url,
                }
            time.sleep(2 * attempt)

    return {
        "member": member,
        "lead": lead,
        "status": "FAIL",
        "seconds": round(time.perf_counter() - started, 2),
        "error": "unexpected download state",
        "url": url,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cycle", default="00Z")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 3:
        raise SystemExit("--workers must be between 1 and 3 for this benchmark")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (args.date, args.cycle, member, lead)
        for member in MEMBERS
        for lead in LEADS
    ]

    print("=" * 72)
    print("SANKET-X M2.6 — CONTROLLED PARALLEL BENCHMARK")
    print("=" * 72)
    print(f"Date       : {args.date}")
    print(f"Cycle      : {args.cycle}")
    print(f"Members    : {', '.join(MEMBERS)}")
    print(f"Leads      : {LEADS}")
    print(f"Tasks      : {len(tasks)}")
    print(f"Workers    : {args.workers}")
    print(f"cfgrib     : {'YES' if cfgrib is not None else 'NO'}")
    print(f"Output     : {BENCHMARK_DIR}")
    print()

    started = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                date,
                cycle,
                member,
                lead,
                args.keep_raw,
            )
            for date, cycle, member, lead in tasks
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            status = result["status"]
            label = (
                f"{result['member']} f{result['lead']:03d}"
            )
            seconds = result.get("seconds", 0)

            if status == "PASS":
                print(
                    f"{label:<18} PASS | "
                    f"{result.get('size_mb', 0):.2f} MB | "
                    f"{seconds:.2f}s | "
                    f"readable={result.get('readable')}"
                )
            elif status == "SKIP":
                print(f"{label:<18} SKIP | {seconds:.2f}s")
            else:
                print(
                    f"{label:<18} FAIL | "
                    f"{seconds:.2f}s | "
                    f"{result.get('error', '')}"
                )

    total = time.perf_counter() - started
    passed = sum(r["status"] == "PASS" for r in results)
    skipped = sum(r["status"] == "SKIP" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)

    downloaded = [
        r for r in results if r["status"] == "PASS" and r.get("bytes")
    ]
    total_mb = sum(r["bytes"] for r in downloaded) / (1024 * 1024)
    throughput = total_mb / total if total > 0 else 0.0

    report = {
        "date": args.date,
        "cycle": args.cycle,
        "workers": args.workers,
        "tasks": len(tasks),
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "total_seconds": round(total, 2),
        "total_downloaded_mb": round(total_mb, 2),
        "effective_throughput_mb_per_sec": round(throughput, 4),
        "results": sorted(
            results,
            key=lambda x: (x["member"], x["lead"]),
        ),
    }

    report_path = BENCHMARK_DIR / f"parallel_{args.workers}workers.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PARALLEL BENCHMARK VERDICT")
    print("=" * 72)
    print(f"Tasks             : {len(tasks)}")
    print(f"PASS              : {passed}")
    print(f"SKIP              : {skipped}")
    print(f"FAIL              : {failed}")
    print(f"Total time        : {total:.2f}s ({total/60:.2f} min)")
    print(f"Downloaded        : {total_mb:.2f} MB")
    print(f"Effective rate    : {throughput:.4f} MB/s")
    print(f"Report            : {report_path}")

    if failed == 0 and passed + skipped == len(tasks):
        print("STATUS: PASS")
        return 0

    print("STATUS: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
