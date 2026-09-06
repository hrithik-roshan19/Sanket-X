from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("forecastguard.live.gefs")

S3_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
NOMADS_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"

BACKEND_DIR = Path(__file__).resolve().parents[2]
CITIES_JSON = BACKEND_DIR / "scripts" / "india_cities.json"

BBOX = {"toplat": 38, "bottomlat": 6, "leftlon": 68, "rightlon": 98}

LEAD_DAYS = list(range(1, 11))
STEP_HOURS = 3
MAX_LEAD_H = 240

VAR_SPEC: dict[str, dict] = {
    "t2m_c":         {"short_name": "t2m",   "grib": ("TMP", "2 m above ground"),   "agg": "mean"},
    "rh2m_pct":      {"short_name": "r2",    "grib": ("RH", "2 m above ground"),    "agg": "mean"},
    "psfc_hpa":      {"short_name": "sp",    "grib": ("PRES", "surface"),           "agg": "mean"},
    "mslp_hpa":      {"short_name": "prmsl", "grib": ("PRMSL", "mean sea level"),   "agg": "mean"},
    "pwat_kgm2":     {"short_name": "pwat",  "grib": ("PWAT", "entire atmosphere"), "agg": "mean"},
    "soilw_vol_pct": {"short_name": "soilw", "grib": ("SOILW", "0-0.1 m below"),    "agg": "mean"},
    "u10":           {"short_name": "u10",   "grib": ("UGRD", "10 m above ground"), "agg": "mean"},
    "v10":           {"short_name": "v10",   "grib": ("VGRD", "10 m above ground"), "agg": "mean"},
    "apcp_mm":       {"short_name": "tp",    "grib": ("APCP", "surface"),
                      "agg": "sum", "step_multiple": 6},
}

SUM_VARIABLES = frozenset(c for c, spec in VAR_SPEC.items() if spec.get("agg") == "sum")

NOMADS_VARS = ["TMP", "RH", "PRES", "PRMSL", "PWAT", "APCP", "UGRD", "VGRD", "SOILW"]

HTTP_RETRIES = 5
HTTP_BACKOFF = 2.0

NOMADS_MAX_WORKERS = 6
RETRYABLE_STATUS = {302, 429, 500, 502, 503, 504}

_session = requests.Session()
_session.headers["User-Agent"] = "Sanket/phase6-live (SIH 2026; NCMRWF PS 26079)"
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=32)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


class CycleUnavailable(RuntimeError):
    pass


@dataclass
class FetchReport:
    init_date: date
    cycle_hour: str
    members: list
    transport: str = ""
    steps_expected: int = 0
    steps_fetched: int = 0
    bytes_downloaded: int = 0
    missing_steps: list = field(default_factory=list)
    steps_recovered: int = 0
    seconds: float = 0.0
    undersampled: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.steps_fetched == self.steps_expected and not self.undersampled

    @property
    def undersampled_sums(self) -> list:
        return [u for u in self.undersampled
                if any(f"/{var}(" in u for var in SUM_VARIABLES)]

    @property
    def step_completeness(self) -> float:
        if self.steps_expected <= 0:
            return 0.0
        return self.steps_fetched / self.steps_expected


def _get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    last: Optional[Exception] = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = _session.get(url, params=params, headers=headers, timeout=180)
            if r.status_code in (200, 206):
                return r
            if r.status_code in (403, 404):
                raise FileNotFoundError(f"{r.status_code} for {r.url}")
            last = requests.HTTPError(f"{r.status_code} for {r.url}")
            if r.status_code not in RETRYABLE_STATUS:
                break
        except FileNotFoundError:
            raise
        except requests.RequestException as exc:
            last = exc
        time.sleep(HTTP_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"GET failed after {HTTP_RETRIES} tries: {url} ({last})")


def _s3_key(init: date, hh: str, member: str, fh: int) -> str:
    return (f"gefs.{init:%Y%m%d}/{hh}/atmos/pgrb2sp25/"
            f"{member}.t{hh}z.pgrb2s.0p25.f{fh:03d}")


def _fetch_step_nomads(init: date, hh: str, member: str, fh: int) -> bytes:
    params = {
        "dir": f"/gefs.{init:%Y%m%d}/{hh}/atmos/pgrb2sp25",
        "file": f"{member}.t{hh}z.pgrb2s.0p25.f{fh:03d}",
        "subregion": "",
        **{f"var_{v}": "on" for v in NOMADS_VARS},
        **{k: str(v) for k, v in BBOX.items()},
    }
    r = _get(NOMADS_URL, params=params)
    if not r.content.startswith(b"GRIB"):
        raise FileNotFoundError(f"NOMADS returned no GRIB for f{fh:03d} {member}")
    return r.content


def _parse_idx(text: str) -> list:
    lines = [l for l in text.splitlines() if l.strip()]
    starts = [int(l.split(":", 3)[1]) for l in lines]
    out = []
    for i, l in enumerate(lines):
        p = l.split(":")
        out.append({
            "start": int(p[1]),
            "end": starts[i + 1] if i + 1 < len(starts) else None,
            "abbrev": p[3],
            "level": p[4],
        })
    return out


def _fetch_step_s3(init: date, hh: str, member: str, fh: int) -> bytes:
    key = _s3_key(init, hh, member, fh)
    recs = _parse_idx(_get(f"{S3_BUCKET}/{key}.idx").text)

    wanted = []
    for spec in VAR_SPEC.values():
        abbrev, level_sub = spec["grib"]
        hit = next((r for r in recs
                    if r["abbrev"] == abbrev and level_sub in r["level"]), None)
        if hit is not None:
            wanted.append(hit)

    wanted.sort(key=lambda r: r["start"])
    spans: list = []
    for r in wanted:
        if spans and spans[-1][1] == r["start"]:
            s, _ = spans.pop()
            spans.append((s, r["end"]))
        else:
            spans.append((r["start"], r["end"]))

    blobs = []
    for start, end in spans:
        rng = f"bytes={start}-{end - 1}" if end is not None else f"bytes={start}-"
        blobs.append(_get(f"{S3_BUCKET}/{key}", headers={"Range": rng}).content)
    return b"".join(blobs)


def _fetch_step(init: date, hh: str, member: str, fh: int, transport: str) -> bytes:
    if transport == "nomads":
        return _fetch_step_nomads(init, hh, member, fh)
    return _fetch_step_s3(init, hh, member, fh)


def choose_transport(init: date, hh: str, member: str = "gec00") -> str:
    try:
        _fetch_step_nomads(init, hh, member, 3)
        return "nomads"
    except (FileNotFoundError, RuntimeError) as exc:
        log.info("NOMADS unavailable for %s %sZ (%s); falling back to S3", init, hh, exc)
    try:
        _get(f"{S3_BUCKET}/{_s3_key(init, hh, member, 3)}.idx")
        return "s3"
    except Exception as exc:  # noqa: BLE001
        raise CycleUnavailable(f"cycle {init} {hh}Z not published at NOMADS or S3: {exc}")


def _extract_points(blob: bytes, cities: pd.DataFrame, scratch: Path) -> dict:
    import cfgrib

    tmp = scratch / f".step_{time.time_ns()}_{id(blob) & 0xffff}.grib2"
    tmp.write_bytes(blob)
    try:
        datasets = cfgrib.open_datasets(str(tmp), backend_kwargs={"indexpath": ""})
        lats = cities["lat"].to_numpy()
        lons = cities["lon"].to_numpy()

        out: dict = {}
        for ds in datasets:
            import xarray as xr
            sel_lat = xr.DataArray(lats, dims="city")
            sel_lon = xr.DataArray(lons % 360, dims="city")
            for col, spec in VAR_SPEC.items():
                sn = spec["short_name"]
                if sn not in ds.data_vars or col in out:
                    continue
                vals = ds[sn].sel(latitude=sel_lat, longitude=sel_lon, method="nearest")
                out[col] = np.asarray(vals.values, dtype=float).reshape(-1)
        return out
    finally:
        tmp.unlink(missing_ok=True)


def _lead_day_of(init_dt: datetime, fh: int) -> tuple:
    valid = init_dt + timedelta(hours=fh)
    day = (valid - timedelta(seconds=1)).date()
    return day, (day - init_dt.date()).days + 1


def _steps_by_day(init_dt: datetime, step_multiple: int = STEP_HOURS) -> dict:
    out: dict = {}
    for fh in range(step_multiple, MAX_LEAD_H + 1, step_multiple):
        _, lead = _lead_day_of(init_dt, fh)
        out.setdefault(lead, []).append(fh)
    return out


def _expected_samples(step_multiple: int) -> int:
    return 24 // step_multiple


def _wind_speed_dir(u: np.ndarray, v: np.ndarray):
    spd = np.sqrt(u ** 2 + v ** 2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return spd, direction


def load_cities() -> pd.DataFrame:
    if not CITIES_JSON.exists():
        raise FileNotFoundError(f"missing city table: {CITIES_JSON}")
    return pd.DataFrame(pd.read_json(CITIES_JSON))


def _recover_steps_from_s3(
    init: date,
    cycle_hour: str,
    failed: list,
    cities: pd.DataFrame,
    scratch: Path,
    step_values: dict,
    report: FetchReport,
    workers: int,
) -> list:
    def job(member: str, fh: int):
        blob = _fetch_step_s3(init, cycle_hour, member, fh)
        return member, fh, len(blob), _extract_points(blob, cities, scratch)

    recovered = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(failed)))) as pool:
        futures = {pool.submit(job, m, fh): (m, fh) for m, fh in failed}
        for fut in as_completed(futures):
            m, fh = futures[fut]
            try:
                member, hour, nbytes, values = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("S3 could not fill %s f%03d either: %s", m, fh, exc)
                continue
            step_values[(member, hour)] = values
            report.bytes_downloaded += nbytes
            report.steps_fetched += 1
            recovered.append(f"{member}/f{hour:03d}")

    if recovered:
        filled = set(recovered)
        report.missing_steps = [s for s in report.missing_steps if s not in filled]
        report.steps_recovered += len(recovered)
        log.info("S3 filled %d of %d gaps NOMADS left in %s %sZ",
                 len(recovered), len(failed), init, cycle_hour)
    return recovered


def fetch_cycle(
    init: date,
    cycle_hour: str,
    members: list,
    workers: int = 20,
    scratch: Optional[Path] = None,
    transport: Optional[str] = None,
) -> tuple:
    cities = load_cities()
    scratch = scratch or Path(BACKEND_DIR / "data" / "_live_scratch")
    scratch.mkdir(parents=True, exist_ok=True)

    transport = transport or choose_transport(init, cycle_hour, members[0])
    requested_workers = workers
    if transport == "nomads":
        workers = min(workers, NOMADS_MAX_WORKERS)
    all_steps = list(range(STEP_HOURS, MAX_LEAD_H + 1, STEP_HOURS))
    report = FetchReport(init_date=init, cycle_hour=cycle_hour, members=list(members),
                         transport=transport, steps_expected=len(all_steps) * len(members))

    t0 = time.time()
    step_values: dict = {}

    def job(member: str, fh: int):
        blob = _fetch_step(init, cycle_hour, member, fh, transport)
        return member, fh, len(blob), _extract_points(blob, cities, scratch)

    failed: list = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, m, fh): (m, fh)
                   for m in members for fh in all_steps}
        for fut in as_completed(futures):
            m, fh = futures[fut]
            try:
                member, hour, nbytes, values = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("step %s f%03d failed: %s", m, fh, exc)
                report.missing_steps.append(f"{m}/f{fh:03d}")
                failed.append((m, fh))
                continue
            step_values[(member, hour)] = values
            report.bytes_downloaded += nbytes
            report.steps_fetched += 1

    if transport == "nomads" and failed:
        _recover_steps_from_s3(init, cycle_hour, failed, cities, scratch, step_values,
                               report, requested_workers)

    report.seconds = time.time() - t0
    if not step_values:
        raise CycleUnavailable(f"no steps could be fetched for {init} {cycle_hour}Z")

    frame = _reduce_to_daily(step_values, cities, init, cycle_hour, members, transport,
                             report)
    return frame, report


def _reduce_to_daily(step_values: dict, cities: pd.DataFrame, init: date,
                     cycle_hour: str, members: list, transport: str,
                     report: Optional[FetchReport] = None) -> pd.DataFrame:
    rows = []
    n_city = len(cities)
    init_dt = datetime(init.year, init.month, init.day, int(cycle_hour), tzinfo=timezone.utc)

    by_cadence = {m: _steps_by_day(init_dt, m) for m in {STEP_HOURS, 6}}

    for member in members:
        for lead in LEAD_DAYS:
            record: dict = {}
            used_steps: dict = {}

            for col, spec in VAR_SPEC.items():
                step_mult = spec.get("step_multiple", STEP_HOURS)
                hours = by_cadence[step_mult].get(lead, [])
                if len(hours) < _expected_samples(step_mult):
                    continue
                stack = [step_values[(member, h)][col]
                         for h in hours
                         if (member, h) in step_values and col in step_values[(member, h)]]
                if not stack:
                    continue
                if report is not None and len(stack) < len(hours):
                    report.undersampled.append(
                        f"{member}/D{lead}/{col}({len(stack)}/{len(hours)})"
                    )
                arr = np.vstack(stack)
                record[col] = arr.sum(axis=0) if spec["agg"] == "sum" else arr.mean(axis=0)
                used_steps[col] = hours

            if not record:
                continue

            block = pd.DataFrame({"city": cities["city"].to_numpy()})
            block["state"] = cities["state"].to_numpy()
            block["region"] = cities["region"].to_numpy()
            block["latitude"] = cities["lat"].to_numpy()
            block["longitude"] = cities["lon"].to_numpy()
            block["init_date"] = init_dt.date()
            block["valid_date"] = init_dt.date() + timedelta(days=lead - 1)
            block["lead_day"] = lead
            block["member"] = member.replace("ge", "", 1)

            for col, values in record.items():
                block[col] = values[:n_city]

            if "t2m_c" in block:
                block["t2m_c"] = block["t2m_c"] - 273.15
            if "mslp_hpa" in block:
                block["mslp_hpa"] = block["mslp_hpa"] / 100.0
            if "psfc_hpa" in block:
                block["psfc_hpa"] = block["psfc_hpa"] / 100.0
            if "soilw_vol_pct" in block:
                block["soilw_vol_pct"] = block["soilw_vol_pct"] * 100.0
            if {"u10", "v10"}.issubset(block.columns):
                spd, drc = _wind_speed_dir(block["u10"].to_numpy(), block["v10"].to_numpy())
                block["wspd10m_ms"], block["wdir10m_deg"] = spd, drc
                block = block.drop(columns=["u10", "v10"])

            src = (f"{'NOMADS grib-filter' if transport == 'nomads' else S3_BUCKET}"
                   f" gefs.{init:%Y%m%d}/{cycle_hour}/atmos/pgrb2sp25/{member}")
            for col in ("t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "psfc_hpa",
                        "pwat_kgm2", "wspd10m_ms", "wdir10m_deg", "soilw_vol_pct"):
                if col in block.columns:
                    block[f"src_{col}"] = src
                    key = "u10" if col.startswith("wspd") or col.startswith("wdir") else col
                    block[f"srcmsg_{col}"] = ",".join(
                        f"f{h:03d}" for h in used_steps.get(key, used_steps.get(col, []))
                    )
            rows.append(block)

    if not rows:
        raise CycleUnavailable("no lead days could be reduced from the fetched steps")

    out = pd.concat(rows, ignore_index=True)
    lead_col = ["city", "state", "region", "latitude", "longitude",
                "init_date", "valid_date", "lead_day", "member"]
    value_cols = [c for c in ["t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "psfc_hpa",
                              "pwat_kgm2", "wspd10m_ms", "wdir10m_deg", "soilw_vol_pct"]
                  if c in out.columns]
    src_cols = sorted(c for c in out.columns if c.startswith(("src_", "srcmsg_")))
    return out[lead_col + value_cols + src_cols].sort_values(
        ["city", "member", "lead_day"]
    ).reset_index(drop=True)


def write_cycle_csv(frame: pd.DataFrame, init: date, cycle_hour: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gefs_operational_forecast_{init:%Y%m%d}_{cycle_hour}z.csv"
    frame.to_csv(path, index=False)
    return path


def latest_expected_cycle(now: Optional[datetime] = None, lag_hours: float = 5.5,
                          cycles: Optional[list] = None) -> tuple:
    now = now or datetime.now(timezone.utc)
    cycles = cycles or ["00", "06", "12", "18"]
    hours = sorted(int(c) for c in cycles)

    probe = now - timedelta(hours=lag_hours)
    day = probe.date()
    for _ in range(3):
        for h in reversed(hours):
            issued = datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc)
            if issued <= now - timedelta(hours=lag_hours):
                return day, f"{h:02d}"
        day = day - timedelta(days=1)
    raise CycleUnavailable("no cycle old enough to be published")


def recent_cycles(count: int, now: Optional[datetime] = None, lag_hours: float = 5.5,
                  cycles: Optional[list] = None, stride_hours: int = 24) -> list:
    init, hh = latest_expected_cycle(now, lag_hours, cycles)
    anchor = datetime(init.year, init.month, init.day, int(hh), tzinfo=timezone.utc)
    out = []
    for i in range(count):
        t = anchor - timedelta(hours=stride_hours * i)
        out.append((t.date(), f"{t.hour:02d}"))
    return out
