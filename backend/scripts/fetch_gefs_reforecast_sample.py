"""Build a real forecast-side sample for Sanket from the NOAA GEFSv12 reforecast.

Source
------
NOAA GEFSv12 reforecast, public S3 bucket ``noaa-gefs-retrospective`` (US Government
work, public domain). 2000-2019, one 00 UTC cycle per day, 5 members (``c00`` control +
``p01``-``p04``), 0.25 deg global grid, 3-hourly steps. Documentation:
https://registry.opendata.aws/noaa-gefs-reforecast/  and the bucket's
``Description_of_reforecast_data.pdf``.

What this script does
---------------------
For a fixed set of Indian city points, a set of 00 UTC initialisation dates in 2019, and
all 5 members, it pulls the Day 1-10 forecast for eight surface variables, extracts the
value at each city by nearest-grid-point, aggregates the 3-hourly data to a daily value
per lead day, converts to canonical units, and writes one wide CSV/parquet row per
``(city, init_date, member, valid_date)``.

Only the GRIB2 messages actually needed are downloaded, via HTTP ``Range`` requests keyed
off each file's ``.idx`` sidecar - not the whole 30-70 MB files.

NOTHING here is synthetic. Every value traces to a specific GRIB2 message in the public
archive (the ``source_grib`` / ``source_msg`` columns record which one).

Lead-time coverage is variable-dependent and is reported honestly at the end:
  * tmp_2m, spfh_2m, pres_sfc, pres_msl, apcp_sfc, pwat_eatm ... Day 1-10
  * ugrd_hgt / vgrd_hgt (10 m wind) .................................. Day 1-5 only
  * soilw_bgrnd (soil moisture) ..................................... ~Day 1-3 only
The short-coverage variables are kept for the leads they do cover; the ML layer already
handles variables with sparse rows rather than assuming full Day 1-10.

Usage
-----
    python backend/scripts/fetch_gefs_reforecast_sample.py            # full sample
    python backend/scripts/fetch_gefs_reforecast_sample.py --max-inits 2 --members c00
    python backend/scripts/fetch_gefs_reforecast_sample.py --resume   # skip done inits

Runs from anywhere; output always lands in ``backend/data/samples/``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore")  # LibreSSL/urllib3 + cfgrib chatter

import numpy as np
import pandas as pd
import requests
import xarray as xr

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

BUCKET = "https://noaa-gefs-retrospective.s3.amazonaws.com"
REFORECAST_YEAR = 2019
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
OUT_DIR = BACKEND_DIR / "data" / "samples"
CITIES_JSON = SCRIPT_DIR / "india_cities.json"

ALL_MEMBERS = ["c00", "p01", "p02", "p03", "p04"]
LEAD_DAYS = list(range(1, 11))  # Day 1 .. Day 10

# Initialisation pattern: one 00 UTC cycle per listed month-day. Fortnightly through the
# monsoon (Jun-Sep), monthly otherwise, so each year sampled spans the full seasonal cycle
# including the regime where medium-range busts over India matter most.
#
# The same month-days are used for every year, which keeps the seasonal sampling aligned
# across years - the point of adding years is more *independent monsoons*, not a denser
# sample of one. Nearby initialisations see largely the same weather, so breadth across
# years buys far more than density within a year.
INIT_MMDD = [
    "01-09", "02-13", "03-13", "04-10", "05-08",
    "06-05", "06-19", "07-03", "07-17", "07-31",
    "08-14", "08-28", "09-11", "09-25", "10-16",
    "11-13", "12-11",
]

# The reforecast archive runs 2000-2019 (verified against the bucket: every year in that
# span answers 200 for both the GRIB and its .idx). Anything outside it does not exist.
ARCHIVE_YEARS = range(2000, 2020)


def init_dates_for(years) -> list:
    return [f"{y}-{md}" for y in years for md in INIT_MMDD]


def parse_years(spec: str) -> list:
    """`2019`, `2010-2019` or `2011,2014,2019` -> a sorted list of years."""
    out: set = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    bad = sorted(y for y in out if y not in ARCHIVE_YEARS)
    if bad:
        raise ValueError(f"outside the GEFSv12 reforecast archive (2000-2019): {bad}")
    return sorted(out)


INIT_DATES = init_dates_for([REFORECAST_YEAR])   # default: the original 2019 sample

# GRIB2 file-prefix -> how to read and canonicalise it.
#   short_name : cfgrib variable key after decode
#   level_key / level_val : filter when a file holds several levels (wind, soil)
#   max_lead_h : messages past this are absent in the archive for this variable
#   accum : APCP-style bucketed accumulation rather than instantaneous samples
VAR_SPEC: dict[str, dict] = {
    "tmp_2m":       {"short_name": "t2m",   "max_lead_h": 240, "accum": False},
    "spfh_2m":      {"short_name": "sh2",   "max_lead_h": 240, "accum": False},
    "pres_sfc":     {"short_name": "sp",    "max_lead_h": 240, "accum": False},
    "pres_msl":     {"short_name": "msl",   "max_lead_h": 240, "accum": False},
    "pwat_eatm":    {"short_name": "pwat",  "max_lead_h": 240, "accum": False},
    "apcp_sfc":     {"short_name": "tp",    "max_lead_h": 240, "accum": True},
    "ugrd_hgt":     {"short_name": "u10",   "max_lead_h": 120, "accum": False,
                     "level_key": "heightAboveGround", "level_val": 10.0},
    "vgrd_hgt":     {"short_name": "v10",   "max_lead_h": 120, "accum": False,
                     "level_key": "heightAboveGround", "level_val": 10.0},
    "soilw_bgrnd":  {"short_name": "soilw", "max_lead_h": 72,  "accum": False,
                     "level_key": "depthBelowLandLayer", "level_val": 0.0},
}

HTTP_RETRIES = 5
HTTP_BACKOFF = 2.0
DOWNLOAD_WORKERS = 20

_session = requests.Session()
_session.headers["User-Agent"] = "Sanket/phase2-sample (SIH 2026)"


# --------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------

def _get(url: str, headers: dict | None = None) -> requests.Response:
    last = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = _session.get(url, headers=headers, timeout=120)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                r.raise_for_status()
            last = requests.HTTPError(f"{r.status_code} for {url}")
        except (requests.RequestException,) as exc:  # noqa: PERF203
            last = exc
        time.sleep(HTTP_BACKOFF * (attempt + 1))
    raise RuntimeError(f"GET failed after {HTTP_RETRIES} tries: {url}\n  last error: {last}")


def key_for(var_prefix: str, init: str, member: str) -> str:
    """S3 key for a Days:1-10 reforecast GRIB2 file."""
    ymd = init.replace("-", "")
    cyc = f"{ymd}00"
    return (
        f"GEFSv12/reforecast/{init[:4]}/{cyc}/{member}/Days:1-10/"
        f"{var_prefix}_{cyc}_{member}.grib2"
    )


# --------------------------------------------------------------------------------------
# .idx parsing + message selection
# --------------------------------------------------------------------------------------

class IdxRecord:
    __slots__ = ("msg", "start", "end", "abbrev", "level", "fcst")

    def __init__(self, msg, start, end, abbrev, level, fcst):
        self.msg, self.start, self.end = msg, start, end
        self.abbrev, self.level, self.fcst = abbrev, level, fcst

    def __repr__(self):  # debug aid
        return f"<msg{self.msg} {self.abbrev}/{self.level}/{self.fcst} [{self.start}:{self.end}]>"


def parse_idx(text: str) -> list[IdxRecord]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    starts = [int(ln.split(":", 3)[1]) for ln in lines]
    recs: list[IdxRecord] = []
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        msg = int(parts[0])
        start = int(parts[1])
        end = starts[i + 1] if i + 1 < len(starts) else None  # None -> to EOF
        abbrev = parts[3]
        level = parts[4]
        fcst = parts[5] if len(parts) > 5 else ""
        recs.append(IdxRecord(msg, start, end, abbrev, level, fcst))
    return recs


def _fcst_hour(fcst: str) -> int | None:
    """'87 hour fcst' -> 87 ; '18-24 hour acc fcst' -> 24 (end of window)."""
    tok = fcst.split()
    if not tok:
        return None
    head = tok[0]
    if "-" in head:
        try:
            return int(head.split("-")[1])
        except ValueError:
            return None
    try:
        return int(head)
    except ValueError:
        return None


def _accum_window(fcst: str) -> tuple[int, int] | None:
    tok = fcst.split()
    if not tok or "-" not in tok[0]:
        return None
    a, b = tok[0].split("-")
    return int(a), int(b)


def select_for_day(recs: list[IdxRecord], spec: dict, lead_day: int) -> list[IdxRecord]:
    """Messages whose valid time falls in ((lead_day-1)*24, lead_day*24] hours.

    Instantaneous vars: the 3-hourly samples in the window (typically 8, at
    +3h..+24h relative to the day start). Accumulation vars: a minimal set of
    buckets that tile the 24 h exactly, preferring 6-hourly then 3-hourly.
    """
    lo, hi = (lead_day - 1) * 24, lead_day * 24
    if lo >= spec["max_lead_h"]:
        return []

    if spec.get("level_key") == "heightAboveGround":
        recs = [r for r in recs if r.level.strip() == "10 m above ground"]
    elif spec.get("level_key") == "depthBelowLandLayer":
        recs = [r for r in recs if r.level.strip() == "0-0.1 m below ground"]

    if not spec.get("accum"):
        out = []
        for r in recs:
            h = _fcst_hour(r.fcst)
            if h is not None and lo < h <= hi and h <= spec["max_lead_h"]:
                out.append(r)
        return out

    # Accumulation: greedily tile [lo, hi] with non-overlapping buckets.
    buckets = []
    for r in recs:
        win = _accum_window(r.fcst)
        if win and lo <= win[0] and win[1] <= hi and win[1] <= spec["max_lead_h"]:
            buckets.append((win, r))
    buckets.sort(key=lambda x: (x[0][0], -(x[0][1] - x[0][0])))  # earlier start, wider first
    chosen, cursor = [], lo
    while cursor < hi:
        step = next((b for b in buckets if b[0][0] == cursor), None)
        if step is None:
            break
        chosen.append(step[1])
        cursor = step[0][1]
    return chosen if cursor == hi else []


def merge_ranges(recs: Iterable[IdxRecord]) -> list[tuple[int, int | None, list[IdxRecord]]]:
    """Collapse adjacent byte ranges so contiguous messages download in one GET."""
    recs = sorted(recs, key=lambda r: r.start)
    spans: list[tuple[int, int | None, list[IdxRecord]]] = []
    for r in recs:
        if spans and spans[-1][1] == r.start:
            s, _, group = spans.pop()
            spans.append((s, r.end, group + [r]))
        else:
            spans.append((r.start, r.end, [r]))
    return spans


# --------------------------------------------------------------------------------------
# GRIB decode + point extraction
# --------------------------------------------------------------------------------------

def extract_points(grib_bytes: bytes, spec: dict, cities: pd.DataFrame) -> pd.DataFrame:
    """Decode concatenated GRIB2 messages, sample each city by nearest grid point.

    Returns long rows: city, valid_time, value (raw units, pre-conversion).
    """
    tmp = OUT_DIR / f".decode_{int(time.time()*1e6)}_{id(grib_bytes) & 0xffff}.grib2"
    tmp.write_bytes(grib_bytes)
    try:
        # Level selection already happened at the .idx stage (select_for_day keeps only
        # the wanted level's messages), so no cfgrib filter_by_keys is needed here -
        # and passing one breaks when a single message makes the level coord scalar.
        ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = ds[spec["short_name"]]

        lats = xr.DataArray(cities["lat"].to_numpy(), dims="city")
        lons = xr.DataArray(cities["lon"].to_numpy() % 360, dims="city")
        pts = var.sel(latitude=lats, longitude=lons, method="nearest")

        if "step" not in pts.dims:
            pts = pts.expand_dims("step")
        vt = pts["valid_time"].values
        vt = np.atleast_1d(vt)

        rows = []
        vals = pts.transpose("step", "city").values
        cnames = cities["city"].to_numpy()
        for si in range(vals.shape[0]):
            for ci in range(vals.shape[1]):
                rows.append((cnames[ci], pd.Timestamp(vt[si]), float(vals[si, ci])))
        return pd.DataFrame(rows, columns=["city", "valid_time", "raw_value"])
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------
# Canonicalisation
# --------------------------------------------------------------------------------------

def rh_from_specific_humidity(q: np.ndarray, t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Relative humidity [%] from specific humidity, temperature, pressure.

    Bolton (1980) saturation vapour pressure; standard q -> vapour-pressure inversion.
    """
    e = q * p_pa / (0.622 + 0.378 * q)
    es = 611.2 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def wind_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spd = np.sqrt(u**2 + v**2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return spd, direction


# canonical wide-column name per grib prefix (post daily-aggregation, pre final unit fix)
DAILY_AGG = {  # how to collapse 3-hourly -> one value per lead day
    "tmp_2m": "mean", "spfh_2m": "mean", "pres_sfc": "mean", "pres_msl": "mean",
    "pwat_eatm": "mean", "apcp_sfc": "sum", "ugrd_hgt": "mean", "vgrd_hgt": "mean",
    "soilw_bgrnd": "mean",
}


# --------------------------------------------------------------------------------------
# Main pull
# --------------------------------------------------------------------------------------

def pull_one_file(var_prefix: str, init: str, member: str, cities: pd.DataFrame) -> pd.DataFrame:
    """All covered lead days for one (variable, init, member). Long rows:
    city, init_date, member, lead_day, valid_date, source_grib, source_msgs, <daily value>.
    """
    spec = VAR_SPEC[var_prefix]
    key = key_for(var_prefix, init, member)
    try:
        idx_txt = _get(f"{BUCKET}/{key}.idx").text
    except RuntimeError as exc:
        print(f"    ! missing idx  {key}  ({exc})", file=sys.stderr)
        return pd.DataFrame()
    recs = parse_idx(idx_txt)

    per_day = []
    for lead in LEAD_DAYS:
        chosen = select_for_day(recs, spec, lead)
        if not chosen:
            continue
        frames = []
        for start, end, group in merge_ranges(chosen):
            rng = f"bytes={start}-{end - 1}" if end is not None else f"bytes={start}-"
            blob = _get(f"{BUCKET}/{key}", headers={"Range": rng}).content
            frames.append(extract_points(blob, spec, cities))
        long = pd.concat(frames, ignore_index=True)
        # Day k is defined as valid at init + k days; aggregate every 3-hourly sample in
        # the hour window ((k-1)*24, k*24] into one value per city for that lead day.
        # (Grouping on the raw valid_time's calendar day would split the +24h sample,
        # which lands at 00:00 of the next day, into the following lead day.)
        agg = DAILY_AGG[var_prefix]
        daily = long.groupby("city", as_index=False)["raw_value"].agg(agg)
        # Day k is built from forecast hours ((k-1)*24, k*24]. For a 00 UTC init those
        # valid times are calendar day init+(k-1) - hour 3 through hour 24 all fall on the
        # init day itself, with only the closing +24 h sample landing at the next
        # midnight. Labelling it init+k (as this script originally did) put every forecast
        # a day later than its own contents, so each row was verified against the wrong
        # day's observation: measured over the 2019 sample that cost ~8% MAE on average,
        # and 15-16% on pressure and precipitable water.
        daily["valid_date"] = pd.Timestamp(init) + pd.Timedelta(days=lead - 1)
        daily["init_date"] = pd.Timestamp(init)
        daily["member"] = member
        daily["lead_day"] = lead
        daily["source_grib"] = key
        daily["source_msgs"] = ",".join(str(r.msg) for r in chosen)
        per_day.append(daily)

    if not per_day:
        return pd.DataFrame()
    out = pd.concat(per_day, ignore_index=True)
    return out.rename(columns={"raw_value": var_prefix})


def build(cities: pd.DataFrame, inits: list, members: list, resume: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part_dir = OUT_DIR / "_gefs_parts"
    part_dir.mkdir(exist_ok=True)

    for init in inits:
        part_path = part_dir / f"{init}.parquet"
        if resume and part_path.exists():
            print(f"= {init}  (cached, skipping)")
            continue
        t0 = time.time()
        print(f"* {init}  members={','.join(members)}")

        merged: pd.DataFrame | None = None
        jobs = [(vp, init, m) for m in members for vp in VAR_SPEC]
        results: dict[tuple[str, str], pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            fut = {pool.submit(pull_one_file, vp, init, m, cities): (vp, m) for vp, init_, m in jobs}
            for f in as_completed(fut):
                vp, m = fut[f]
                df = f.result()
                results[(vp, m)] = df
                n = 0 if df.empty else len(df)
                print(f"    {vp:<12} {m}  rows={n}")

        # assemble wide table keyed by (city, init_date, member, lead_day, valid_date):
        # first stack every member for a given variable, then merge one column per
        # variable across the shared keys (no column-name collisions that way).
        keys = ["city", "init_date", "member", "lead_day", "valid_date"]
        for vp in VAR_SPEC:
            parts = [df for (v, _m), df in results.items() if v == vp and not df.empty]
            if not parts:
                continue
            stacked = pd.concat(parts, ignore_index=True)
            stacked = stacked[keys + ["source_grib", "source_msgs", vp]].rename(
                columns={"source_grib": f"src_{vp}", "source_msgs": f"srcmsg_{vp}"}
            )
            merged = stacked if merged is None else merged.merge(stacked, on=keys, how="outer")

        if merged is None or merged.empty:
            print(f"    (no data for {init})")
            continue

        merged = _canonicalise(merged, cities)
        merged.to_parquet(part_path, index=False)
        print(f"  -> {part_path.name}  {len(merged):,} rows  {time.time()-t0:,.0f}s")

    _finalise(cities, part_dir, sorted({int(i[:4]) for i in inits}))


def _canonicalise(df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Raw GEFS columns -> canonical units and derived fields."""
    out = df.copy()

    if "tmp_2m" in out:
        out["t2m_c"] = out["tmp_2m"] - 273.15
    if {"spfh_2m", "tmp_2m", "pres_sfc"}.issubset(out.columns):
        out["rh2m_pct"] = rh_from_specific_humidity(
            out["spfh_2m"].to_numpy(), out["tmp_2m"].to_numpy(), out["pres_sfc"].to_numpy()
        )
    if "pres_msl" in out:
        out["mslp_hpa"] = out["pres_msl"] / 100.0
    if "pres_sfc" in out:
        out["psfc_hpa"] = out["pres_sfc"] / 100.0
    if "pwat_eatm" in out:
        out["pwat_kgm2"] = out["pwat_eatm"]
    if "apcp_sfc" in out:
        out["apcp_mm"] = out["apcp_sfc"]  # kg m-2 == mm
    if {"ugrd_hgt", "vgrd_hgt"}.issubset(out.columns):
        spd, drc = wind_speed_dir(out["ugrd_hgt"].to_numpy(), out["vgrd_hgt"].to_numpy())
        out["wspd10m_ms"], out["wdir10m_deg"] = spd, drc
    if "soilw_bgrnd" in out:
        out["soilw_vol_pct"] = out["soilw_bgrnd"] * 100.0

    meta = cities.rename(columns={"lat": "latitude", "lon": "longitude"})
    out = out.merge(meta[["city", "state", "region", "latitude", "longitude"]], on="city", how="left")

    canon = [
        "city", "state", "region", "latitude", "longitude",
        "init_date", "valid_date", "lead_day", "member",
        "t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "psfc_hpa",
        "pwat_kgm2", "wspd10m_ms", "wdir10m_deg", "soilw_vol_pct",
    ]
    src_cols = sorted(c for c in out.columns if c.startswith(("src_", "srcmsg_")))
    keep = [c for c in canon if c in out.columns] + src_cols
    out = out[keep].sort_values(["city", "init_date", "member", "lead_day"]).reset_index(drop=True)
    out["init_date"] = pd.to_datetime(out["init_date"]).dt.date
    out["valid_date"] = pd.to_datetime(out["valid_date"]).dt.date
    return out


def _finalise(cities: pd.DataFrame, part_dir: Path, years: list) -> None:
    """Write one CSV/parquet per year, from that year's cached parts.

    Per year, and only for the years this run asked for, on purpose. An earlier version
    globbed every part into a single file named 2019 - so a run limited to one member
    silently rewrote the committed 2019 sample 1,440 rows short (36 cities x 4 missing
    members x 10 leads), and `git status` stayed clean because the CSV is gitignored.
    Scoping the write to the requested years makes that impossible.
    """
    for year in years:
        parts = sorted(part_dir.glob(f"{year}-*.parquet"))
        if not parts:
            print(f"No parts for {year}; nothing to finalise.", file=sys.stderr)
            continue
        _write_year(pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True), year)


def _write_year(full: pd.DataFrame, year: int) -> None:
    csv_path = OUT_DIR / f"gefs_reforecast_india_{year}.csv"
    pq_path = OUT_DIR / f"gefs_reforecast_india_{year}.parquet"
    full.to_csv(csv_path, index=False)
    full.to_parquet(pq_path, index=False)

    print("\n" + "=" * 78)
    print(f"FORECAST SAMPLE  ->  {csv_path.relative_to(BACKEND_DIR)}")
    print(f"                     {pq_path.relative_to(BACKEND_DIR)}")
    print("=" * 78)
    print(f"rows            : {len(full):,}")
    print(f"cities          : {full['city'].nunique()}")
    print(f"init cycles      : {full['init_date'].nunique()}  "
          f"({full['init_date'].min()} .. {full['init_date'].max()})")
    print(f"members         : {sorted(full['member'].unique())}")
    print(f"valid dates      : {full['valid_date'].min()} .. {full['valid_date'].max()}")
    print("\nnon-null value counts by canonical variable and lead day:")
    value_cols = [c for c in ["t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "psfc_hpa",
                              "pwat_kgm2", "wspd10m_ms", "wdir10m_deg", "soilw_vol_pct"]
                  if c in full.columns]
    cov = full.groupby("lead_day")[value_cols].count()
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print(cov.to_string())
    print("\nEvery row's values are reproducible from the src_<var> / srcmsg_<var> columns.")


# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default=str(REFORECAST_YEAR), metavar="SPEC",
                    help="which reforecast years to sample: 2019, 2010-2019, or "
                         "2011,2014,2019. Archive covers 2000-2019 (default: 2019).")
    ap.add_argument("--inits", default=None, metavar="DATES",
                    help="explicit comma list of YYYY-MM-DD init dates, instead of the "
                         "seasonal pattern. Used by CI to split a year across parallel "
                         "jobs; --years is ignored when this is given.")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="take every Nth init date starting at I (1-based), for splitting "
                         "a fetch across parallel CI jobs. Strided rather than contiguous "
                         "so a failed shard thins the seasonal sample evenly instead of "
                         "removing a whole quarter.")
    ap.add_argument("--max-inits", type=int, default=None, help="use only the first N init dates")
    ap.add_argument("--members", default=",".join(ALL_MEMBERS),
                    help="comma list, subset of c00,p01,p02,p03,p04")
    ap.add_argument("--resume", action="store_true", help="skip init dates already in _gefs_parts/")
    ap.add_argument("--list-only", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    if not CITIES_JSON.exists():
        sys.exit(f"missing {CITIES_JSON} - run the city-extraction step first")
    cities = pd.DataFrame(json.loads(CITIES_JSON.read_text()))

    if args.inits:
        inits = [d.strip() for d in args.inits.split(",") if d.strip()]
        bad = [d for d in inits if len(d) != 10 or int(d[:4]) not in ARCHIVE_YEARS]
        if bad:
            sys.exit(f"bad or out-of-archive init dates: {bad}")
        years = sorted({int(d[:4]) for d in inits})
    else:
        try:
            years = parse_years(args.years)
        except ValueError as exc:
            sys.exit(str(exc))
        inits = init_dates_for(years)
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/", 1))
            if not (1 <= i <= n):
                raise ValueError
        except ValueError:
            sys.exit(f"--shard must look like 1/4, got {args.shard!r}")
        inits = inits[i - 1::n]
    if args.max_inits is not None:
        inits = inits[: args.max_inits]
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    bad = set(members) - set(ALL_MEMBERS)
    if bad:
        sys.exit(f"unknown members: {sorted(bad)}")

    est_files = len(inits) * len(members) * len(VAR_SPEC)
    print(f"years={years[0]}..{years[-1]} ({len(years)})  "
          f"cities={len(cities)}  inits={len(inits)}  members={len(members)}  "
          f"variables={len(VAR_SPEC)}  -> ~{est_files} GRIB files, "
          f"~{est_files * 8} range GETs")
    if args.list_only:
        for i in inits:
            print("  init", i)
        return

    build(cities, inits, members, resume=args.resume)


if __name__ == "__main__":
    main()
