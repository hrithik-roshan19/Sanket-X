"""How much does the ground truth itself disagree with an independent reanalysis?

Sanket derives every bust label from ERA5: a forecast busts when |forecast - observed|
exceeds that variable's own p90 historical error. That makes ERA5 the arbiter of what
counts as being wrong - so it is worth knowing how much ERA5 is itself uncertain.

This script answers that with a number instead of an assurance. It pulls the same city
points from two independent reanalyses and reports how far apart they are:

  * **ERA5** (ECMWF) via the Open-Meteo archive - the product Sanket already verifies
    against, fetched exactly as `fetch_era5_observations.py` fetches it.
  * **MERRA-2** (NASA GMAO) via the NASA POWER daily point API - a genuinely separate
    model, assimilation system and observing-system treatment. No API key needed.

The comparison is deliberately reported against the live bust thresholds, because the
disagreement only matters relative to the error size being called a bust.

NOTHING here trains or scores anything, and it must not: MERRA-2 is a *different product*,
so using it as a label alongside ERA5 would make the labels mutually inconsistent - the
same reason `train_pipeline.py` withholds provisional observations from training. This is
a measurement of label uncertainty, not a source of labels.

Usage
-----
    python -m scripts.compare_verification_products
    python -m scripts.compare_verification_products --start 2025-01-01 --end 2026-08-20
    python -m scripts.compare_verification_products --max-cities 5     # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
# NOT data/samples: conftest's iter_sample_files() parametrises the parser tests over
# everything in there, so writing analysis output to it silently adds test cases.
OUT_DIR = BACKEND_DIR / "data" / "analysis"
CITIES_JSON = SCRIPT_DIR / "india_cities.json"

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

DEFAULT_START = "2025-01-01"
DEFAULT_END = "2026-08-20"

HTTP_RETRIES = 5
HTTP_BACKOFF = 3.0
POLITE_GAP_S = 1.2  # both are free public services; stay well under their limits

# canonical name -> (Open-Meteo daily field, NASA POWER parameter, unit, threshold key)
# The threshold key names the entry in /api/model/status thresholds.bust_threshold that
# this variable's disagreement should be judged against.
VARIABLES = {
    "temperature_c": ("temperature_2m_mean", "T2M", "degC", "temperature_c"),
    "humidity_pct": ("relative_humidity_2m_mean", "RH2M", "%", "humidity_pct"),
    "rainfall_mm": ("precipitation_sum", "PRECTOTCORR", "mm/day", "rainfall_mm"),
    "wind_speed_ms": ("wind_speed_10m_mean", "WS10M", "m/s", "wind_speed_ms"),
}

# Measured from the live model run; override with --thresholds if they drift.
DEFAULT_THRESHOLDS = {
    "temperature_c": 4.4802,
    "humidity_pct": 23.5358,
    "rainfall_mm": 15.1820,
    "wind_speed_ms": 2.3647,
}

_session = requests.Session()
_session.headers["User-Agent"] = "Sanket/verification-check (SIH 2026; NCMRWF PS 26079)"


def _get_json(url: str, params: dict) -> dict:
    last = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = _session.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} for {r.url}"
        except requests.RequestException as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(HTTP_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"GET failed after {HTTP_RETRIES} tries: {url} ({last})")


def fetch_era5(city: dict, start: str, end: str) -> pd.DataFrame:
    """Daily ERA5 at one point, the same product and API the pipeline verifies against."""
    js = _get_json(ERA5_URL, {
        "latitude": city["lat"], "longitude": city["lon"],
        "start_date": start, "end_date": end,
        "daily": ",".join(v[0] for v in VARIABLES.values()),
        "timezone": "UTC",
        # Open-Meteo defaults wind to km/h; NASA POWER's WS10M is m/s. Without this the
        # comparison silently comes out as a constant negative bias equal to the MAE -
        # which is the fingerprint of a unit mismatch, not a real disagreement.
        "wind_speed_unit": "ms",
    })
    d = js.get("daily") or {}
    out = pd.DataFrame({"date": pd.to_datetime(d.get("time", []))})
    for canon, (om_field, _, _, _) in VARIABLES.items():
        out[canon] = pd.to_numeric(pd.Series(d.get(om_field, [])), errors="coerce")
    return out


def fetch_power(city: dict, start: str, end: str) -> pd.DataFrame:
    """Daily MERRA-2 at one point via NASA POWER. -999 is POWER's missing-value flag."""
    js = _get_json(POWER_URL, {
        "parameters": ",".join(v[1] for v in VARIABLES.values()),
        "community": "AG",
        "latitude": city["lat"], "longitude": city["lon"],
        "start": start.replace("-", ""), "end": end.replace("-", ""),
        "format": "JSON",
    })
    params = js["properties"]["parameter"]
    dates = sorted(params[VARIABLES["temperature_c"][1]].keys())
    out = pd.DataFrame({"date": pd.to_datetime(dates, format="%Y%m%d")})
    for canon, (_, power_key, _, _) in VARIABLES.items():
        series = params.get(power_key, {})
        vals = pd.Series([series.get(d) for d in dates], dtype="float64")
        out[canon] = vals.where(vals > -998)   # -999 = not computable / out of range
    return out


def compare(paired: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    rows = []
    for canon, (_, _, unit, tkey) in VARIABLES.items():
        a, b = paired[f"{canon}_era5"], paired[f"{canon}_merra2"]
        m = a.notna() & b.notna()
        if not m.any():
            continue
        a, b = a[m], b[m]
        diff = b - a
        thr = thresholds.get(tkey)
        mae = diff.abs().mean()
        rows.append({
            "variable": canon,
            "unit": unit,
            "n": int(m.sum()),
            "bias": diff.mean(),
            "mae": mae,
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "corr": float(a.corr(b)),
            "bust_threshold": thr,
            # The number that actually matters: disagreement between two reanalyses,
            # expressed as a fraction of the error size Sanket calls a bust.
            "mae_pct_of_threshold": (100.0 * mae / thr) if thr else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--max-cities", type=int, default=None)
    ap.add_argument("--thresholds", help="JSON file of {variable: bust_threshold}")
    ap.add_argument("--out", default=str(OUT_DIR / "verification_product_agreement.csv"))
    args = ap.parse_args()

    if not CITIES_JSON.exists():
        sys.exit(f"missing {CITIES_JSON}")
    cities = json.loads(CITIES_JSON.read_text())
    if args.max_cities:
        cities = cities[: args.max_cities]

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        thresholds.update(json.loads(Path(args.thresholds).read_text()))

    print(f"{len(cities)} cities, {args.start} -> {args.end}")
    print("ERA5 (Open-Meteo archive) vs MERRA-2 (NASA POWER). Neither needs a key.\n")

    frames = []
    for i, c in enumerate(cities, 1):
        try:
            era5 = fetch_era5(c, args.start, args.end)
            merra = fetch_power(c, args.start, args.end)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:2d}/{len(cities)}] {c['city']:<18s} FAILED: {exc}")
            continue
        j = era5.merge(merra, on="date", suffixes=("_era5", "_merra2"))
        j.insert(0, "city", c["city"])
        frames.append(j)
        print(f"  [{i:2d}/{len(cities)}] {c['city']:<18s} {len(j):5d} paired days")
        time.sleep(POLITE_GAP_S)

    if not frames:
        sys.exit("no city returned data from both products")

    paired = pd.concat(frames, ignore_index=True)
    table = compare(paired, thresholds)

    print(f"\npaired city-days: {len(paired):,} across {paired.city.nunique()} cities\n")
    print(f"{'variable':16s} {'n':>7s} {'bias':>8s} {'MAE':>8s} {'RMSE':>8s} {'corr':>6s} "
          f"{'bust thr':>9s} {'MAE % of thr':>13s}")
    for _, r in table.iterrows():
        # Bracket access, not attribute access: `corr` (and friends) are Series *methods*,
        # so r.corr silently yields a bound method instead of the column.
        print(f"{r['variable']:16s} {r['n']:7,d} {r['bias']:8.2f} {r['mae']:8.2f} "
              f"{r['rmse']:8.2f} {r['corr']:6.3f} {r['bust_threshold']:9.2f} "
              f"{r['mae_pct_of_threshold']:12.0f}%")

    print("\nHow to read this. Near-zero bias means the two products agree on average, so "
          "\nthe ERA5 ingestion is not systematically off. The last column is the point: it "
          "\nis how far two independent reanalyses sit apart, as a fraction of the error "
          "\nSanket calls a bust. It is the irreducible uncertainty in the label itself.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    paired.to_csv(str(args.out).replace(".csv", "_paired.csv"), index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
