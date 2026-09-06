"""IMD gridded rainfall ingestion primitives.

This module deliberately does not fabricate data. It reads a downloaded IMD NetCDF
file and normalizes it into a tidy daily grid. The exact variable/coordinate names are
validated rather than guessed silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = {
    "time": ("time", "date", "valid_time"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
    "rainfall": ("rain", "rainfall", "rf", "precip", "precipitation"),
}


def _pick(names, candidates):
    lower = {str(n).lower(): n for n in names}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def read_imd_rainfall_netcdf(path: str | Path) -> pd.DataFrame:
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("xarray is required for IMD NetCDF ingestion") from exc

    ds = xr.open_dataset(path)
    try:
        time_name = _pick(ds.coords, CANDIDATES["time"]) or _pick(ds.variables, CANDIDATES["time"])
        lat_name = _pick(ds.coords, CANDIDATES["lat"])
        lon_name = _pick(ds.coords, CANDIDATES["lon"])
        rain_name = _pick(ds.data_vars, CANDIDATES["rainfall"])
        missing = [k for k, v in {"time": time_name, "lat": lat_name, "lon": lon_name, "rainfall": rain_name}.items() if v is None]
        if missing:
            raise ValueError(f"IMD NetCDF missing recognizable fields: {missing}")

        da = ds[rain_name]
        # Daily IMD product must not be silently resampled from an unknown frequency.
        time_values = pd.to_datetime(ds[time_name].values)
        if len(time_values) == 0:
            raise ValueError("IMD rainfall file has no timestamps")
        values = np.asarray(da.values, dtype="float64")
        if np.isinf(values).any():
            raise ValueError("IMD rainfall contains infinite values")
        if np.nanmin(values) < 0:
            raise ValueError("IMD rainfall contains negative values")

        frame = da.to_dataframe(name="rainfall_mm").reset_index()
        frame = frame.rename(columns={time_name: "valid_date", lat_name: "lat", lon_name: "lon"})
        frame["valid_date"] = pd.to_datetime(frame["valid_date"], errors="coerce").dt.date
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        frame["rainfall_mm"] = pd.to_numeric(frame["rainfall_mm"], errors="coerce")
        frame = frame.dropna(subset=["valid_date", "lat", "lon"])
        if not frame["lat"].between(-90, 90).all() or not frame["lon"].between(-180, 180).all():
            raise ValueError("IMD rainfall coordinates outside valid ranges")
        return frame[["valid_date", "lat", "lon", "rainfall_mm"]].sort_values(["valid_date", "lat", "lon"]).reset_index(drop=True)
    finally:
        ds.close()
