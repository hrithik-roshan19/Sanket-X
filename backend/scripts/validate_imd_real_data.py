from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]

GEO_PATH = (
    BACKEND_DIR
    / "data"
    / "geo"
    / "imd_meteorological_subdivisions.geojson"
)

EXPECTED_SUBDIVISIONS = 36

EXPECTED_LAT_MIN = 6.5
EXPECTED_LAT_MAX = 38.5

EXPECTED_LON_MIN = 66.5
EXPECTED_LON_MAX = 100.0

GRID_STEP = 0.25

EXPECTED_CRS = "EPSG:4326"


def validate_geometry(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing IMD geometry: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("type") != "FeatureCollection":
        raise ValueError(
            "Geometry must be a FeatureCollection."
        )

    features = payload.get(
        "features",
        [],
    )

    if len(features) != EXPECTED_SUBDIVISIONS:
        raise ValueError(
            f"Expected {EXPECTED_SUBDIVISIONS} "
            f"subdivisions, got {len(features)}."
        )

    # Validate the CRS metadata written by the
    # Sanket-X geometry preparation step.
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "geopandas is required to validate "
            "IMD subdivision geometry."
        ) from exc

    gdf = gpd.read_file(path)

    if gdf.crs is None:
        raise ValueError(
            "IMD subdivision geometry has no CRS."
        )

    if str(gdf.crs).upper() != EXPECTED_CRS:
        raise ValueError(
            "Expected IMD subdivision geometry CRS "
            f"{EXPECTED_CRS}, got {gdf.crs}."
        )

    if len(gdf) != EXPECTED_SUBDIVISIONS:
        raise ValueError(
            "GeoPandas feature count does not match "
            f"{EXPECTED_SUBDIVISIONS}."
        )

    if gdf.geometry.isna().any():
        raise ValueError(
            "One or more subdivision geometries are null."
        )

    invalid_count = int(
        (~gdf.geometry.is_valid).sum()
    )

    if invalid_count:
        raise ValueError(
            f"{invalid_count} subdivision geometries "
            "are invalid."
        )

    empty_count = int(
        gdf.geometry.is_empty.sum()
    )

    if empty_count:
        raise ValueError(
            f"{empty_count} subdivision geometries "
            "are empty."
        )

    # Geographic bounds must be realistic for India
    # and the supplied IMD rainfall grid.
    min_lon, min_lat, max_lon, max_lat = (
        gdf.total_bounds
    )

    if not (
        EXPECTED_LON_MIN <= min_lon <= EXPECTED_LON_MAX
        and EXPECTED_LON_MIN <= max_lon <= EXPECTED_LON_MAX
        and EXPECTED_LAT_MIN <= min_lat <= EXPECTED_LAT_MAX
        and EXPECTED_LAT_MIN <= max_lat <= EXPECTED_LAT_MAX
    ):
        raise ValueError(
            "IMD subdivision geographic bounds are "
            "outside the expected rainfall-grid extent. "
            f"Got [{min_lon}, {min_lat}, "
            f"{max_lon}, {max_lat}]."
        )

    ids = []
    source_ids = []
    names = []

    for index, feature in enumerate(
        features,
        start=1,
    ):
        geometry = feature.get(
            "geometry"
        )

        if not geometry:
            raise ValueError(
                f"Feature {index} has no geometry."
            )

        if geometry.get("type") not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError(
                f"Feature {index} has unsupported "
                "subdivision geometry."
            )

        properties = feature.get(
            "properties",
            {},
        )

        # The downloader creates a stable internal
        # Sanket-X identifier.
        sanket_x_id = properties.get(
            "sanket_x_id"
        )

        # The original IMD source identifier is preserved.
        imd_source_id = properties.get(
            "imd_source_id"
        )

        # Current IMD source subdivision name is
        # properties["subdivisio"]; the downloader
        # preserves it as sanket_x_subdivision.
        name = properties.get(
            "sanket_x_subdivision"
        )

        if sanket_x_id is None:
            raise ValueError(
                f"Feature {index} is missing "
                "sanket_x_id."
            )

        if imd_source_id is None:
            raise ValueError(
                f"Feature {index} is missing "
                "imd_source_id."
            )

        if not name:
            raise ValueError(
                f"Feature {index} is missing "
                "sanket_x_subdivision."
            )

        sanket_x_id = str(
            sanket_x_id
        ).strip()

        imd_source_id = str(
            imd_source_id
        ).strip()

        name = str(
            name
        ).strip()

        if not sanket_x_id:
            raise ValueError(
                f"Feature {index} has an empty "
                "sanket_x_id."
            )

        if not imd_source_id:
            raise ValueError(
                f"Feature {index} has an empty "
                "imd_source_id."
            )

        if not name:
            raise ValueError(
                f"Feature {index} has an empty "
                "subdivision name."
            )

        ids.append(
            sanket_x_id
        )

        source_ids.append(
            imd_source_id
        )

        names.append(
            name
        )

    if len(set(ids)) != EXPECTED_SUBDIVISIONS:
        raise ValueError(
            "Sanket-X subdivision IDs are not unique."
        )

    if len(set(source_ids)) != EXPECTED_SUBDIVISIONS:
        raise ValueError(
            "IMD source subdivision IDs are not unique."
        )

    if len(set(names)) != EXPECTED_SUBDIVISIONS:
        raise ValueError(
            "IMD subdivision names are not unique."
        )

    # Confirm the expected grid spacing used by the
    # rainfall dataset.
    if GRID_STEP != 0.25:
        raise ValueError(
            "Unexpected IMD rainfall grid step."
        )

    return {
        "count": EXPECTED_SUBDIVISIONS,
        "unique_ids": len(set(ids)),
        "unique_source_ids": len(set(source_ids)),
        "unique_names": len(set(names)),
        "crs": str(gdf.crs),
        "bounds": [
            float(min_lon),
            float(min_lat),
            float(max_lon),
            float(max_lat),
        ],
        "geometry_valid": True,
        "geometry_empty": False,
        "valid": True,
    }


def validate_rainfall_file(
    path: Path,
) -> dict:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "xarray is required to validate "
            "IMD NetCDF rainfall."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(
            f"Rainfall file does not exist: {path}"
        )

    with xr.open_dataset(
        path
    ) as ds:
        required_coords = {
            "TIME",
            "LATITUDE",
            "LONGITUDE",
        }

        missing_coords = (
            required_coords
            - set(ds.coords)
        )

        if missing_coords:
            raise ValueError(
                "Rainfall NetCDF is missing required "
                f"coordinates: {sorted(missing_coords)}"
            )

        if "RAINFALL" not in ds.data_vars:
            raise ValueError(
                "Rainfall NetCDF does not contain "
                "the expected RAINFALL variable."
            )

        rainfall = ds["RAINFALL"]

        expected_dims = {
            "TIME",
            "LATITUDE",
            "LONGITUDE",
        }

        actual_dims = set(
            rainfall.dims
        )

        if actual_dims != expected_dims:
            raise ValueError(
                "Unexpected RAINFALL dimensions. "
                f"Expected {sorted(expected_dims)}, "
                f"got {list(rainfall.dims)}."
            )

        time_values = ds["TIME"].values
        lat_values = np.asarray(
            ds["LATITUDE"].values,
            dtype=float,
        )
        lon_values = np.asarray(
            ds["LONGITUDE"].values,
            dtype=float,
        )

        if len(time_values) == 0:
            raise ValueError(
                "Rainfall NetCDF has no TIME records."
            )

        if len(lat_values) == 0:
            raise ValueError(
                "Rainfall NetCDF has no latitude values."
            )

        if len(lon_values) == 0:
            raise ValueError(
                "Rainfall NetCDF has no longitude values."
            )

        if not np.all(
            np.isfinite(lat_values)
        ):
            raise ValueError(
                "Latitude coordinate contains "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(lon_values)
        ):
            raise ValueError(
                "Longitude coordinate contains "
                "non-finite values."
            )

        if not np.all(
            np.diff(lat_values) > 0
        ):
            raise ValueError(
                "Latitude coordinate is not strictly "
                "increasing."
            )

        if not np.all(
            np.diff(lon_values) > 0
        ):
            raise ValueError(
                "Longitude coordinate is not strictly "
                "increasing."
            )

        if not np.isclose(
            np.diff(lat_values),
            GRID_STEP,
            atol=1e-6,
        ).all():
            raise ValueError(
                "Latitude grid is not consistently "
                "0.25 degrees."
            )

        if not np.isclose(
            np.diff(lon_values),
            GRID_STEP,
            atol=1e-6,
        ).all():
            raise ValueError(
                "Longitude grid is not consistently "
                "0.25 degrees."
            )

        lat_min = float(
            lat_values.min()
        )
        lat_max = float(
            lat_values.max()
        )

        lon_min = float(
            lon_values.min()
        )
        lon_max = float(
            lon_values.max()
        )

        if not np.isclose(
            lat_min,
            EXPECTED_LAT_MIN,
            atol=1e-6,
        ):
            raise ValueError(
                f"Unexpected latitude minimum: "
                f"{lat_min}."
            )

        if not np.isclose(
            lat_max,
            EXPECTED_LAT_MAX,
            atol=1e-6,
        ):
            raise ValueError(
                f"Unexpected latitude maximum: "
                f"{lat_max}."
            )

        if not np.isclose(
            lon_min,
            EXPECTED_LON_MIN,
            atol=1e-6,
        ):
            raise ValueError(
                f"Unexpected longitude minimum: "
                f"{lon_min}."
            )

        if not np.isclose(
            lon_max,
            EXPECTED_LON_MAX,
            atol=1e-6,
        ):
            raise ValueError(
                f"Unexpected longitude maximum: "
                f"{lon_max}."
            )

        values = np.asarray(
            rainfall.values,
            dtype=float,
        )

        expected_shape = (
            len(time_values),
            len(lat_values),
            len(lon_values),
        )

        if values.shape != expected_shape:
            raise ValueError(
                "RAIN​​FALL array shape does not match "
                "TIME/LATITUDE/LONGITUDE dimensions."
            )

        finite = values[
            np.isfinite(values)
        ]

        if finite.size == 0:
            raise ValueError(
                "Rainfall dataset contains "
                "no finite values."
            )

        negative_fraction = float(
            np.mean(finite < 0)
        )

        if negative_fraction > 0.01:
            raise ValueError(
                "More than 1% of finite rainfall "
                "values are negative."
            )

        return {
            "variable": "RAINFALL",
            "dimensions": list(
                rainfall.dims
            ),
            "shape": list(
                rainfall.shape
            ),
            "time_records": int(
                len(time_values)
            ),
            "latitude_points": int(
                len(lat_values)
            ),
            "longitude_points": int(
                len(lon_values)
            ),
            "latitude_range": [
                lat_min,
                lat_max,
            ],
            "longitude_range": [
                lon_min,
                lon_max,
            ],
            "grid_step_degrees": GRID_STEP,
            "finite_values": int(
                finite.size
            ),
            "min_mm": float(
                finite.min()
            ),
            "max_mm": float(
                finite.max()
            ),
            "negative_fraction": (
                negative_fraction
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--geometry",
        type=Path,
        default=GEO_PATH,
    )

    parser.add_argument(
        "--rainfall",
        type=Path,
        default=None,
        help=(
            "Optional IMD yearly NetCDF file."
        ),
    )

    args = parser.parse_args()

    report = {
        "geometry": None,
        "rainfall": None,
        "valid": False,
    }

    try:
        report["geometry"] = (
            validate_geometry(
                args.geometry
            )
        )

        if args.rainfall:
            report["rainfall"] = (
                validate_rainfall_file(
                    args.rainfall
                )
            )

        report["valid"] = True

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        return 0

    except Exception as exc:
        print(
            f"VALIDATION FAILED: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )