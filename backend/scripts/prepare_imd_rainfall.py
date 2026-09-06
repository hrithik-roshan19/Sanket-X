from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]

DEFAULT_GEOMETRY = (
    BACKEND_DIR
    / "data"
    / "geo"
    / "imd_meteorological_subdivisions.geojson"
)

DEFAULT_OUTPUT = (
    BACKEND_DIR
    / "data"
    / "samples"
    / "imd_rainfall_subdivisions.parquet"
)

SOURCE_URL = (
    "https://www.imdpune.gov.in/"
    "cmpg/Griddata/"
    "Rainfall_25_NetCDF.html"
)

EXPECTED_SUBDIVISION_COUNT = 36

# The standard IMD 0.25-degree gridded rainfall dataset
# does not provide a usable grid-cell center inside
# Lakshadweep for this spatial aggregation method.
#
# IMPORTANT:
# We do NOT fabricate, interpolate, or copy rainfall
# values for this subdivision.
KNOWN_GRID_COVERAGE_EXCEPTIONS = {
    "IMD-01": "LAKSHADWEEP",
}


def _find_coord(ds, candidates):
    """
    Find a coordinate or variable using case-insensitive matching.
    """

    lookup = {
        str(name).lower(): str(name)
        for name in ds.variables
    }

    for candidate in candidates:

        if candidate in ds.coords:
            return candidate

        if candidate in ds.variables:
            return candidate

        actual = lookup.get(
            candidate.lower()
        )

        if actual is not None:
            return actual

    raise ValueError(
        f"Could not find coordinate among {candidates}. "
        f"Available variables: {list(ds.variables)}"
    )


def _canonical_subdivision_columns(polygons):
    """
    Validate and normalize Sanket-X subdivision identity fields.
    """

    required = [
        "sanket_x_id",
        "sanket_x_subdivision",
        "imd_source_id",
    ]

    missing = [
        column
        for column in required
        if column not in polygons.columns
    ]

    if missing:
        raise ValueError(
            "Geometry is missing required columns: "
            f"{missing}"
        )

    polygons = polygons.copy()

    polygons["sanket_x_id"] = (
        polygons["sanket_x_id"]
        .astype(str)
        .str.strip()
    )

    polygons["sanket_x_subdivision"] = (
        polygons["sanket_x_subdivision"]
        .astype(str)
        .str.strip()
    )

    polygons["imd_source_id"] = (
        pd.to_numeric(
            polygons["imd_source_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    expected_ids = {
        f"IMD-{i:02d}"
        for i in range(
            1,
            EXPECTED_SUBDIVISION_COUNT + 1,
        )
    }

    actual_ids = set(
        polygons["sanket_x_id"]
    )

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    extra_ids = sorted(
        actual_ids - expected_ids
    )

    if missing_ids or extra_ids:
        raise ValueError(
            "Subdivision ID validation failed. "
            f"Missing={missing_ids}, "
            f"Extra={extra_ids}"
        )

    if polygons["sanket_x_id"].duplicated().any():
        raise ValueError(
            "Duplicate Sanket-X subdivision IDs found."
        )

    if polygons["imd_source_id"].isna().any():
        raise ValueError(
            "One or more subdivisions have invalid "
            "IMD source IDs."
        )

    if polygons["imd_source_id"].duplicated().any():
        raise ValueError(
            "Duplicate IMD source IDs found."
        )

    if (
        polygons["sanket_x_subdivision"]
        .eq("")
        .any()
    ):
        raise ValueError(
            "One or more subdivision names are empty."
        )

    return polygons


def _build_grid(gpd, lats, lons):
    """
    Build all rainfall grid-cell centers as one GeoDataFrame.
    """

    lon_grid, lat_grid = np.meshgrid(
        lons,
        lats,
    )

    grid = pd.DataFrame(
        {
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
        }
    )

    geometry = gpd.points_from_xy(
        grid["longitude"],
        grid["latitude"],
    )

    return gpd.GeoDataFrame(
        grid,
        geometry=geometry,
        crs="EPSG:4326",
    )


def _validate_grid_coordinates(
    lats,
    lons,
):
    """
    Validate the expected regular 0.25-degree IMD grid.
    """

    if len(lats) < 2:
        raise ValueError(
            "Rainfall latitude coordinate is too short."
        )

    if len(lons) < 2:
        raise ValueError(
            "Rainfall longitude coordinate is too short."
        )

    lat_diff = np.diff(lats)
    lon_diff = np.diff(lons)

    if not np.all(
        np.isfinite(lats)
    ):
        raise ValueError(
            "Latitude coordinate contains non-finite values."
        )

    if not np.all(
        np.isfinite(lons)
    ):
        raise ValueError(
            "Longitude coordinate contains non-finite values."
        )

    if not np.all(
        lat_diff > 0
    ):
        raise ValueError(
            "Latitude coordinate must be strictly increasing."
        )

    if not np.all(
        lon_diff > 0
    ):
        raise ValueError(
            "Longitude coordinate must be strictly increasing."
        )

    if not np.allclose(
        lat_diff,
        0.25,
        atol=1e-6,
    ):
        raise ValueError(
            "Rainfall latitude grid is not 0.25 degree."
        )

    if not np.allclose(
        lon_diff,
        0.25,
        atol=1e-6,
    ):
        raise ValueError(
            "Rainfall longitude grid is not 0.25 degree."
        )


def _spatially_assign_grid(
    gpd,
    grid,
    polygons,
):
    """
    Assign rainfall grid-cell centers to IMD subdivisions.

    First uses `within`.
    Boundary points are then resolved using `covered_by`.

    Returns one row per assigned grid-cell center.
    """

    polygon_columns = [
        "sanket_x_id",
        "sanket_x_subdivision",
        "imd_source_id",
        "geometry",
    ]

    polygon_view = polygons[
        polygon_columns
    ].copy()

    joined = gpd.sjoin(
        grid,
        polygon_view,
        how="left",
        predicate="within",
    )

    unmatched_mask = (
        joined["sanket_x_id"].isna()
    )

    if unmatched_mask.any():

        unmatched = grid.loc[
            unmatched_mask
        ].copy()

        boundary_matches = gpd.sjoin(
            unmatched,
            polygon_view,
            how="left",
            predicate="covered_by",
        )

        joined.loc[
            unmatched_mask,
            [
                "sanket_x_id",
                "sanket_x_subdivision",
                "imd_source_id",
            ],
        ] = boundary_matches[
            [
                "sanket_x_id",
                "sanket_x_subdivision",
                "imd_source_id",
            ]
        ].to_numpy()

    # Check whether a grid center was assigned
    # to more than one subdivision.
    #
    # This should never happen for valid non-overlapping
    # subdivision polygons.
    assigned_id_counts = (
        joined.reset_index()
        .groupby("index")
        .size()
    )

    ambiguous = assigned_id_counts[
        assigned_id_counts > 1
    ]

    if not ambiguous.empty:
        raise ValueError(
            f"{len(ambiguous)} grid-cell centers "
            "matched multiple subdivisions."
        )

    return joined[
        joined["sanket_x_id"].notna()
    ].copy()


def _add_array_indices(
    assigned,
    lats,
    lons,
):
    """
    Map geographic grid coordinates back to NumPy indices.
    """

    lat_to_index = {
        float(value): index
        for index, value in enumerate(lats)
    }

    lon_to_index = {
        float(value): index
        for index, value in enumerate(lons)
    }

    assigned = assigned.copy()

    assigned["lat_index"] = (
        assigned["latitude"]
        .map(lat_to_index)
    )

    assigned["lon_index"] = (
        assigned["longitude"]
        .map(lon_to_index)
    )

    if (
        assigned["lat_index"].isna().any()
        or assigned["lon_index"].isna().any()
    ):
        raise ValueError(
            "Could not map assigned grid cells "
            "back to rainfall array indices."
        )

    assigned["lat_index"] = (
        assigned["lat_index"]
        .astype(int)
    )

    assigned["lon_index"] = (
        assigned["lon_index"]
        .astype(int)
    )

    return assigned


def _build_coverage_report(
    assigned,
    polygons,
):
    """
    Build subdivision-level spatial coverage information.
    """

    coverage = (
        assigned.groupby(
            "sanket_x_id",
            as_index=False,
        )
        .agg(
            grid_cell_count=(
                "latitude",
                "size",
            ),
            coverage_weight=(
                "area_weight",
                "sum",
            ),
            subdivision_name=(
                "sanket_x_subdivision",
                "first",
            ),
        )
    )

    expected_ids = set(
        polygons["sanket_x_id"]
    )

    covered_ids = set(
        coverage["sanket_x_id"]
    )

    missing_ids = sorted(
        expected_ids - covered_ids
    )

    return coverage, missing_ids


def _generate_daily_subdivision_rainfall(
    assigned,
    rainfall_values,
    times,
):
    """
    Calculate cosine(latitude)-weighted daily rainfall
    for every covered subdivision.
    """

    rows = []

    for region_id, region_grid in assigned.groupby(
        "sanket_x_id",
        sort=True,
    ):

        lat_idx = (
            region_grid["lat_index"]
            .to_numpy()
        )

        lon_idx = (
            region_grid["lon_index"]
            .to_numpy()
        )

        weights = (
            region_grid["area_weight"]
            .to_numpy(
                dtype=float
            )
        )

        rainfall_subset = (
            rainfall_values[
                :,
                lat_idx,
                lon_idx,
            ]
        )

        valid_mask = (
            np.isfinite(
                rainfall_subset
            )
            & (
                rainfall_subset >= 0
            )
        )

        weighted_values = np.where(
            valid_mask,
            rainfall_subset
            * weights[None, :],
            0.0,
        )

        weight_matrix = np.where(
            valid_mask,
            weights[None, :],
            0.0,
        )

        numerator = (
            weighted_values.sum(
                axis=1
            )
        )

        denominator = (
            weight_matrix.sum(
                axis=1
            )
        )

        valid_days = (
            denominator > 0
        )

        daily_rainfall = np.full(
            len(times),
            np.nan,
            dtype=float,
        )

        daily_rainfall[
            valid_days
        ] = (
            numerator[valid_days]
            / denominator[valid_days]
        )

        grid_cell_count = int(
            len(region_grid)
        )

        coverage_weight = float(
            weights.sum()
        )

        subdivision_name = str(
            region_grid[
                "sanket_x_subdivision"
            ].iloc[0]
        )

        for index, timestamp in enumerate(
            times
        ):

            value = daily_rainfall[
                index
            ]

            if not np.isfinite(
                value
            ):
                continue

            rows.append(
                {
                    "region_id": str(
                        region_id
                    ),
                    "subdivision_name": (
                        subdivision_name
                    ),
                    "valid_date": timestamp,
                    "rainfall_mm": float(
                        value
                    ),
                    "grid_cell_count": (
                        grid_cell_count
                    ),
                    "coverage_weight": (
                        coverage_weight
                    ),
                }
            )

    return rows


def _validate_daily_output(
    grouped,
    times,
):
    """
    Validate uniqueness and complete daily coverage.
    """

    duplicate_keys = grouped.duplicated(
        [
            "region_id",
            "valid_date",
        ]
    )

    if duplicate_keys.any():
        duplicates = (
            grouped.loc[
                duplicate_keys,
                [
                    "region_id",
                    "valid_date",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate subdivision/date records "
            f"were generated: {duplicates}"
        )

    expected_date_count = len(
        times
    )

    counts = (
        grouped.groupby(
            "region_id"
        )["valid_date"]
        .nunique()
    )

    incomplete = counts[
        counts != expected_date_count
    ]

    if not incomplete.empty:
        raise ValueError(
            "Incomplete daily coverage detected: "
            f"{incomplete.to_dict()}"
        )

    expected_ids = set(
        grouped["region_id"]
    )

    if not expected_ids:
        raise ValueError(
            "No subdivision IDs found in output."
        )


def prepare(
    rainfall_path: Path,
    output_path: Path,
    geometry_path: Path,
) -> None:

    try:
        import geopandas as gpd
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "This step requires geopandas and xarray."
        ) from exc

    if not rainfall_path.exists():
        raise FileNotFoundError(
            rainfall_path
        )

    if not geometry_path.exists():
        raise FileNotFoundError(
            geometry_path
        )

    print(
        "Loading IMD subdivision geometry..."
    )

    polygons = gpd.read_file(
        geometry_path
    )

    if len(polygons) != (
        EXPECTED_SUBDIVISION_COUNT
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_SUBDIVISION_COUNT} subdivisions, "
            f"found {len(polygons)}."
        )

    polygons = polygons.to_crs(
        "EPSG:4326"
    )

    polygons = (
        _canonical_subdivision_columns(
            polygons
        )
    )

    if polygons.geometry.is_empty.any():
        raise ValueError(
            "One or more subdivision geometries are empty."
        )

    if (
        ~polygons.geometry.is_valid
    ).any():
        raise ValueError(
            "One or more subdivision geometries are invalid."
        )

    print(
        "Subdivision geometry validation: PASS"
    )

    with xr.open_dataset(
        rainfall_path
    ) as ds:

        lat_name = _find_coord(
            ds,
            [
                "lat",
                "latitude",
                "Latitude",
                "LAT",
                "LATITUDE",
            ],
        )

        lon_name = _find_coord(
            ds,
            [
                "lon",
                "longitude",
                "Longitude",
                "LON",
                "LONGITUDE",
            ],
        )

        time_name = _find_coord(
            ds,
            [
                "time",
                "date",
                "Date",
                "TIME",
            ],
        )

        data_vars = list(
            ds.data_vars
        )

        if not data_vars:
            raise ValueError(
                "No rainfall variable found."
            )

        rainfall_candidates = [
            name
            for name in data_vars
            if str(name).upper()
            == "RAINFALL"
        ]

        rainfall_var = (
            rainfall_candidates[0]
            if rainfall_candidates
            else data_vars[0]
        )

        rain = ds[
            rainfall_var
        ]

        lats = np.asarray(
            ds[lat_name].values,
            dtype=float,
        )

        lons = np.asarray(
            ds[lon_name].values,
            dtype=float,
        )

        times = (
            pd.to_datetime(
                ds[time_name].values
            )
            .normalize()
        )

        if rain.ndim != 3:
            raise ValueError(
                "Expected rainfall data to be "
                "three-dimensional: "
                "time x latitude x longitude."
            )

        if len(times) != rain.shape[0]:
            raise ValueError(
                "Rainfall time dimension does not "
                "match time coordinate."
            )

        if len(lats) != rain.shape[1]:
            raise ValueError(
                "Rainfall latitude dimension does not "
                "match latitude coordinate."
            )

        if len(lons) != rain.shape[2]:
            raise ValueError(
                "Rainfall longitude dimension does not "
                "match longitude coordinate."
            )

        _validate_grid_coordinates(
            lats,
            lons,
        )

        print(
            f"Rainfall variable: {rainfall_var}"
        )

        print(
            f"Grid: {len(lats)} x {len(lons)}"
        )

        print(
            f"Dates: {len(times)}"
        )

        grid = _build_grid(
            gpd,
            lats,
            lons,
        )

        assigned = _spatially_assign_grid(
            gpd,
            grid,
            polygons,
        )

        if assigned.empty:
            raise ValueError(
                "No grid cells were assigned "
                "to IMD subdivisions."
            )

        # Cosine(latitude) weighting approximates
        # grid-cell area weighting on a regular
        # latitude/longitude grid.
        assigned["area_weight"] = np.maximum(
            np.cos(
                np.deg2rad(
                    assigned[
                        "latitude"
                    ].to_numpy(
                        dtype=float
                    )
                )
            ),
            0.0,
        )

        coverage, missing_ids = (
            _build_coverage_report(
                assigned,
                polygons,
            )
        )

        print()
        print(
            f"Grid cells assigned: "
            f"{len(assigned):,}"
        )

        print(
            "Subdivisions covered: "
            f"{len(coverage)}/"
            f"{EXPECTED_SUBDIVISION_COUNT}"
        )

        if missing_ids:

            missing_names = (
                polygons[
                    polygons[
                        "sanket_x_id"
                    ].isin(
                        missing_ids
                    )
                ][
                    [
                        "sanket_x_id",
                        "sanket_x_subdivision",
                    ]
                ]
                .to_dict(
                    "records"
                )
            )

            print(
                "Subdivisions with no assigned "
                "0.25-degree grid-cell center:"
            )

            for item in missing_names:
                print(
                    f"  {item['sanket_x_id']}: "
                    f"{item['sanket_x_subdivision']}"
                )

            unexpected_missing = (
                set(missing_ids)
                - set(
                    KNOWN_GRID_COVERAGE_EXCEPTIONS
                )
            )

            if unexpected_missing:

                raise ValueError(
                    "Unexpected subdivision coverage "
                    "failure: "
                    f"{sorted(unexpected_missing)}. "
                    "No rainfall values were fabricated."
                )

            print()
            print(
                "Known IMD grid-coverage exception:"
            )

            for region_id in missing_ids:
                print(
                    f"  {region_id}: "
                    f"{KNOWN_GRID_COVERAGE_EXCEPTIONS.get(region_id, '')}"
                )

            print(
                "No rainfall value will be "
                "fabricated for this subdivision."
            )

        # We deliberately do not create records
        # for subdivisions with no valid grid-cell
        # representation.
        #
        # For the current 2019 dataset:
        # 35 subdivisions x 365 days = 12,775 rows.
        rainfall_values = np.asarray(
            rain.values,
            dtype=float,
        )

        expected_shape = (
            len(times),
            len(lats),
            len(lons),
        )

        if rainfall_values.shape != (
            expected_shape
        ):
            raise ValueError(
                "Unexpected rainfall array shape: "
                f"{rainfall_values.shape}; "
                f"expected {expected_shape}."
            )

        assigned = _add_array_indices(
            assigned,
            lats,
            lons,
        )

        rows = (
            _generate_daily_subdivision_rainfall(
                assigned,
                rainfall_values,
                times,
            )
        )

    if not rows:
        raise ValueError(
            "No valid subdivision rainfall "
            "records were generated."
        )

    grouped = pd.DataFrame(
        rows
    )

    grouped = grouped.sort_values(
        [
            "region_id",
            "valid_date",
        ]
    ).reset_index(
        drop=True
    )

    _validate_daily_output(
        grouped,
        times,
    )

    # Verify subdivision names are actually present.
    blank_names = (
        grouped[
            "subdivision_name"
        ]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_names.any():
        raise ValueError(
            "Generated output contains blank "
            "subdivision names."
        )

    # Add Sanket-X verification schema.
    grouped["variable"] = (
        "rainfall_mm"
    )

    grouped["value_type"] = (
        "observed"
    )

    grouped["value"] = (
        grouped["rainfall_mm"]
    )

    grouped["verification_status"] = (
        "final"
    )

    grouped["source"] = (
        "IMD_0.25deg_daily_gridded"
    )

    grouped["source_url"] = (
        SOURCE_URL
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped.to_parquet(
        output_path,
        index=False,
    )

    region_counts = (
        grouped.groupby(
            "region_id"
        )
        .size()
        .to_dict()
    )

    metadata = {
        "source": (
            "India Meteorological Department"
        ),
        "dataset": (
            "IMD New High Spatial Resolution "
            "(0.25x0.25 degree) Long Period "
            "Daily Gridded Rainfall Data Set "
            "Over India"
        ),
        "unit": "mm",
        "grid_resolution": (
            "0.25 x 0.25 degree"
        ),
        "aggregation": (
            "cosine(latitude) area-weighted "
            "grid-cell-center assignment"
        ),
        "subdivision_count": (
            EXPECTED_SUBDIVISION_COUNT
        ),
        "covered_subdivision_count": int(
            grouped[
                "region_id"
            ].nunique()
        ),
        "rows": int(
            len(grouped)
        ),
        "date_count": int(
            len(times)
        ),
        "rainfall_variable": (
            rainfall_var
        ),
        "latitude_coordinate": (
            lat_name
        ),
        "longitude_coordinate": (
            lon_name
        ),
        "time_coordinate": (
            time_name
        ),
        "region_rows": (
            region_counts
        ),
        "source_url": (
            SOURCE_URL
        ),
        "fabricated_values": False,
        "grid_coverage_exceptions": (
            KNOWN_GRID_COVERAGE_EXCEPTIONS
        ),
        "coverage_note": (
            "Subdivisions without a usable "
            "0.25-degree grid-cell center "
            "are excluded rather than assigned "
            "fabricated or interpolated rainfall."
        ),
    }

    metadata_path = (
        output_path.with_suffix(
            ".metadata.json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 60
    )

    print(
        f"Saved {len(grouped):,} rows"
    )

    print(
        "Subdivisions: "
        f"{grouped['region_id'].nunique()}/"
        f"{EXPECTED_SUBDIVISION_COUNT}"
    )

    print(
        "Dates: "
        f"{grouped['valid_date'].nunique()}"
    )

    print(
        "Date range: "
        f"{grouped['valid_date'].min().date()} "
        "to "
        f"{grouped['valid_date'].max().date()}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )

    print(
        "=" * 60
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare IMD 0.25-degree daily rainfall "
            "aggregated to IMD meteorological subdivisions."
        )
    )

    parser.add_argument(
        "--rainfall",
        required=True,
        type=Path,
        help="Path to IMD rainfall NetCDF file.",
    )

    parser.add_argument(
        "--geometry",
        type=Path,
        default=DEFAULT_GEOMETRY,
        help=(
            "Path to IMD subdivision GeoJSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output subdivision rainfall Parquet."
        ),
    )

    args = parser.parse_args()

    prepare(
        rainfall_path=args.rainfall,
        output_path=args.output,
        geometry_path=args.geometry,
    )


if __name__ == "__main__":
    main()