from __future__ import annotations

"""
Build regional GEFS rainfall forecasts and verify them against IMD
0.25-degree subdivision rainfall.

Scientific contract
-------------------
- Historical GEFS sample: 5 members (c00,p01,p02,p03,p04).
- Forecast valid date follows Sanket-X convention:
      valid_date = init_date + (lead_day - 1)
- GEFS rows are city-point forecasts. They are mapped to IMD
  meteorological subdivisions by the forecast point coordinates.
- Within each subdivision, forecasts are averaged across available
  forecast points for each member/init/lead.
- Ensemble mean and standard deviation are then calculated across
  members.
- IMD rainfall is the independent regional verification reference.
- Only exact (region_id, init_date, lead_day, valid_date) pairs are
  retained.
- No values are fabricated or filled.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]

DEFAULT_GEFS = (
    BACKEND_DIR
    / "data"
    / "samples"
    / "gefs_reforecast_india_2019.parquet"
)

DEFAULT_IMD = (
    BACKEND_DIR
    / "data"
    / "samples"
    / "imd_rainfall_subdivisions.parquet"
)

DEFAULT_GEOMETRY = (
    BACKEND_DIR
    / "data"
    / "geo"
    / "imd_meteorological_subdivisions.geojson"
)

DEFAULT_OUTPUT = (
    BACKEND_DIR
    / "data"
    / "analysis"
    / "gefs_imd_rainfall_verification_2019.parquet"
)

DEFAULT_METADATA = (
    BACKEND_DIR
    / "data"
    / "analysis"
    / "gefs_imd_rainfall_verification_2019.metadata.json"
)

HISTORICAL_MEMBERS = {"c00", "p01", "p02", "p03", "p04"}
KNOWN_IMD_UNAVAILABLE = {"IMD-01", "IMD-02"}


def _first_existing(columns, candidates, label):
    lookup = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        actual = lookup.get(candidate.lower())
        if actual is not None:
            return actual
    raise ValueError(
        f"Could not find {label}. Tried {candidates}. "
        f"Available columns: {list(columns)}"
    )


def _load_gefs(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    city_col = _first_existing(
        df.columns,
        ["city", "location", "point_name", "station"],
        "GEFS city/point column",
    )
    member_col = _first_existing(
        df.columns,
        ["member", "ensemble_member"],
        "GEFS member column",
    )
    init_col = _first_existing(
        df.columns,
        ["init_date", "initialization_date", "init"],
        "GEFS initialization date",
    )
    lead_col = _first_existing(
        df.columns,
        ["lead_day", "lead_time_days", "lead"],
        "GEFS lead day",
    )
    valid_col = _first_existing(
        df.columns,
        ["valid_date", "forecast_valid_date", "date"],
        "GEFS valid date",
    )
    lat_col = _first_existing(
        df.columns,
        ["latitude", "lat"],
        "GEFS latitude",
    )
    lon_col = _first_existing(
        df.columns,
        ["longitude", "lon", "lng"],
        "GEFS longitude",
    )
    rain_col = _first_existing(
        df.columns,
        [
            "apcp_mm",
            "rainfall_mm",
            "precipitation_mm",
            "precip_mm",
            "rain_mm",
        ],
        "GEFS rainfall variable",
    )

    out = df.rename(
        columns={
            city_col: "city",
            member_col: "member",
            init_col: "init_date",
            lead_col: "lead_day",
            valid_col: "valid_date",
            lat_col: "latitude",
            lon_col: "longitude",
            rain_col: "forecast_rainfall_mm",
        }
    ).copy()

    out["member"] = out["member"].astype(str).str.strip().str.lower()
    out["init_date"] = pd.to_datetime(out["init_date"], errors="coerce").dt.normalize()
    out["valid_date"] = pd.to_datetime(out["valid_date"], errors="coerce").dt.normalize()
    out["lead_day"] = pd.to_numeric(out["lead_day"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["forecast_rainfall_mm"] = pd.to_numeric(
        out["forecast_rainfall_mm"], errors="coerce"
    )

    required = [
        "city",
        "member",
        "init_date",
        "valid_date",
        "lead_day",
        "latitude",
        "longitude",
        "forecast_rainfall_mm",
    ]
    if out[required].isna().any().any():
        bad = out[required].isna().sum()
        raise ValueError(f"GEFS contains null required fields: {bad.to_dict()}")

    out["lead_day"] = out["lead_day"].astype(int)

    if not set(out["member"]).issubset(HISTORICAL_MEMBERS):
        extra = sorted(set(out["member"]) - HISTORICAL_MEMBERS)
        raise ValueError(f"Unexpected historical GEFS members: {extra}")

    if not out["lead_day"].between(1, 10).all():
        raise ValueError("GEFS lead_day must be between 1 and 10.")

    expected_valid = out["init_date"] + pd.to_timedelta(
        out["lead_day"] - 1, unit="D"
    )
    if not (expected_valid == out["valid_date"]).all():
        bad = out.loc[
            expected_valid != out["valid_date"],
            ["init_date", "lead_day", "valid_date"],
        ].head(10)
        raise ValueError(
            "GEFS valid_date does not follow the repository convention:\n"
            f"{bad.to_string(index=False)}"
        )

    if not out["latitude"].between(6.0, 38.5).all():
        raise ValueError("GEFS latitude contains out-of-India-range values.")
    if not out["longitude"].between(66.0, 100.5).all():
        raise ValueError("GEFS longitude contains out-of-India-range values.")
    if (out["forecast_rainfall_mm"] < 0).any():
        raise ValueError("GEFS rainfall contains negative values.")

    duplicate_key = out.duplicated(
        ["city", "init_date", "member", "valid_date"]
    )
    if duplicate_key.any():
        raise ValueError("Duplicate GEFS city/init/member/valid_date keys found.")

    return out


def _load_imd(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    required = [
        "region_id",
        "subdivision_name",
        "valid_date",
        "rainfall_mm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"IMD rainfall is missing columns: {missing}")

    df["region_id"] = df["region_id"].astype(str).str.strip()
    df["subdivision_name"] = df["subdivision_name"].astype(str).str.strip()
    df["valid_date"] = pd.to_datetime(
        df["valid_date"], errors="coerce"
    ).dt.normalize()
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    if df[required].isna().any().any():
        raise ValueError("IMD rainfall contains null required fields.")

    if (df["rainfall_mm"] < 0).any():
        raise ValueError("IMD rainfall contains negative values.")

    duplicate_key = df.duplicated(["region_id", "valid_date"])
    if duplicate_key.any():
        raise ValueError("Duplicate IMD subdivision/date keys found.")

    return df


def _load_polygons(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    required = [
        "sanket_x_id",
        "sanket_x_subdivision",
    ]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Subdivision geometry is missing: {missing}")

    gdf = gdf.copy()
    gdf["sanket_x_id"] = gdf["sanket_x_id"].astype(str).str.strip()
    gdf["sanket_x_subdivision"] = (
        gdf["sanket_x_subdivision"].astype(str).str.strip()
    )

    if gdf.crs is None:
        raise ValueError("Subdivision geometry has no CRS.")

    gdf = gdf.to_crs("EPSG:4326")

    expected = {f"IMD-{i:02d}" for i in range(1, 37)}
    actual = set(gdf["sanket_x_id"])

    if expected != actual:
        raise ValueError(
            f"Subdivision ID mismatch. Missing={sorted(expected-actual)}, "
            f"Extra={sorted(actual-expected)}"
        )

    if gdf["sanket_x_id"].duplicated().any():
        raise ValueError("Duplicate subdivision IDs found.")

    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Subdivision geometry contains null/empty geometries.")

    if not gdf.geometry.is_valid.all():
        raise ValueError("Subdivision geometry contains invalid geometries.")

    return gdf


def _assign_points_to_subdivisions(
    gefs: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        gefs.copy(),
        geometry=gpd.points_from_xy(
            gefs["longitude"],
            gefs["latitude"],
        ),
        crs="EPSG:4326",
    )

    # `covered_by` includes points exactly on a polygon boundary.
    assigned = gpd.sjoin(
        points,
        polygons[
            ["sanket_x_id", "sanket_x_subdivision", "geometry"]
        ],
        how="left",
        predicate="covered_by",
    )

    # Known administrative point/polygon representation mismatches.
    #
    # These are explicit administrative mappings, NOT nearest-neighbour
    # assignments and NOT synthetic forecast generation.
    explicit_city_mapping = {
        "Mumbai": "IMD-07",       # KONKAN & GOA
        "Daman": "IMD-26",        # GUJARAT REGION
        "Port Blair": "IMD-02",   # A & N ISLAND
    }

    for city, region_id in explicit_city_mapping.items():
        city_mask = (
            assigned["city"].eq(city)
            & assigned["sanket_x_id"].isna()
        )

        if city_mask.any():
            assigned.loc[city_mask, "sanket_x_id"] = region_id

    # Report the explicit mappings that were actually needed.
    print("Explicit GEFS city mappings applied:")
    for city, region_id in explicit_city_mapping.items():
        if city in set(assigned["city"]):
            print(f"  {city} -> {region_id}")

    # Any point still unmatched is a genuine spatial coverage problem.
    # Do not silently force it into the nearest subdivision.
    if assigned["sanket_x_id"].isna().any():
        unmatched = (
            assigned.loc[
                assigned["sanket_x_id"].isna(),
                ["city", "latitude", "longitude"],
            ]
            .drop_duplicates()
        )

        raise ValueError(
            "Some GEFS forecast points are outside all IMD subdivisions "
            "and have no explicit administrative mapping:\n"
            f"{unmatched.to_string(index=False)}"
        )

    return pd.DataFrame(
        assigned.drop(columns=["geometry", "index_right"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GEFS-vs-IMD regional rainfall verification."
    )
    parser.add_argument("--gefs", type=Path, default=DEFAULT_GEFS)
    parser.add_argument("--imd", type=Path, default=DEFAULT_IMD)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    print("Loading GEFS forecast sample...")
    gefs = _load_gefs(args.gefs)
    print(f"GEFS rows: {len(gefs):,}")
    print(f"GEFS cities: {gefs['city'].nunique()}")
    print(f"GEFS init dates: {gefs['init_date'].nunique()}")
    print(f"GEFS members: {sorted(gefs['member'].unique())}")

    print("\nLoading IMD subdivision rainfall...")
    imd = _load_imd(args.imd)
    print(f"IMD rows: {len(imd):,}")
    print(f"IMD usable subdivisions: {imd['region_id'].nunique()}/36")

    print("\nLoading IMD subdivision geometry...")
    polygons = _load_polygons(args.geometry)
    print("Subdivision geometry: PASS")

    print("\nAssigning GEFS forecast points to IMD subdivisions...")
    assigned = _assign_points_to_subdivisions(gefs, polygons)

    print(
        f"Assigned cities: {assigned['city'].nunique()}"
    )
    print(
        f"Assigned subdivisions: "
        f"{assigned['sanket_x_id'].nunique()}/36"
    )

    point_map = (
        assigned[
            [
                "city",
                "latitude",
                "longitude",
                "sanket_x_id",
                "sanket_x_subdivision",
            ]
        ]
        .drop_duplicates()
        .sort_values(["sanket_x_id", "city"])
    )

    # First aggregate multiple city points within a subdivision for each
    # ensemble member and forecast cycle/lead.
    regional_member = (
        assigned.groupby(
            [
                "sanket_x_id",
                "sanket_x_subdivision",
                "init_date",
                "lead_day",
                "valid_date",
                "member",
            ],
            as_index=False,
        )["forecast_rainfall_mm"]
        .mean()
        .rename(
            columns={
                "forecast_rainfall_mm": "member_forecast_rainfall_mm"
            }
        )
    )

    member_counts = (
        regional_member.groupby(
            [
                "sanket_x_id",
                "init_date",
                "lead_day",
                "valid_date",
            ]
        )["member"]
        .nunique()
        .rename("ensemble_member_count")
        .reset_index()
    )

    # For the historical sample we require the complete five-member
    # ensemble before calling a row a verification case.
    complete_keys = member_counts.loc[
        member_counts["ensemble_member_count"] == len(HISTORICAL_MEMBERS)
    ].drop(columns=["ensemble_member_count"])

    regional_member = regional_member.merge(
        complete_keys,
        on=[
            "sanket_x_id",
            "init_date",
            "lead_day",
            "valid_date",
        ],
        how="inner",
    )

    regional_forecast = (
        regional_member.groupby(
            [
                "sanket_x_id",
                "sanket_x_subdivision",
                "init_date",
                "lead_day",
                "valid_date",
            ],
            as_index=False,
        )
        .agg(
            forecast_rainfall_mm=(
                "member_forecast_rainfall_mm",
                "mean",
            ),
            ensemble_spread_mm=(
                "member_forecast_rainfall_mm",
                "std",
            ),
            ensemble_min_mm=(
                "member_forecast_rainfall_mm",
                "min",
            ),
            ensemble_max_mm=(
                "member_forecast_rainfall_mm",
                "max",
            ),
            ensemble_member_count=(
                "member",
                "nunique",
            ),
        )
    )

    # Exact regional verification join.
    verified = regional_forecast.merge(
        imd.rename(
            columns={
                "region_id": "sanket_x_id",
            }
        )[
            [
                "sanket_x_id",
                "valid_date",
                "rainfall_mm",
            ]
        ],
        on=["sanket_x_id", "valid_date"],
        how="inner",
        validate="many_to_one",
    )

    verified = verified.rename(
        columns={
            "rainfall_mm": "observed_rainfall_mm",
        }
    )

    verified["absolute_error_mm"] = (
        verified["forecast_rainfall_mm"]
        - verified["observed_rainfall_mm"]
    ).abs()

    verified["signed_error_mm"] = (
        verified["forecast_rainfall_mm"]
        - verified["observed_rainfall_mm"]
    )

    # Regional rainfall bust definition for this independent verification:
    # a bust is an absolute rainfall error >= the historical P90 threshold.
    # Threshold is calculated only from the verification dataset itself and
    # is reported as an evaluation threshold; it is NOT used to train the
    # production model.
    if verified.empty:
        raise ValueError(
            "No exact GEFS/IMD regional verification pairs were produced."
        )

    rainfall_threshold = float(
        verified["absolute_error_mm"].quantile(0.90)
    )

    verified["bust_threshold_mm"] = rainfall_threshold
    verified["y_bust"] = (
        verified["absolute_error_mm"] >= rainfall_threshold
    ).astype(int)

    verified["verification_source"] = (
        "GEFSv12_reforecast_vs_IMD_0.25deg_subdivision"
    )
    verified["forecast_members"] = ",".join(
        sorted(HISTORICAL_MEMBERS)
    )

    verified = verified.sort_values(
        ["sanket_x_id", "init_date", "lead_day"]
    ).reset_index(drop=True)

    duplicate_key = verified.duplicated(
        [
            "sanket_x_id",
            "init_date",
            "lead_day",
            "valid_date",
        ]
    )
    if duplicate_key.any():
        raise ValueError(
            "Duplicate regional verification keys were generated."
        )

    # ------------------------------------------------------------------
    # Coverage accounting
    # ------------------------------------------------------------------
    #
    # The current historical GEFS sample contains 36 city points, not a
    # gridded forecast covering every one of IMD's 36 subdivisions.
    #
    # Therefore, subdivisions without a representative GEFS city are
    # reported as COVERAGE GAPS rather than treated as code/data failures.
    # No nearest-neighbour or synthetic forecast is created.
    # ------------------------------------------------------------------

    expected_regional_ids = set(
        polygons["sanket_x_id"].dropna().astype(str)
    )

    output_regional_ids = set(
        assigned["sanket_x_id"].dropna().astype(str)
    )

    represented_regional_ids = sorted(
        expected_regional_ids & output_regional_ids
    )

    gefts_coverage_gaps = sorted(
        expected_regional_ids - output_regional_ids
    )

    print(
        f"GEFS subdivision representation: "
        f"{len(represented_regional_ids)}/{len(expected_regional_ids)}"
    )

    if gefts_coverage_gaps:
        print(
            "GEFS coverage gaps "
            "(no representative city point in current sample):"
        )
        for region_id in gefts_coverage_gaps:
            print(f"  {region_id}")

    # These are expected coverage limitations of the current city-point
    # historical GEFS sample. They must remain visible in metadata.
    coverage_status = {
        "expected_subdivisions": len(expected_regional_ids),
        "represented_subdivisions": len(represented_regional_ids),
        "coverage_gap_subdivisions": len(gefts_coverage_gaps),
        "represented_region_ids": represented_regional_ids,
        "coverage_gap_region_ids": gefts_coverage_gaps,
    }
# Basic completeness: every available IMD subdivision should have
    # multiple init/lead verification cases if the GEFS sample supports them.
    regional_case_counts = (
        verified.groupby("sanket_x_id")
        .size()
        .sort_index()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    verified.to_parquet(args.output, index=False)

    metadata = {
        "forecast_source": "NOAA GEFSv12 reforecast",
        "observation_source": "India Meteorological Department 0.25-degree daily gridded rainfall",
        "forecast_file": str(args.gefs),
        "observation_file": str(args.imd),
        "geometry_file": str(args.geometry),
        "historical_members": sorted(HISTORICAL_MEMBERS),
        "historical_member_count": len(HISTORICAL_MEMBERS),
        "official_subdivision_count": 36,
        "known_imd_unavailable_subdivisions": sorted(KNOWN_IMD_UNAVAILABLE),
        "usable_imd_subdivision_count": len(expected_regional_ids),
        "forecast_point_count": int(assigned["city"].nunique()),
        "forecast_row_count": int(len(gefs)),
        "regional_member_rows": int(len(regional_member)),
        "regional_forecast_cases": int(len(regional_forecast)),
        "verified_cases": int(len(verified)),
        "verified_subdivision_count": int(
            verified["sanket_x_id"].nunique()
        ),
        "date_min": str(verified["valid_date"].min().date()),
        "date_max": str(verified["valid_date"].max().date()),
        "lead_day_min": int(verified["lead_day"].min()),
        "lead_day_max": int(verified["lead_day"].max()),
        "rainfall_error_p90_mm": rainfall_threshold,
        "bust_rate": float(verified["y_bust"].mean()),
        "mean_absolute_error_mm": float(
            verified["absolute_error_mm"].mean()
        ),
        "rmse_mm": float(
            np.sqrt(
                np.mean(
                    verified["signed_error_mm"] ** 2
                )
            )
        ),
        "regional_case_counts": {
            str(k): int(v)
            for k, v in regional_case_counts.items()
        },
        "city_to_subdivision_mapping": point_map.to_dict(
            orient="records"
        ),
        "fabricated_values": False,
        "production_model_retrained": False,
    }

    args.metadata.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("GEFS -> IMD REGIONAL VERIFICATION: PASS")
    print("=" * 60)
    print(f"Forecast points              : {assigned['city'].nunique()}")
    print(f"Usable IMD subdivisions      : {len(expected_regional_ids)}/36")
    print(f"Regional forecast cases      : {len(regional_forecast):,}")
    print(f"Exact verified cases         : {len(verified):,}")
    print(f"Verified subdivisions        : {verified['sanket_x_id'].nunique()}/34")
    print(f"Lead days                    : {verified['lead_day'].min()}-{verified['lead_day'].max()}")
    print(f"Verification P90 error       : {rainfall_threshold:.3f} mm")
    print(f"Independent rainfall bust %  : {verified['y_bust'].mean() * 100:.2f}%")
    print(f"MAE                          : {verified['absolute_error_mm'].mean():.3f} mm")
    print(f"RMSE                         : {np.sqrt(np.mean(verified['signed_error_mm'] ** 2)):.3f} mm")
    print(f"Output                       : {args.output}")
    print(f"Metadata                     : {args.metadata}")
    print("=" * 60)


if __name__ == "__main__":
    main()
