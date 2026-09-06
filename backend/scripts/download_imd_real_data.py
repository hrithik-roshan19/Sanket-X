from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen


BACKEND_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BACKEND_DIR / "data"

GEO_DIR = DATA_DIR / "geo"

IMD_GEO_PATH = (
    GEO_DIR / "imd_meteorological_subdivisions.geojson"
)

IMD_RAINFALL_DIR = (
    DATA_DIR / "imd_rainfall"
)

IMD_SUBDIVISION_URL = (
    "https://mausam.imd.gov.in/"
    "imd_latest/contents/district_shapefiles/"
    "sd_boundary.json"
)

IMD_RAINFALL_PAGE = (
    "https://www.imdpune.gov.in/"
    "cmpg/Griddata/Rainfall_25_NetCDF.html"
)

EXPECTED_COUNT = 36

# The IMD source geometry uses projected coordinates.
# These coordinates correspond to UTM Zone 44N.
SOURCE_CRS = "EPSG:32644"

# Sanket-X uses latitude/longitude for the rainfall grid.
OUTPUT_CRS = "EPSG:4326"


def download_bytes(
    url: str,
    timeout: int = 60,
) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Sanket-X/1.0 "
                "(scientific-weather-validation)"
            )
        },
    )

    with urlopen(
        request,
        timeout=timeout,
    ) as response:
        return response.read()


def validate_geojson(
    payload: dict,
) -> dict:
    if payload.get("type") != "FeatureCollection":
        raise ValueError(
            "IMD subdivision file is not a FeatureCollection."
        )

    features = payload.get(
        "features",
        [],
    )

    if len(features) != EXPECTED_COUNT:
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_COUNT} IMD meteorological "
            f"subdivisions, found {len(features)}."
        )

    names = []
    ids = []

    for index, feature in enumerate(
        features,
        start=1,
    ):
        if feature.get("type") != "Feature":
            raise ValueError(
                f"Invalid feature at position {index}."
            )

        geometry = feature.get("geometry")

        if not geometry:
            raise ValueError(
                f"Subdivision feature {index} "
                "has no geometry."
            )

        if geometry.get("type") not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError(
                f"Subdivision feature {index} geometry "
                "must be Polygon or MultiPolygon."
            )

        properties = feature.get(
            "properties",
            {},
        )

        # The current official IMD JSON exposes the
        # subdivision identifier as a top-level GeoJSON
        # feature "id". OBJECTID_1 is preserved as the
        # original source database identifier.
        source_feature_id = feature.get("id")

        if source_feature_id is None:
            source_feature_id = properties.get(
                "OBJECTID_1"
            )

        if source_feature_id is None:
            raise ValueError(
                f"Subdivision feature {index} "
                "is missing an ID."
            )

        # Current IMD source uses "subdivisio".
        subdivision_name = (
            properties.get("subdivisio")
            or properties.get("subdivision")
            or properties.get("name")
            or properties.get("subdivision_name")
        )

        if not subdivision_name:
            raise ValueError(
                f"Subdivision feature {index} "
                "is missing a name."
            )

        subdivision_name = str(
            subdivision_name
        ).strip()

        if not subdivision_name:
            raise ValueError(
                f"Subdivision feature {index} "
                "has an empty name."
            )

        # Preserve all original IMD properties and add
        # stable Sanket-X metadata.
        feature["properties"] = {
            **properties,
            "sanket_x_id": f"IMD-{index:02d}",
            "imd_source_id": str(
                source_feature_id
            ),
            "sanket_x_subdivision": (
                subdivision_name
            ),
        }

        ids.append(
            str(source_feature_id)
        )

        names.append(
            subdivision_name
        )

    if len(set(ids)) != EXPECTED_COUNT:
        raise ValueError(
            "IMD subdivision source IDs "
            "are not unique."
        )

    if len(set(names)) != EXPECTED_COUNT:
        raise ValueError(
            "IMD subdivision names "
            "are not unique."
        )

    payload["_sanket_x_validation"] = {
        "source": IMD_SUBDIVISION_URL,
        "expected_count": EXPECTED_COUNT,
        "actual_count": len(features),
        "validated": True,
        "id_source": (
            "feature.id or properties.OBJECTID_1"
        ),
        "name_source": (
            "properties.subdivisio"
        ),
        "source_crs": SOURCE_CRS,
        "output_crs": OUTPUT_CRS,
    }

    return payload


def transform_to_wgs84(
    payload: dict,
) -> dict:
    """
    Convert the IMD source geometry from the
    source projected CRS to WGS84 latitude/longitude.

    Source:
        EPSG:32644

    Output:
        EPSG:4326
    """

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "This step requires geopandas. "
            "Install it with: "
            "python -m pip install geopandas"
        ) from exc

    source_gdf = gpd.GeoDataFrame.from_features(
        payload["features"],
        crs=SOURCE_CRS,
    )

    if source_gdf.empty:
        raise ValueError(
            "IMD subdivision GeoDataFrame is empty."
        )

    if len(source_gdf) != EXPECTED_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_COUNT} subdivision geometries, "
            f"found {len(source_gdf)}."
        )

    # Geometry validity check before transformation.
    invalid_count = int(
        (~source_gdf.geometry.is_valid).sum()
    )

    if invalid_count:
        raise ValueError(
            f"{invalid_count} IMD subdivision geometries "
            "are invalid."
        )

    # Every polygon must have positive area.
    non_positive_area = int(
        (source_gdf.geometry.area <= 0).sum()
    )

    if non_positive_area:
        raise ValueError(
            f"{non_positive_area} IMD subdivision "
            "geometries have non-positive area."
        )

    output_gdf = source_gdf.to_crs(
        OUTPUT_CRS
    )

    if output_gdf.empty:
        raise ValueError(
            "Transformed IMD subdivision "
            "GeoDataFrame is empty."
        )

    # Validate WGS84 geographic bounds.
    min_lon, min_lat, max_lon, max_lat = (
        output_gdf.total_bounds
    )

    if not (
        -180 <= min_lon <= 180
        and -180 <= max_lon <= 180
        and -90 <= min_lat <= 90
        and -90 <= max_lat <= 90
    ):
        raise ValueError(
            "Transformed IMD subdivision bounds are "
            "outside valid WGS84 longitude/latitude "
            "ranges."
        )

    if min_lon >= max_lon:
        raise ValueError(
            "Invalid WGS84 longitude bounds."
        )

    if min_lat >= max_lat:
        raise ValueError(
            "Invalid WGS84 latitude bounds."
        )

    # Make sure the transformed geometries still contain
    # all 36 subdivisions.
    if len(output_gdf) != EXPECTED_COUNT:
        raise ValueError(
            "Subdivision count changed during "
            "CRS transformation."
        )

    output_payload = json.loads(
        output_gdf.to_json()
    )

    output_payload["_sanket_x_validation"] = {
        "source": IMD_SUBDIVISION_URL,
        "expected_count": EXPECTED_COUNT,
        "actual_count": len(
            output_payload["features"]
        ),
        "validated": True,
        "source_crs": SOURCE_CRS,
        "output_crs": OUTPUT_CRS,
        "id_source": (
            "feature.id or properties.OBJECTID_1"
        ),
        "name_source": (
            "properties.subdivisio"
        ),
        "wgs84_bounds": [
            float(min_lon),
            float(min_lat),
            float(max_lon),
            float(max_lat),
        ],
        "crs_note": (
            "The official IMD subdivision JSON does "
            "not provide explicit CRS metadata. "
            "The source coordinates are interpreted "
            "as UTM Zone 44N (EPSG:32644) and transformed "
            "to WGS84 (EPSG:4326) so they are compatible "
            "with the IMD 0.25-degree rainfall grid."
        ),
    }

    return output_payload


def download_subdivisions(
    output: Path,
) -> None:
    print(
        "Downloading official IMD subdivision geometry..."
    )
    print(IMD_SUBDIVISION_URL)

    raw = download_bytes(
        IMD_SUBDIVISION_URL
    )

    try:
        payload = json.loads(
            raw.decode("utf-8")
        )
    except UnicodeDecodeError:
        payload = json.loads(
            raw.decode("utf-8-sig")
        )

    # Validate the raw IMD structure first.
    payload = validate_geojson(
        payload
    )

    # Transform projected source geometry to WGS84.
    output_payload = transform_to_wgs84(
        payload
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            output_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved: {output}"
    )

    print(
        "Validated 36/36 IMD subdivisions."
    )

    print(
        f"CRS: {SOURCE_CRS} -> {OUTPUT_CRS}"
    )

    print(
        "WGS84 bounds:",
        output_payload[
            "_sanket_x_validation"
        ]["wgs84_bounds"],
    )


def create_rainfall_readme() -> None:
    IMD_RAINFALL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    readme = f"""# IMD 0.25° Gridded Rainfall

Source:
{IMD_RAINFALL_PAGE}

Dataset:
IMD New High Spatial Resolution
0.25° x 0.25° Daily Gridded Rainfall
1901-2024

Unit:
millimetres (mm)

Grid:
135 x 129

First grid point:
6.5N, 66.5E

Last grid point:
38.5N, 100.0E

Coordinate system:
Latitude / Longitude

IMPORTANT:
The yearly NetCDF files are distributed through the
official IMD Climate Research & Services rainfall page.

Do not replace these files with synthetic or third-party
rainfall data for the IMD verification gate.

Downloaded files must be validated before ingestion.

Sanket-X subdivision geometry:
Source CRS: {SOURCE_CRS}
Output CRS: {OUTPUT_CRS}
"""

    (
        IMD_RAINFALL_DIR
        / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    print(
        "Created rainfall data directory:"
    )

    print(IMD_RAINFALL_DIR)

    print(
        "Official IMD rainfall page:"
    )

    print(IMD_RAINFALL_PAGE)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subdivisions",
        action="store_true",
        help=(
            "Download and validate the official "
            "IMD 36-subdivision GeoJSON."
        ),
    )

    parser.add_argument(
        "--prepare-rainfall-dir",
        action="store_true",
        help=(
            "Create the IMD rainfall storage directory "
            "and provenance README."
        ),
    )

    args = parser.parse_args()

    if not (
        args.subdivisions
        or args.prepare_rainfall_dir
    ):
        parser.error(
            "Select --subdivisions and/or "
            "--prepare-rainfall-dir."
        )

    try:
        if args.subdivisions:
            download_subdivisions(
                IMD_GEO_PATH
            )

        if args.prepare_rainfall_dir:
            create_rainfall_readme()

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )