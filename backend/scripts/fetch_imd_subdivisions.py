"""Fetch and validate the IMD meteorological subdivision geometry.

The official IMD page currently exposes the subdivision boundary JSON used by its
rainfall products. We keep this as a downloaded, versioned input rather than inventing
or approximating subdivision polygons.

Usage:
    python scripts/fetch_imd_subdivisions.py
    python scripts/fetch_imd_subdivisions.py --url <mirror-or-official-url>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from app.data_sources.imd_subdivisions import OFFICIAL_GEOMETRY_URL, IMD_SUBDIVISION_NAMES

OUT = Path(__file__).resolve().parents[1] / "data" / "geo" / "imd_subdivisions.geojson"


def validate_geojson(payload: dict) -> tuple[bool, str]:
    if payload.get("type") != "FeatureCollection":
        return False, "not a GeoJSON FeatureCollection"
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return False, "GeoJSON contains no features"
    names = []
    for feature in features:
        if feature.get("geometry") is None:
            return False, "feature without geometry"
        props = feature.get("properties") or {}
        # IMD has used slightly different property spellings across versions.
        name = props.get("subdivision") or props.get("Subdivision") or props.get("name")
        if name:
            names.append(str(name).strip())
    if names:
        normalized = {x.casefold() for x in names}
        missing = [x for x in IMD_SUBDIVISION_NAMES if x.casefold() not in normalized]
        # Some official files use abbreviations; don't reject solely on naming drift.
        if len(missing) > 10:
            return False, f"too few recognizable subdivision names; missing {len(missing)} of 36"
    return True, f"{len(features)} features"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=OFFICIAL_GEOMETRY_URL)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(args.url, timeout=60, headers={"User-Agent": "Sanket-X/1.0"})
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ERROR: unable to download IMD subdivision geometry: {exc}")

    ok, detail = validate_geojson(payload)
    if not ok:
        raise SystemExit(f"ERROR: invalid subdivision GeoJSON: {detail}")
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"PASS: wrote {out} ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
