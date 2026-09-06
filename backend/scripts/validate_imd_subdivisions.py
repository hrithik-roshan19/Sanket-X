"""Fail-closed validator for the local IMD subdivision geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.data_sources.imd_subdivisions import IMD_SUBDIVISION_NAMES

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "geo" / "imd_subdivisions.geojson"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default=str(DEFAULT))
    args = ap.parse_args()
    path = Path(args.geojson)
    if not path.exists():
        print(f"BLOCKED: geometry not downloaded: {path}")
        print("Run scripts/fetch_imd_subdivisions.py with network access.")
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if payload.get("type") != "FeatureCollection" or not features:
        print("FAIL: invalid GeoJSON")
        return 1
    if len(features) < 36:
        print(f"FAIL: expected at least 36 features, found {len(features)}")
        return 1
    unnamed = [i for i, f in enumerate(features) if not f.get("geometry")]
    if unnamed:
        print(f"FAIL: {len(unnamed)} features have no geometry")
        return 1
    print(f"PASS: {len(features)} subdivision features with geometry")
    print("Registry target: 36 IMD meteorological subdivisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
