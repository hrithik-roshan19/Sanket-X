from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import NamedTuple

from shapely import make_valid
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from app.config import settings
from app.db.base import resolve_path
from app.utils import india_state_codes

GEOJSON_PATH = resolve_path(settings.geo_dir) / "india_states.geojson"

NEAREST_MAX_DEG = 1.5


class RegionMatch(NamedTuple):
    region_id: str | None
    region_name: str | None
    method: str


class GeoResolver:
    def __init__(self, geojson_path: Path = GEOJSON_PATH):
        data = json.loads(Path(geojson_path).read_text())
        self._geoms = []
        self._region_ids: list[str | None] = []
        self._region_names: list[str | None] = []
        for feat in data["features"]:
            props = feat.get("properties", {})
            st_code = str(props.get("st_code", "")).zfill(2)
            rec = india_state_codes.resolve_by_st_code(st_code)
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = make_valid(geom).buffer(0)
            self._geoms.append(geom)
            self._region_ids.append(rec.region_id if rec else None)
            self._region_names.append(rec.region_name if rec else props.get("st_nm"))
        self._tree = STRtree(self._geoms)

    def resolve_point(self, lat: float, lon: float) -> RegionMatch:
        if lat is None or lon is None:
            return RegionMatch(None, None, "unresolved")
        try:
            pt = Point(float(lon), float(lat))
        except (TypeError, ValueError):
            return RegionMatch(None, None, "unresolved")

        for idx in self._tree.query(pt):
            i = int(idx)
            try:
                if self._geoms[i].covers(pt):
                    return RegionMatch(self._region_ids[i], self._region_names[i], "point_in_polygon")
            except Exception:  # noqa: BLE001 - a still-broken polygon shouldn't kill ingest
                continue

        try:
            nearest_idx = int(self._tree.nearest(pt))
            if self._geoms[nearest_idx].distance(pt) <= NEAREST_MAX_DEG:
                return RegionMatch(
                    self._region_ids[nearest_idx],
                    self._region_names[nearest_idx],
                    "nearest_polygon",
                )
        except Exception:  # noqa: BLE001
            pass
        return RegionMatch(None, None, "unresolved")

    def resolve(
        self,
        name: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> RegionMatch:
        if name:
            rec = india_state_codes.resolve_by_name(name)
            if rec is not None:
                return RegionMatch(rec.region_id, rec.region_name, "name")
        if lat is not None and lon is not None:
            return self.resolve_point(lat, lon)
        return RegionMatch(None, None, "unresolved")


@functools.lru_cache(maxsize=1)
def get_resolver() -> GeoResolver:
    return GeoResolver()
