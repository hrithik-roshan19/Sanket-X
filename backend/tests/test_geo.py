"""geo.py region resolution."""

from __future__ import annotations

import pytest

from app.utils.geo import get_resolver
from app.utils import india_state_codes


@pytest.fixture(scope="module")
def resolver():
    return get_resolver()


@pytest.mark.parametrize(
    "name,lat,lon,expected",
    [
        ("New Delhi city", 28.6139, 77.2090, "IN-DL"),
        ("Mumbai", 19.0760, 72.8777, "IN-MH"),
        ("Leh", 34.1526, 77.5770, "IN-LA"),
        ("Chennai", 13.0827, 80.2707, "IN-TN"),
        ("Kohima", 25.6751, 94.1086, "IN-NL"),
    ],
)
def test_point_in_polygon(resolver, name, lat, lon, expected):
    m = resolver.resolve_point(lat, lon)
    assert m.region_id == expected
    assert m.method == "point_in_polygon"


def test_coastal_offset_falls_back_to_nearest(resolver):
    # a point just off the Odisha coast in the Bay of Bengal
    m = resolver.resolve_point(19.6, 86.6)
    assert m.region_id == "IN-OR"
    assert m.method == "nearest_polygon"


def test_open_ocean_is_unresolved(resolver):
    m = resolver.resolve_point(12.0, 60.0)  # mid Arabian Sea
    assert m.region_id is None
    assert m.method == "unresolved"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Orissa", "IN-OR"),
        ("Odisha", "IN-OR"),
        ("Uttaranchal", "IN-UT"),
        ("NCT of Delhi", "IN-DL"),
        ("telengana", "IN-TG"),
    ],
)
def test_name_aliases(resolver, raw, expected):
    m = resolver.resolve(name=raw)
    assert m.region_id == expected
    assert m.method == "name"


def test_name_beats_point_when_both_given(resolver):
    m = resolver.resolve(name="Kerala", lat=28.6, lon=77.2)  # wrong coords on purpose
    assert m.region_id == "IN-KL"
    assert m.method == "name"


def test_every_geojson_polygon_maps_to_a_region_id():
    r = get_resolver()
    unmapped = [n for rid, n in zip(r._region_ids, r._region_names) if rid is None]
    assert not unmapped, f"geojson features with no region_id: {unmapped}"


def test_state_code_table_covers_36():
    assert len(india_state_codes.STATES) == 36
