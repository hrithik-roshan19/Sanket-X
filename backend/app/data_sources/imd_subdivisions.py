"""IMD 36 meteorological subdivision registry and geometry helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMD_SUBDIVISION_NAMES = (
    "Arunachal Pradesh",
    "Assam & Meghalaya",
    "Nagaland, Manipur, Mizoram & Tripura",
    "Sub-Himalayan West Bengal & Sikkim",
    "Gangetic West Bengal",
    "Jharkhand",
    "Bihar",
    "East Uttar Pradesh",
    "West Uttar Pradesh",
    "Uttarakhand",
    "Haryana, Chandigarh & Delhi",
    "Punjab",
    "Himachal Pradesh",
    "Jammu & Kashmir and Ladakh",
    "West Rajasthan",
    "East Rajasthan",
    "West Madhya Pradesh",
    "East Madhya Pradesh",
    "Gujarat Region",
    "Saurashtra & Kutch",
    "Konkan & Goa",
    "Madhya Maharashtra",
    "Marathwada",
    "Vidarbha",
    "Chhattisgarh",
    "Odisha",
    "Coastal Andhra Pradesh",
    "Rayalaseema",
    "Telangana",
    "Tamil Nadu, Puducherry & Karaikal",
    "Coastal Karnataka",
    "North Interior Karnataka",
    "South Interior Karnataka",
    "Kerala & Mahe",
    "Lakshadweep",
    "Andaman & Nicobar Islands",
)

OFFICIAL_GEOMETRY_URL = (
    "https://mausam.imd.gov.in/imd_latest/contents/district_shapefiles/sd_boundary.json"
)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass(frozen=True)
class IMDSubdivision:
    subdivision_id: str
    name: str
    slug: str


SUBDIVISIONS = tuple(
    IMDSubdivision(f"imd-{i:02d}", name, slugify(name))
    for i, name in enumerate(IMD_SUBDIVISION_NAMES, 1)
)

BY_ID = {x.subdivision_id: x for x in SUBDIVISIONS}
BY_SLUG = {x.slug: x for x in SUBDIVISIONS}
BY_NAME = {x.name.casefold(): x for x in SUBDIVISIONS}


def geometry_path(base: Path) -> Path:
    return base / "imd_subdivisions.geojson"
